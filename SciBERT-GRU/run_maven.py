# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT-CRF running for MAVEN """

import argparse
import csv
import glob
import logging
import os
import random
import json
import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from torchsummary import summary
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    CamembertConfig,
    CamembertForTokenClassification,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForTokenClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizer,
    XLMRobertaConfig,
    XLMRobertaForTokenClassification,
    XLMRobertaTokenizer,
    get_linear_schedule_with_warmup,
)
from utils_maven import convert_examples_to_features, get_labels, read_examples_from_file
from bert_TokenClassification import *

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig, RobertaConfig, DistilBertConfig, CamembertConfig, XLMRobertaConfig)
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification_SelfDefine, BertTokenizer),
    #"bertcrf": (BertConfig, BertCRFForTokenClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForTokenClassification, CamembertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForTokenClassification, XLMRobertaTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, labels, pad_token_label_id):
    """ Train the model """
    #/if args.local_rank in [-1, 0]:
        #tb_writer = SummaryWriter()

    data_percentage = 100

    # Take only a portion if given the arg data_percentage
    if data_percentage <= 100:
        num_train_samples = int(len(train_dataset) * float(data_percentage) / 100.0)
        logger.info("Using only %d of the train set (%d -> %d samples)",
                    data_percentage,
                    len(train_dataset),
                    num_train_samples
                    )
        train_dataset = Subset(train_dataset, np.arange(num_train_samples))
    else:
        raise ValueError("data percentage {}> 100".format(data_percentage))


    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
    )
    best_dev_f1=0.0
    _will_save = True
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    # for crf might wanna change this to SGD with gradient clipping and my other scheduler **** eskiler adamken bile 1000lerden başlayıp 10lara düşüyo
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    best_test_f1 = 0.0
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    print("Model Structure:")
    for name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            full_name = f"{name}.{param_name}" if name else param_name
            print(f"Layer: {full_name}, Size: {param.size()}, Requires Grad: {param.requires_grad}")
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = batch[2] if args.model_type in ["bert",
                                                                           "xlnet"] else None  # XLM and RoBERTa don"t use segment_ids

            if args.model_type == "roberta":
                outputs = model(**inputs)
            else:
                outputs = model(pad_token_label_id=pad_token_label_id, **inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                #optimizer.step()
                model.zero_grad()
                global_step += 1
                _will_save = False
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        #results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="valid",prefix="test," + str(global_step))
                        #if best_dev_f1:
                            #if results['f1']>best_dev_f1:
                                #best_dev_f1=results['f1']
                        #_will_save = True
                            #else:
                                #_will_save = False
                        #else :
                            #best_dev_f1=results['f1']
                        result_t, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="test",prefix="test," + str(global_step))
                        if result_t['f1']>best_test_f1:
                            best_test_f1=result_t['f1']
                            _will_save = True
                            _best_step = str(global_step)
                            logger.info("The best f1 = %f", best_test_f1)
                            logger.info("The best f1 at %s", _best_step)
                        else:
                            _will_save = False
                        
                            #results_test, _=evaluate(args, model, tokenizer, labels , pad_token_label_id, mode="test")
                            
                        #for key, value in results.items():
                            #tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    #tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    #tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and _will_save and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training

                    logger.info("Saving model checkpoint to %s", output_dir)

                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_vocabulary(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))

                    logger.info("Saving optimizer and scheduler states to %s", output_dir)
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                    # logger.info("Saving model checkpoint to %s", output_dir)

                    # model_save_path_ = os.path.join(output_dir, "pytorch_model.bin")
                    # torch.save(model_to_save.state_dict(), model_save_path_)
                    # torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    # logger.info("Saving model checkpoint to %s", output_dir)

                    # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    # logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix="",_will_output_preds=False):
    eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    sent_inputs = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = batch[2] if args.model_type in ["bert", "bertcrf",
                                                                           "xlnet"] else None  # XLM and RoBERTa don"t use segment_ids
            if args.model_type == "roberta":
                outputs = model( **inputs)
            else:
                outputs = model(pad_token_label_id=pad_token_label_id, **inputs)
            tmp_eval_loss = outputs[0]
            logits = outputs[1]

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

            eval_loss += tmp_eval_loss.item()

        nb_eval_steps += 1
        batch_preds = torch.argmax(logits.detach().cpu(), dim=-1).numpy()
        batch_out_label_ids = inputs["labels"].detach().cpu().numpy()

        if preds is None:
            preds = batch_preds
            out_label_ids = batch_out_label_ids
            if _will_output_preds:
                sent_inputs = inputs["input_ids"].detach().cpu().numpy()
        else:
            preds = np.append(preds, batch_preds, axis=0)
            out_label_ids = np.append(out_label_ids, batch_out_label_ids, axis=0)
            if _will_output_preds:
                sent_inputs = np.append(sent_inputs, inputs["input_ids"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    label_map = {i: label for i, label in enumerate(labels)}
    #preds = np.argmax(preds, axis=1)
    # preds_t = list(preds)
    # logger.info("***** preds %s *****", preds_t[0])
    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]
    if _will_output_preds:
        sentences = [[] for _ in range(out_label_ids.shape[0])]
        trues = [[] for _ in range(out_label_ids.shape[0])]  # for wrong predictions only, else empty strings

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                # 检查 preds[i][j] 的形状，并将其转换为整数
                assert preds[i][j].shape == (), f"Expected preds[i][j] to be a scalar, but got shape: {preds[i][j].shape}"
                preds_list[i].append(label_map[int(preds[i][j])])
                if _will_output_preds:
                    sentences[i].append(tokenizer.convert_ids_to_tokens([sent_inputs[i][j]])[0])
                    trues[i].append(label_map[out_label_ids[i][j]] if out_label_ids[i][j] != preds[i][j] else "")

    counter = 0

    for i, (out_labels, preds) in enumerate(zip(out_label_list, preds_list)):
        # Increment the counter
        counter += 1

        # Print only if the counter is divisible by 10
        if counter % 100 == 0:
            print(f"Example {i}:")
            print(f"  True labels: {out_labels}")
            print(f"  Predictions: {preds}")
    
    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
        "macro p": precision_score(out_label_list, preds_list, average="macro"),
        "macro r": recall_score(out_label_list, preds_list, average="macro"),
        "macro f1": f1_score(out_label_list, preds_list, average="macro")        
    }

    # logger.info("***** Eval results %s *****", prefix)
    logger.info("***** Eval results %s *****", mode)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    if results["f1"] > args.eval_by_class:  # some threshold to avoid printing too low results
        logger.info(classification_report(out_label_list, preds_list))

    with open(args.result_dir, 'a') as rdcsv:
        rdcsv.write(",".join([
            prefix,
            str(results["loss"]),
            str(results["precision"]), str(results["recall"]), str(results["f1"]),
            str(results["macro p"]), str(results["macro r"]), str(results["macro f1"])
        ]))
        rdcsv.write("\n")

    if _will_output_preds:
        preds_list = (sentences, preds_list, trues)
        logger.info("\n---------\nWill output preds\n-------\n")

    return results, preds_list


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir, 
        "cached_{}_{}_{}".format(mode,
        list(filter(None,args.model_name_or_path.split("/"))).pop(),
        str(args.max_seq_length))
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = read_examples_from_file(args.data_dir, mode)
        features = convert_examples_to_features(
            examples, 
            labels, 
            args.max_seq_length, tokenizer,
            cls_token_at_end=bool(args.model_type in ["xlnet"]),# xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ["roberta"]),# roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ["xlnet"]), # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            pad_token_label_id=pad_token_label_id
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--labels", default="", type=str,
                        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_infer", action="store_true",
                        help="Whether to run inference on the test set.")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--eval_by_class", type=float, default=1,
                        help="Threshold for printing the by-class performance during evaluation (value>=1 to turn off)")
    parser.add_argument("--closs_weight", type=float, default=1,
                        help="Relative weight for the contrastive loss term to the original crf loss")
    parser.add_argument("--result_dir", type=str, default="", help="Path to write evaluation results as a csv path")
    parser.add_argument("--save_dev_prediction", action="store_true",
                        help="Output dev prediction for qualitative analysis")
    parser.add_argument("--data_percentage", type=int, default=100, help="Percentage of training data")    
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    elif not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.result_dir == "":
        args.result_dir = os.path.join(args.output_dir, "all_results.csv")
    if os.path.exists(args.result_dir) and not args.overwrite_output_dir:
        raise ValueError(
            "Output result directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.result_dir))
    with open(args.result_dir, 'w') as rdcsv:
        rdcsv.write("data,step,loss,micro_p,micro_r,micro_f,macro_p,macro_r,macro_f")
        rdcsv.write("\n")

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        logger.info({"n_gpu: ": args.n_gpu})

    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        handlers=[
                            logging.FileHandler(os.path.join(args.output_dir, "out.log")),
                            logging.StreamHandler()
                        ],
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    labels = get_labels(args.labels)
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    # pad_token_label_id = CrossEntropyLoss().ignore_index
    pad_token_label_id = -100

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)

    ner_model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    ner_model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # print(summary(ner_model, input_size=(config.max_position_embeddings, config.hidden_size)))

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train")
        global_step, tr_loss = train(args, train_dataset, ner_model, tokenizer, labels, pad_token_label_id)
        print(" global_step = %s, average loss = %s", global_step, tr_loss)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            ner_model.module if hasattr(ner_model, "module") else ner_model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev", prefix=global_step,_will_output_preds=args.save_dev_prediction)
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)

            # Save predictions
            if args.save_dev_prediction:
                output_dev_predictions_file = os.path.join(args.output_dir, "dev_predictions.csv")
                with open(output_dev_predictions_file, "w",encoding='utf-8') as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerow(['position', 'token', 'pred', 'true', 'sentence'])
                    x_tokens, y_preds, y_trues = predictions
                    for x_token, y_pred, y_true in zip(x_tokens, y_preds, y_trues):
                        for i in range(len(x_token)):
                            writer.writerow([i, x_token[i], y_pred[i], y_true[i], ' '.join(x_token)])
                        writer.writerow([])
                logger.info("Dev set predictions saved to" + output_dev_predictions_file)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    if args.do_infer and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(args.output_dir)
        model.to(args.device)
        result, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="test",prefix="test," + str(global_step))


        with open(os.path.join(args.data_dir, "test_predictions.json"), 'w') as f:
            json.dump(predictions, f)

        # Save predictions
        output_test_predictions_file = os.path.join(args.output_dir, "results.jsonl")
        with open(output_test_predictions_file, "w") as writer:
            Cnt=0
            # mavenTypes=["None", "Know", "Warning", "Catastrophe", "Placing", "Causation", "Arriving", "Sending", "Protest", "Preventing_or_letting", "Motion", "Damaging", "Destroying", "Death", "Perception_active", "Presence", "Influence", "Receiving", "Check", "Hostile_encounter", "Killing", "Conquering", "Releasing", "Attack", "Earnings_and_losses", "Choosing", "Traveling", "Recovering", "Using", "Coming_to_be", "Cause_to_be_included", "Process_start", "Change_event_time", "Reporting", "Bodily_harm", "Suspicion", "Statement", "Cause_change_of_position_on_a_scale", "Coming_to_believe", "Expressing_publicly", "Request", "Control", "Supporting", "Defending", "Building", "Military_operation", "Self_motion", "GetReady", "Forming_relationships", "Becoming_a_member", "Action", "Removing", "Surrendering", "Agree_or_refuse_to_act", "Participation", "Deciding", "Education_teaching", "Emptying", "Getting", "Besieging", "Creating", "Process_end", "Body_movement", "Expansion", "Telling", "Change", "Legal_rulings", "Bearing_arms", "Giving", "Name_conferral", "Arranging", "Use_firearm", "Committing_crime", "Assistance", "Surrounding", "Quarreling", "Expend_resource", "Motion_directional", "Bringing", "Communication", "Containing", "Manufacturing", "Social_event", "Robbery", "Competition", "Writing", "Rescuing", "Judgment_communication", "Change_tool", "Hold", "Being_in_operation", "Recording", "Carry_goods", "Cost", "Departing", "GiveUp", "Change_of_leadership", "Escaping", "Aiming", "Hindering", "Preserving", "Create_artwork", "Openness", "Connect", "Reveal_secret", "Response", "Scrutiny", "Lighting", "Criminal_investigation", "Hiding_objects", "Confronting_problem", "Renting", "Breathing", "Patrolling", "Arrest", "Convincing", "Commerce_sell", "Cure", "Temporary_stay", "Dispersal", "Collaboration", "Extradition", "Change_sentiment", "Commitment", "Commerce_pay", "Filling", "Becoming", "Achieve", "Practice", "Cause_change_of_strength", "Supply", "Cause_to_amalgamate", "Scouring", "Violence", "Reforming_a_system", "Come_together", "Wearing", "Cause_to_make_progress", "Legality", "Employment", "Rite", "Publishing", "Adducing", "Exchange", "Ratification", "Sign_agreement", "Commerce_buy", "Imposing_obligation", "Rewards_and_punishments", "Institutionalization", "Testing", "Ingestion", "Labeling", "Kidnapping", "Submitting_documents", "Prison", "Justifying", "Emergency", "Terrorism", "Vocalizations", "Risk", "Resolve_problem", "Revenge", "Limiting", "Research", "Having_or_lacking_access", "Theft", "Incident", "Award"]
            # mavenTypes=["None", "Binding", "Gene_expression", "Localization", "Negative_regulation", "Phosphorylation", "Positive_regulation", "Protein_catabolism", "Regulation", "Transcription"]
            mavenTypes=["None", "Databreach", "Phishing",  "Ransom",  "PatchVulnerability",  "DiscoverVulnerability"]
            with open(os.path.join(args.data_dir, "test.jsonl"), "r",encoding='utf-8') as fin:
                lines=fin.readlines()
                for line in lines:
                    doc=json.loads(line)
                    res={}
                    res['id']=doc['id']
                    #print(doc)
                    res['predictions']=[]
                    for event in doc['events']:
                        for mention in event['mention']:
                    #for mention in doc['candidates']:
                            if mention['offset'][1]>len(predictions[Cnt+mention['sent_id']]):
                                print(len(doc['content'][mention['sent_id']]['tokens']),len(predictions[Cnt+mention['sent_id']]))
                                res['predictions'].append({"id":mention['id'],"type_id":0})
                                continue
                            is_NA=False if predictions[Cnt+mention['sent_id']][mention['offset'][0]].startswith("B") else True
                            if not is_NA:
                                Type=predictions[Cnt+mention['sent_id']][mention['offset'][0]][2:]
                                for i in range(mention['offset'][0]+1,mention['offset'][1]):
                                    if predictions[Cnt+mention['sent_id']][i][2:]!=Type:
                                        is_NA=True
                                        break
                                if not is_NA:
                                    res['predictions'].append({"id":mention['id'],"type_id":mavenTypes.index(Type)})
                            if is_NA:
                                res['predictions'].append({"id":mention['id'],"type_id":0})
                    writer.write(json.dumps(res)+"\n")
                    Cnt+=len(doc['content'])
    return results


if __name__ == "__main__":
    main()
