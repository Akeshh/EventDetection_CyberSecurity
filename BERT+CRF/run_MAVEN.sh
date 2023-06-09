python3 run_maven.py \
    --data_dir ../Data/ \ #path to the raw MAVEN data files
    --model_type bertcrf \
    --model_name_or_path bert-base-uncased \
    --output_dir ./output \ #path to dump checkpoints
    --max_seq_length 256 \
    --do_lower_case \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-5 \
    --num_train_epochs 50 \
    --save_steps 100 \
    --logging_steps 100 \
    --seed 0 \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --overwrite_output_dir
