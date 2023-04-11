# EventDetection_CyberSecurity
Event Detection Baselines for CASIE  CyberSecurity dataset in MAVEN format. We would use the Transformer based models for developing.

## Data
The training set, development set, and testing set are stored in the MAVEN format in the **.jsonl** files and are located in the **Data** folder. Prior to training, it is necessary to  **unzip the train.zip** to obtain the train.jsonl.

## Environment
I suggest using the following environment configuration to avoid  version conflict issues.
- Python == 3.6.10
- Transformer == 2.6.0
- sklearn == 0.20.2
- seqeval


## Usage
The codes are in the `BERT+CRF` folder.
### On Linux
1.  Run  `run_MAVEN.sh`  for training and evaluation on the devlopment set.
2.  Run  `run_MAVEN_infer.sh`  to get predictions on the test set (dumped to  `OUTPUT_PATH/results.jsonl`).
### On Windows
1.  Run  `run_MAVEN.txt`  for training and evaluation on the devlopment set.
2.  Run  `run_MAVEN_infer.txt`  to get predictions on the test set (dumped to  `OUTPUT_PATH/results.jsonl`).

Hint: ./output/ is the folder sroring your model checkpoints.

See the two scripts for more details.

## Change model architecture
The method for changing model architecture is based on classifier.
### Use CRF as classifier
You can add codes in bert_crf.py to change model architecture.
### Use FFNN as classifier
You can add codes in bert_TokenClassification.py to change model architecture.

See the two scripts for more details.

## Usable Pretrained Language Model (PLM)
Before you use RoBERTa, you should update your `run_maven.py` in BERT folder, the latest version of `run_maven.py` is updated at 11/4/2023 23:00
### BERT:
- bert-base-uncased

- allenai/scibert_scivocab_uncased

- "../NewsBERT/"

### RoBERTa:
- "../roberta-base/"

- "../roberta-large/"

- "../SecRoBERTa/"

Models in " " mean that you need to upload the PLMs from local

change --model_type e.g., `roberta` and --model_name_or_path e.g., `"../SecRoBERTa/"` to use different PLMs.
