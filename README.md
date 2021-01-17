# Gradual Fine-Tuning for Low-Resource Domain Adaptation
Gradually  fine-tuning  in  a  multi-step  process  can  yield  substantial further gains and can be applied without modifying the model or learning objective. This method has been demonstrated to be effective in Event Extraction and Dialogue State Tracking.
<div align=center><img src="https://github.com/fe1ixxu/Gradual-Finetune/blob/master/figure.png"/></div>

## 1. Event Extraction

We use [DYGIE++](https://github.com/dwadden/dygiepp) framwork to perform event extraction on the ACE 2005 corpus by considering Arabic as the target domain and English as the auxiliary domain.

Build virtual environment:
```
cd dygiepp
conda create --name dygiepp python=3.7
conda activate dygiepp
pip install -r requirements.txt
conda develop .   # Adds DyGIE to your PYTHONPATH
```

We have already offered all dataset at `./dygiepp/data/ace-event/collated-data/en-ar/json`, so no complex data preprocessing needed now. The amount of English data (out-of-domain data) has been indicated in the name of data file.

### Train from Scratch
```
allennlp train "training_config/ace05_event.jsonnet" \
    --serialization-dir PATH/TO/STORE/YOUR/MODEL \
    --include-package dygie 
```

### Fine-Tuning
We show how to fine-tune a model in the DYGIE++ repo.
The DYGIE++ uses allennlp framework, so we have to add "from_archive" in the config file:
```
model:{
  "type": "from_archive",
  "archive_file": PATH/TO/STORE/YOUR/TRAINED/MODEL
},
```
You also have to point out the new train file in the config file:
```
data_paths: {
train: "data/ace-event/collated-data/en-ar/json/TRAIN.json",
validation: "data/ace-event/collated-data/en-ar/json/dev.json",
test: "data/ace-event/collated-data/en-ar/json/test.json",
}
```
If we want to modify the learning rate to 8e-6 for XLMR and 2e-4 for the rest, something in the config file is like this:
```
optimizer: {
  type: 'adamw',
  lr: 2e-4,
  weight_decay: 0.0,
  parameter_groups: [
    [
      ['_embedder'],
      {
        lr: 8e-6,
        weight_decay: 0.01,
        finetune: true,
      },
    ],
  ],
}
```

Then, an exmaple to fine-tune a model is:
```
allennlp train "training_config/ace05_event_fine_tune.jsonnet" \
    --serialization-dir PATH/TO/STORE/YOUR/FINETUNED/MODEL \
    --include-package dygie 
```
Please see more details of config details  in `ace05_event_fine_tune.jsonnet` for fine-tuning.
### Evaluation
To evaluate a trained model:
```
allennlp evaluate \
  PATH/FOR/TRAINED/MODEL \
  data/ace-event/collated-data/en-ar/json/test.json \
  --cuda-device 0 \
  --include-package dygie \
```

## 2. Dialogue State Tracking
We utilize [SUMBT](https://github.com/SKTBrain/SUMBT) to perform DST experiments. Original SUMBT does not offer fine-tuning function, but we add it under the directory: `./SUMBT`. We use `--fine_tune FINE/TUNE/DIR` to enable the fine-tuning function. We also add `--domain` arguments to forcus on each domain, and `--data_augmentation DARA/TO/AUG # e.g., 4k, 2k, 500` to represent the amount of augmentated data. 

We have also offered all preprocessed dataset for restaurant and hotel domains here: `SUMBT/data/multiwoz/domains`. The amount of out-of-domain and has been indicated in the name of data file, e.g. train4000.tsv represents 4K out-of-domain data.

Build virtual environment
```
cd SUMBT
conda create --name sumbt python=3.6
conda activate sumbt
pip install -r requirements.txt
```

### Train from Scratch
To train the model (example of training 4k out-of-domain data with in-domain data)
```
DOMAIN=restaurant

DATA_DIR=data/multiwoz/domains/${DOMAIN}
DATA_AUG=4k
EPOCH=200                                                         
PATIENCE=15
LR=1e-4


python3 code/main-multislot.py --do_train --do_eval --num_train_epochs $EPOCH --data_dir $DATA_DIR \
--bert_model bert-base-uncased --do_lower_case --task_name bert-gru-sumbt --nbt rnn --output_dir $OUT_DIR \
--target_slot all --warmup_proportion 0.1 --learning_rate $LR --train_batch_size 3 --eval_batch_size 16 --distance_metric euclidean \
--tf_dir tensorboard --hidden_dim 300 --max_label_length 32 --max_seq_length 64 --max_turn_length 22 --data_augmentation $DARA_AUG \
--domain $DOMAIN --patience 15  

# --data_augmentation only support 4k, 2k, 500, NONE. NONE means no augmentated data
```

### Fine-Tuning
We show how to enable fine-tuning function by adding `--fine_tune` field (example of fine-tuning the previous checkpoint on only in-domain data):
```
OUT_DIR=models/model-4k-None
FINE_TUNE_DIR=models/model-4k
DATA_AUG=NONE

python3 code/main-multislot.py --do_train --do_eval --num_train_epochs $EPOCH --data_dir $DATA_DIR \
--bert_model bert-base-uncased --do_lower_case --task_name bert-gru-sumbt --nbt rnn --output_dir $OUT_DIR \
--target_slot all --warmup_proportion 0.1 --learning_rate $LR --train_batch_size 3 --eval_batch_size 16 --distance_metric euclidean \
--tf_dir tensorboard --hidden_dim 300 --max_label_length 32 --max_seq_length 64 --max_turn_length 22 --data_augmentation $DARA_AUG \
--domain $DOMAIN --patience 15 --fine_tune $FINE_TUNE_DIR

```
### Evaluation
A simple way to evaluate a trained model is just removing `--do_train`:
```
python3 code/main-multislot.py --do_eval --num_train_epochs $EPOCH --data_dir $DATA_DIR \
--bert_model bert-base-uncased --do_lower_case --task_name bert-gru-sumbt --nbt rnn --output_dir $OUT_DIR \
--target_slot all --warmup_proportion 0.1 --learning_rate $LR --train_batch_size 3 --eval_batch_size 16 --distance_metric euclidean \
--tf_dir tensorboard --hidden_dim 300 --max_label_length 32 --max_seq_length 64 --max_turn_length 22 --data_augmentation $DARA_AUG \
--domain $DOMAIN --patience 15  
```

