# Gradual Fine-Tuning for Low-Resource Domain Adaptation
Gradually  fine-tuning  in  a  multi-step  process  can  yield  sub-stantial further gains and can be applied with-out modifying the model or learning objective. This method has been demonstrated to be effective in Event Extraction and Dialogue State Tracking.

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

We have already offered all dataset at `./dygiepp/data/ace-event/collated-data/en-ar/json`, so no complex data preprocessing needed now. The amount of English and Arabic data has been indicated in the name of data file.

### Train from Scratch
```
CUDA_VISIBLE_DEVICES=`free-gpu` allennlp train "training_config/ace05_event.jsonnet" \
    --serialization-dir PATH/TO/STORE/YOUR/MODEL \
    --include-package dygie 
```

### Fine-Tuning
We show how to fine-tune a model in the DYGIE++ repo.
The DYGIE++ uses allennlp framework, so we have to add "from_archive" in the config file:
```
model:{
  "type": "from_archive",
  "archive_file": PATH/TO/STORE/YOUR/MODEL
},
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
CUDA_VISIBLE_DEVICES=`free-gpu` allennlp train "training_config/ace05_event_fine_tune.jsonnet" \
    --serialization-dir PATH/TO/STORE/YOUR/FINETUNED/MODEL \
    --include-package dygie 
```
Please see more details of config details  in `ace05_event_fine_tune.jsonnet` for fine-tuning.
### Evaluation
To evaluate a trained model:
```
CUDA_VISIBLE_DEVICES=`free-gpu` allennlp evaluate \
  PATH/FOR/TRAINED/MODEL \
  data/ace-event/collated-data/en-ar/json/test.json \
  --cuda-device 0 \
  --include-package dygie \
```
