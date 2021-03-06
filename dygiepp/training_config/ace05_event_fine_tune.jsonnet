local template = import "template.libsonnet";

template.DyGIE {
  bert_model: "xlm-roberta-base",
  cuda_device: 0,
  data_paths: {
    train: "data/ace-event/collated-data/en-ar/json/ar-train.json",
    validation: "data/ace-event/collated-data/en-ar/json/dev.json",
    test: "data/ace-event/collated-data/en-ar/json/test.json",
  },
  loss_weights: {
    ner: 0.5,
    relation: 0.5,
    coref: 0.0,
    events: 1.0
  },
  target_task: "events",
  model:{
    "type": "from_archive",
    "archive_file": PATH/TO/STORE/YOUR/TRAINED/MODEL
  },
  trainer: {
    checkpointer: {
      num_serialized_models_to_keep: 3,
    },
    num_epochs: 50,
    grad_norm: 5.0,
    validation_metric: '+MEAN__trig_class_f1',
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
    },
    learning_rate_scheduler: {
      type: 'slanted_triangular'
    }
  }
}
