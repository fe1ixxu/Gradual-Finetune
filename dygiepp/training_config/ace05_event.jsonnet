local template = import "template2.libsonnet";

template.DyGIE {
  bert_model: "xlm-roberta-base",
  cuda_device: 0,
  data_paths: {
    train: "data/ace-event/collated-data/en-ar/large-filter/en1k-train.json",
    validation: "data/ace-event/collated-data/en-ar/large-filter/dev.json",
    test: "data/ace-event/collated-data/en-ar/large-filter/test.json",
  },
  loss_weights: {
    ner: 0.5,
    relation: 0.5,
    coref: 0.0,
    events: 1.0
  },
  target_task: "events",
  trainer: {
    checkpointer: {
      num_serialized_models_to_keep: 3,
    },
    num_epochs: 50,
    grad_norm: 5.0,
    validation_metric: '-loss',
    optimizer: {
      type: 'adamw',
      lr: 1e-3,
      weight_decay: 0.0,
      parameter_groups: [
        [
          ['_embedder'],
          {
            lr: 5e-5,
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
