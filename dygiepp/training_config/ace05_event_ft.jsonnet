local template = import "template.libsonnet";

template.DyGIE {
  bert_model: "xlm-roberta-base",
  cuda_device: 0,
  data_paths: {
    train: "data/ace-event/collated-data/en-ar/large-filter/en500-ar-train.json",
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
  model:{
    "type": "from_archive",
    "archive_file": "/export/c12/haoranxu/dygiepp/large/en1k-ar/model.tar.gz"
  }
}
