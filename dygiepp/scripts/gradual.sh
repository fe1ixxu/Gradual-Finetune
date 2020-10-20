# CUDA_VISIBLE_DEVICES=`free-gpu` allennlp train "training_config/ace05_event.jsonnet" \
#     --serialization-dir "/export/c12/haoranxu/dygiepp/large/en1k-ar" \
#     --include-package dygie 

# CUDA_VISIBLE_DEVICES=`free-gpu` allennlp train "training_config/ace05_event_ft.jsonnet" \
#     --serialization-dir "/export/c12/haoranxu/dygiepp/large/en1k-500-ar" \
#     --include-package dygie 

# CUDA_VISIBLE_DEVICES=`free-gpu` allennlp train "training_config/ace05_event_ft2.jsonnet" \
#     --serialization-dir "/export/c12/haoranxu/dygiepp/large/en1k-500-200-ar" \
#     --include-package dygie 

# CUDA_VISIBLE_DEVICES=`free-gpu` allennlp train "training_config/ace05_event_ft3.jsonnet" \
#     --serialization-dir "/export/c12/haoranxu/dygiepp/large/en1k-500-200-NONE-ar" \
#     --include-package dygie \
#     --override "{'trainer': {'optimizer': {lr: 4e-4} } }"


# CUDA_VISIBLE_DEVICES=`free-gpu` allennlp train "training_config/ace05_event.jsonnet" \
#     --serialization-dir "/export/c12/haoranxu/dygiepp/loss/en1k-ar" \
#     --include-package dygie 

CUDA_VISIBLE_DEVICES=`free-gpu` allennlp train "training_config/ace05_event_tr.jsonnet" \
    --serialization-dir "/export/c12/haoranxu/dygiepp/loss/en1k-500-ar" \
    --include-package dygie 

CUDA_VISIBLE_DEVICES=`free-gpu` allennlp train "training_config/ace05_event_tr2.jsonnet" \
    --serialization-dir "/export/c12/haoranxu/dygiepp/loss/en1k-500-200-ar" \
    --include-package dygie 
# --override "{'trainer': {'optimizer': {lr: 4e-4, parameter_groups: [[['_embedder'], { lr: 1e-5, weight_decay: 0.01, finetune: true, }, ],],} } }"

CUDA_VISIBLE_DEVICES=`free-gpu` allennlp train "training_config/ace05_event_tr3.jsonnet" \
    --serialization-dir "/export/c12/haoranxu/dygiepp/loss/en1k-500-200-NONE-ar" \
    --include-package dygie 

# --override "{'trainer': {'optimizer': {lr: 2e-4, parameter_groups: [[['_embedder'], { lr: 8e-6, weight_decay: 0.01, finetune: true, }, ],],} } }"

#validation_metric: '+MEAN__arg_class_f1'

