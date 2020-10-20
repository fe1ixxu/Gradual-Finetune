# CUDA_VISIBLE_DEVICES=`free-gpu` allennlp train "training_config/ace05_event.jsonnet" \
#     --serialization-dir "/export/c12/haoranxu/dygiepp/large/en1k-ar" \
#     --include-package dygie 

# CUDA_VISIBLE_DEVICES=`free-gpu` allennlp train "training_config/ace05_event_ft.jsonnet" \
#     --serialization-dir "/export/c12/haoranxu/dygiepp/large/en1k-500-ar" \
#     --include-package dygie 

CUDA_VISIBLE_DEVICES=`free-gpu` allennlp train "training_config/ace05_event_ft2.jsonnet" \
    --serialization-dir "/export/c12/haoranxu/dygiepp/large/en1k-500-200-ar" \
    --include-package dygie 

CUDA_VISIBLE_DEVICES=`free-gpu` allennlp train "training_config/ace05_event_ft3.jsonnet" \
    --serialization-dir "/export/c12/haoranxu/dygiepp/large/en1k-500-200-NONE-ar" \
    --include-package dygie 