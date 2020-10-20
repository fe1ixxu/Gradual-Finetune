# USAGE: `bash train.sh [config_name]`
#
# The `config_name` is the name of one of the `jsonnet` config files in the
# `training_config` directory, for instance `scierc`. The result of training
# will be placed under `models/[config_name]`.

CUDA_VISIBLE_DEVICES=`free-gpu` allennlp train "training_config/ace05_event_tr3.jsonnet" \
    --serialization-dir "/export/c12/haoranxu/dygiepp/trigger/en1k-500-200-NONE-ar-2-trigger" \
    --include-package dygie 