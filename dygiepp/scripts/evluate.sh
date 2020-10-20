CUDA_VISIBLE_DEVICES=`free-gpu` allennlp evaluate \
  /export/c12/haoranxu/dygiepp/loss/en1k-500-200-NONE-ar \
  data/ace-event/collated-data/en-ar/large-filter/test.json \
  --cuda-device 0 \
  --include-package dygie \