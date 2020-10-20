#!/bin/bash
source /home/haoranxu/Anaconda/python3/bin/activate sumbt

#$ -cwd
#$ -j y -o log/base_model-taxi-one-step
#$ -e erlog
#$ -m eas
#$ -M hxu64@jhu.edu
#$ -l ram_free=6g,gpu=1
#$ -pe smp 4
#$ -V



DOMAIN=hotel
# rm -rf /export/b15/haoranxu/gradual/outputs/${DOMAIN}/base_model-4k-500

OUT_DIR=/export/b15/haoranxu/gradual/outputs/${DOMAIN}/base_model # Dir where you want to store the model
FINE_TUNE_DIR=/export/b15/haoranxu/gradual/outputs/${DOMAIN}/v2_base_model-4k-2k-500    # Dir where the pretrained model is (only used for fine-tuning command)
DARA_AUG=NONE                                                     # Which augmentated data we will use, only support 4k, 2k, 500, and NONE
EPOCH=200                                                          # Training epoch
PATIENCE=15
LR=1e-4

DATA_DIR=data/multiwoz/new/${DOMAIN}                     # Dir where the data is

# Command used for training from sractch:
CUDA_VISIBLE_DEVICES=`free-gpu` python3 code/main-multislot.py  --do_eval --num_train_epochs $EPOCH --data_dir $DATA_DIR \
--bert_model bert-base-uncased --do_lower_case --task_name bert-gru-sumbt --nbt rnn --output_dir $OUT_DIR \
--target_slot all --warmup_proportion 0.1 --learning_rate $LR --train_batch_size 3 --eval_batch_size 16 --distance_metric euclidean \
--tf_dir tensorboard --hidden_dim 300 --max_label_length 32 --max_seq_length 64 --max_turn_length 22 --data_augmentation $DARA_AUG \
--domain $DOMAIN --patience $PATIENCE

# OUT_DIR=/export/b15/haoranxu/gradual/outputs/${DOMAIN}/base_model-4k-NONE  # Dir where you want to store the model
# FINE_TUNE_DIR=/export/b15/haoranxu/gradual/outputs/${DOMAIN}/base_model-4k    # Dir where the pretrained model is (only used for fine-tuning command)
# DARA_AUG=NONE                                                     # Which augmentated data we will use, only support 4k, 2k, 500, and NONE
# EPOCH=200                                                          # Training epoch
# PATIENCE=15
# LR=4e-5

# DATA_DIR=data/multiwoz/new/${DOMAIN}                     # Dir where the data is


# # Command used for fine-tuning:
# CUDA_VISIBLE_DEVICES=`free-gpu` python3 code/main-multislot.py --do_train --do_eval --num_train_epochs $EPOCH --data_dir $DATA_DIR \
# --bert_model bert-base-uncased --do_lower_case --task_name bert-gru-sumbt --nbt rnn --output_dir $OUT_DIR \
# --target_slot all --warmup_proportion 0.1 --learning_rate $LR --train_batch_size 3 --eval_batch_size 16 --distance_metric euclidean \
# --tf_dir tensorboard --hidden_dim 300 --max_label_length 32 --max_seq_length 64 --max_turn_length 22 --data_augmentation $DARA_AUG \
# --fine_tune $FINE_TUNE_DIR --patience $PATIENCE --domain $DOMAIN


# OUT_DIR=/export/b15/haoranxu/gradual/outputs/${DOMAIN}/base_model-2k-NONE  # Dir where you want to store the model
# FINE_TUNE_DIR=/export/b15/haoranxu/gradual/outputs/${DOMAIN}/base_model-2k    # Dir where the pretrained model is (only used for fine-tuning command)
# DARA_AUG=NONE                                                     # Which augmentated data we will use, only support 4k, 2k, 500, and NONE
# EPOCH=200                                                          # Training epoch
# PATIENCE=15
# LR=4e-5

# DATA_DIR=data/multiwoz/new/${DOMAIN}                     # Dir where the data is


# # Command used for fine-tuning:
# CUDA_VISIBLE_DEVICES=`free-gpu` python3 code/main-multislot.py --do_train --do_eval --num_train_epochs $EPOCH --data_dir $DATA_DIR \
# --bert_model bert-base-uncased --do_lower_case --task_name bert-gru-sumbt --nbt rnn --output_dir $OUT_DIR \
# --target_slot all --warmup_proportion 0.1 --learning_rate $LR --train_batch_size 3 --eval_batch_size 16 --distance_metric euclidean \
# --tf_dir tensorboard --hidden_dim 300 --max_label_length 32 --max_seq_length 64 --max_turn_length 22 --data_augmentation $DARA_AUG \
# --fine_tune $FINE_TUNE_DIR --patience $PATIENCE --domain $DOMAIN

# OUT_DIR=/export/b15/haoranxu/gradual/outputs/${DOMAIN}/base_model-500-NONE  # Dir where you want to store the model
# FINE_TUNE_DIR=/export/b15/haoranxu/gradual/outputs/${DOMAIN}/base_model-500    # Dir where the pretrained model is (only used for fine-tuning command)
# DARA_AUG=NONE                                                     # Which augmentated data we will use, only support 4k, 2k, 500, and NONE
# EPOCH=200                                                          # Training epoch
# PATIENCE=15
# LR=4e-5

# DATA_DIR=data/multiwoz/new/${DOMAIN}                     # Dir where the data is


# # Command used for fine-tuning:
# CUDA_VISIBLE_DEVICES=`free-gpu` python3 code/main-multislot.py --do_train --do_eval --num_train_epochs $EPOCH --data_dir $DATA_DIR \
# --bert_model bert-base-uncased --do_lower_case --task_name bert-gru-sumbt --nbt rnn --output_dir $OUT_DIR \
# --target_slot all --warmup_proportion 0.1 --learning_rate $LR --train_batch_size 3 --eval_batch_size 16 --distance_metric euclidean \
# --tf_dir tensorboard --hidden_dim 300 --max_label_length 32 --max_seq_length 64 --max_turn_length 22 --data_augmentation $DARA_AUG \
# --fine_tune $FINE_TUNE_DIR --patience $PATIENCE --domain $DOMAIN