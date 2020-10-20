#!/bin/bash
source /home/haoranxu/Anaconda/python3/bin/activate sumbt

#$ -cwd
#$ -j y -o log/base_model-attraction-500-gradual
#$ -e erlog
#$ -m eas
#$ -M hxu64@jhu.edu
#$ -l ram_free=30g,gpu=1
#$ -pe smp 4
#$ -V


START=500
DOMAIN=attraction
OUT_DIR=/export/b15/haoranxu/gradual/outputs


OUT_DIR=${OUT_DIR}/${DOMAIN}/base_model-${START}   # Dir where you want to store the model
DARA_AUG=${START}                                                     # Which augmentated data we will use, only support 4k, 2k, 500, and NONE
EPOCH=200                                                          # Training epoch
PATIENCE=15
LR=1e-4
DATA_DIR=data/multiwoz/new/${DOMAIN}                     # Dir where the data is

# Command used for training from sractch:
CUDA_VISIBLE_DEVICES=`free-gpu` python3 code/main-multislot.py --do_train --do_eval --num_train_epochs $EPOCH --data_dir $DATA_DIR \
--bert_model bert-base-uncased --do_lower_case --task_name bert-gru-sumbt --nbt rnn --output_dir $OUT_DIR \
--target_slot all --warmup_proportion 0.1 --learning_rate $LR --train_batch_size 3 --eval_batch_size 16 --distance_metric euclidean \
--tf_dir tensorboard --hidden_dim 300 --max_label_length 32 --max_seq_length 64 --max_turn_length 22 --data_augmentation $DARA_AUG \
--domain $DOMAIN --patience $PATIENCE

if [ ${START} == 4k ]
then
    for var in "2k" "500" "NONE"
    do
        FINE_TUNE_DIR=${OUT_DIR}
        DARA_AUG=${var}
        OUT_DIR=${FINE_TUNE_DIR}-${var}
        if [ ${var} == NONE ]
        then
            LR=4e-5
        elif [ ${var} == 2k ]
        then 
            LR=1e-4
        elif [ ${var} == 500 ]
        then
            LR=1e-4
        fi

        echo ${OUT_DIR} ${FINE_TUNE_DIR} ${LR}
        CUDA_VISIBLE_DEVICES=`free-gpu` python3 code/main-multislot.py --do_train --do_eval --num_train_epochs $EPOCH --data_dir $DATA_DIR \
        --bert_model bert-base-uncased --do_lower_case --task_name bert-gru-sumbt --nbt rnn --output_dir $OUT_DIR \
        --target_slot all --warmup_proportion 0.1 --learning_rate $LR --train_batch_size 3 --eval_batch_size 16 --distance_metric euclidean \
        --tf_dir tensorboard --hidden_dim 300 --max_label_length 32 --max_seq_length 64 --max_turn_length 22 --data_augmentation $DARA_AUG \
        --fine_tune $FINE_TUNE_DIR --patience $PATIENCE --domain $DOMAIN
    done
elif [ ${START} == 2k ]
then
    for var in "500" "NONE"
    do
        FINE_TUNE_DIR=${OUT_DIR}
        DARA_AUG=${var}
        OUT_DIR=${FINE_TUNE_DIR}-${var}
        if [ ${var} == NONE ]
        then
            LR=4e-5
        elif [ ${var} == 2k ]
        then 
            LR=1e-4
        elif [ ${var} == 500 ]
        then
            LR=1e-4
        fi

        echo ${OUT_DIR} ${FINE_TUNE_DIR} ${LR}
        CUDA_VISIBLE_DEVICES=`free-gpu` python3 code/main-multislot.py --do_train --do_eval --num_train_epochs $EPOCH --data_dir $DATA_DIR \
        --bert_model bert-base-uncased --do_lower_case --task_name bert-gru-sumbt --nbt rnn --output_dir $OUT_DIR \
        --target_slot all --warmup_proportion 0.1 --learning_rate $LR --train_batch_size 3 --eval_batch_size 16 --distance_metric euclidean \
        --tf_dir tensorboard --hidden_dim 300 --max_label_length 32 --max_seq_length 64 --max_turn_length 22 --data_augmentation $DARA_AUG \
        --fine_tune $FINE_TUNE_DIR --patience $PATIENCE --domain $DOMAIN
    done
elif [ ${START} == 500 ]
then
    for var in "NONE"
    do
        FINE_TUNE_DIR=${OUT_DIR}
        DARA_AUG=${var}
        OUT_DIR=${FINE_TUNE_DIR}-${var}
        if [ ${var} == NONE ]
        then
            LR=4e-5
        elif [ ${var} == 2k ]
        then 
            LR=1e-4
        elif [ ${var} == 500 ]
        then
            LR=1e-4
        fi

        echo ${OUT_DIR} ${FINE_TUNE_DIR} ${LR}
        CUDA_VISIBLE_DEVICES=`free-gpu` python3 code/main-multislot.py --do_train --do_eval --num_train_epochs $EPOCH --data_dir $DATA_DIR \
        --bert_model bert-base-uncased --do_lower_case --task_name bert-gru-sumbt --nbt rnn --output_dir $OUT_DIR \
        --target_slot all --warmup_proportion 0.1 --learning_rate $LR --train_batch_size 3 --eval_batch_size 16 --distance_metric euclidean \
        --tf_dir tensorboard --hidden_dim 300 --max_label_length 32 --max_seq_length 64 --max_turn_length 22 --data_augmentation $DARA_AUG \
        --fine_tune $FINE_TUNE_DIR --patience $PATIENCE --domain $DOMAIN
    done
else
    echo Nothing to do !
fi

