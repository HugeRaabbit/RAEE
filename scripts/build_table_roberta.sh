#!/bin/bash

DEVICE=0
PROJECT_DIR=/RAEE
DATA_DIR=/GLUE_data/full-ft
OUTPUT_DIR=/RAEE_output
LOG_DIR=$PROJECT_DIR/logs
MODEL_NAME_OR_PATH=/roberta-large
QUERY_ENCODER_PATH=/bert_base_uncased
MAX_LENGTH=512
BATCH_SIZE=8
INDEX_TYPE=IVF*num*_HNSW32,PQ64
DATASET_SPLIT=1.0
SEED=88
TASK_LIST=('SST-2' 'sst-5' 'mr' 'cr' 'mpqa' 'subj' 'trec' 'CoLA')

for TASK in "${TASK_LIST[@]}"; do
    echo $TASK
    case $TASK in
        SST-2)
            TEMPLATE="*cls**sent_0* It was*mask*.*sep*"
            MAPPING="{\"0\":\"terrible\",\"1\":\"great\"}"
            ;;
        sst-5)
            TEMPLATE="*cls**sent_0* It was*mask*.*sep*"
            MAPPING="{\"0\":\"terrible\",\"1\":\"bad\",\"2\":\"okay\",\"3\":\"good\",\"4\":\"great\"}"
            ;;
        mr)
            TEMPLATE="*cls**sent_0* It was*mask*.*sep*"
            MAPPING="{\"0\":\"terrible\",\"1\":\"great\"}"
            ;;
        cr)
            TEMPLATE="*cls**sent_0* It was*mask*.*sep*"
            MAPPING="{\"0\":\"terrible\",\"1\":\"great\"}"
            ;;
        mpqa)
            TEMPLATE="*cls**sent_0* It was*mask*.*sep*"
            MAPPING="{\"0\":\"terrible\",\"1\":\"great\"}"
            ;;
        subj)
            TEMPLATE="*cls**sent_0* This is*mask*.*sep*"
            MAPPING="{\"0\":\"subjective\",\"1\":\"objective\"}"
            ;;
        trec)
            TEMPLATE="*cls**mask*:*sent_0**sep*"
            MAPPING="{\"0\":\"Description\",\"1\":\"Entity\",\"2\":\"Expression\",\"3\":\"Human\",\"4\":\"Location\",\"5\":\"Number\"}"
            ;;
        CoLA)
            TEMPLATE="*cls**sent_0* This is*mask*.*sep*"
            MAPPING="{\"0\":\"incorrect\",\"1\":\"correct\"}"
            ;;
    esac

    if [ ! -d $LOG_DIR ]; then
        mkdir -p $LOG_DIR
    fi

    CUDA_VISIBLE_DEVICES=$DEVICE python $PROJECT_DIR/encoder/build_tables.py \
        --task_name $TASK \
        --data_dir $DATA_DIR/$TASK/0-13 \
        --dataset_split $DATASET_SPLIT \
        --output_dir $OUTPUT_DIR \
        --use_cuda \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --query_encoder_path $QUERY_ENCODER_PATH \
        --index_type $INDEX_TYPE \
        --batch_size $BATCH_SIZE \
        --max_length $MAX_LENGTH \
        --template "$TEMPLATE" \
        --mapping $MAPPING \
        --seed $SEED \
        > $LOG_DIR/$TASK-roberta-tables.log 2>&1
done