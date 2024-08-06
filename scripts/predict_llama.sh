#!/bin/bash

DEVICE=0
PROJECT_DIR=/RAEE
DATA_DIR=/GLUE_data/full-ft
TABLE_DIR=/RAEE_output
LOG_DIR=$PROJECT_DIR/logs
MODEL_NAME_OR_PATH=/llama-3-8b
QUERY_ENCODER_PATH=/bert_base_uncased
BATCH_SIZE=8
MAX_LENGTH=512   
DATASET_SPLIT=1.0
TOPK=12
NPROBE=512
SEED=88
TASK_LIST=('SST-2' 'sst-5' 'mr' 'cr' 'mpqa' 'subj' 'trec' 'CoLA')

for TASK in "${TASK_LIST[@]}"; do
    echo $TASK
    case $TASK in
        SST-2)
            TEMPLATE="What is the sentiment of the sentence '*sent_0*'? Print negative or positive. The answer is "
            MAPPING="{\"0\":\"negative\",\"1\":\"positive\"}"
            ;;
        sst-5)
            TEMPLATE="What is the sentiment of the sentence '*sent_0*'? Print terrible, bad, okay, good or great. The answer is " 
            MAPPING="{\"0\":\"terrible\",\"1\":\"bad\",\"2\":\"okay\",\"3\":\"good\",\"4\":\"great\"}"
            ;;
        mr)
            TEMPLATE="What is the sentiment of the sentence '*sent_0*'? Print negative or positive. The answer is "
            MAPPING="{\"0\":\"negative\",\"1\":\"positive\"}"
            ;;
        cr)
            TEMPLATE="What is the sentiment of the sentence '*sent_0*'? Print negative or positive. The answer is "
            MAPPING="{\"0\":\"negative\",\"1\":\"positive\"}"
            ;;
        mpqa)
            TEMPLATE="What is the sentiment of the sentence '*sent_0*'? Print negative or positive. The answer is "
            MAPPING="{\"0\":\"negative\",\"1\":\"positive\"}"
            ;;
        subj)
            TEMPLATE="What is the subjectivity of the sentence '*sent_0*'? Print subjective or objective. The answer is "
            MAPPING="{\"0\":\"subjective\",\"1\":\"objective\"}"
            ;;
        trec)
            TEMPLATE="Print the category for the sentence '*sent_0*': Description, Entity, Expression, Human, Location or Number. The answer is"
            MAPPING="{\"0\":\"Description\",\"1\":\"Entity\",\"2\":\"Expression\",\"3\":\"Human\",\"4\":\"Location\",\"5\":\"Number\"}"
            ;;
        CoLA)
            TEMPLATE="Is the sentence '*sent_0*' grammatically acceptable? Print no or yes. The answer is "
            MAPPING="{\"0\":\"no\",\"1\":\"yes\"}"
            ;;
    esac
    
    CUDA_VISIBLE_DEVICES=$DEVICE python $PROJECT_DIR/decoder/predict.py \
        --task_name $TASK \
        --data_dir $DATA_DIR/$TASK/0-13 \
        --dataset_split $DATASET_SPLIT \
        --use_cuda \
        --do_exit \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --query_encoder_path $QUERY_ENCODER_PATH \
        --table_dir $TABLE_DIR \
        --batch_size $BATCH_SIZE \
        --max_length $MAX_LENGTH \
        --template "$TEMPLATE" \
        --mapping $MAPPING \
        --topk $TOPK \
        --nprobe $NPROBE \
        --seed $SEED \
        > $LOG_DIR/$TASK-llama-early-exit-results.log 2>&1

    CUDA_VISIBLE_DEVICES=$DEVICE python $PROJECT_DIR/decoder/predict.py \
        --task_name $TASK \
        --data_dir $DATA_DIR/$TASK/0-13 \
        --dataset_split $DATASET_SPLIT \
        --use_cuda \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --batch_size $BATCH_SIZE \
        --max_length $MAX_LENGTH \
        --template "$TEMPLATE" \
        --mapping $MAPPING \
        --seed $SEED \
        > $LOG_DIR/$TASK-llama-origin-results.log 2>&1
done