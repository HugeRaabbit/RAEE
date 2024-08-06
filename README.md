# RAEE

## Setup
To run the code, please install the `requirements.txt` file:

```
pip install -r requirements.txt
```

## Usage
We experimented with 3 kinds of models, 2 processing steps on 8 `GLUE` tasks.
Please see the `scripts` and run the shell files to build or predict on each task.

```
bash run_[MODEL_TYPE].sh
```

The model corresponding to `run_decoder.sh` is `Llama-3-8B`;

the model corresponding to `run_encoder.sh` is `Roberta-Large`;

the model corresponding to `run_encoder_decoder.sh` is `T5-Large`.

Before running the script, remember to fill in the corresponding file path or model path in each `build_table_[MODEL_NAME].sh` or `predict_[MODEL_NAME].sh` script. 

### Note:

`MODEL_NAME_OR_PATH` should be filled in with the storage path of `Llama-3-8B`/`Roberta-Large`/`T5-Large`;

`QUERY_ENCODER_PATH` should be filled in with the storage path of `Bert_base_uncased`; 

`INDEX_TYPE`, `DATASET_SPLIT`, `SEED` are fixed values ​​and should not be modified by yourself;

`BATCH_SIZE` in `build_table_t5.sh` and `predict_t5.sh` should not be set to `1`.
