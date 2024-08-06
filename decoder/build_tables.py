import os
import sys
import json
import time
import faiss
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if os.path.exists(os.path.join(os.path.dirname(__file__), "../utils")):
    logger.info("Importing the module `utils` from the parent directory")
    sys.path.append(os.path.join(os.path.dirname(__file__), "../utils"))
    from model_util import (
        model_loader,
    )
    from func_util import (
        format_input_text, 
        data_loader, 
        convert_token_to_id_llama,
        collect_query_embs,
        set_seed,
    )
else:
    raise ValueError("Cannot find the module `utils` in the parent directory")

def build_early_exit_table(
    template,
    data, 
    label_word_mapping,
    tokenizer, 
    max_length, 
    model, 
    num_layers, 
    cls_layer=None,
    label_ids=None,
):
    # Format the input texts and labels
    input_texts = [format_input_text(template, tokenizer, example) for example in data]
    gold_labels = [convert_token_to_id_llama(tokenizer, label_word_mapping[str(example.label)]) for example in data]

    # Tokenize the input text with padding and truncation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
    inputs = tokenizer(
        input_texts, 
        padding='max_length', 
        truncation=True if max_length > 0 else False, 
        max_length=max_length if max_length > 0 else None, 
        add_special_tokens=False,
        return_tensors="pt", 
    )
    mask_pos = [mask.tolist().index(0) - 1 if 0 in mask else len(mask) - 1 for mask in inputs["attention_mask"]]
    inputs = {k: v.to(model.device) if type(v) == torch.Tensor else v for k, v in inputs.items()}

    # Get the hidden states of the model
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    embeddings = None
    if cls_layer is not None:
        embeddings = hidden_states[cls_layer].detach().cpu().numpy()[:, 0, :]
    assert len(hidden_states) == num_layers + 1, f"The number of hidden layers ({len(hidden_states)}) does not match the model configuration ({num_layers + 1})"

    # Collect the exiting probabilities
    exit_probs = []
    for i in range(num_layers + 1):
        hidden_state = hidden_states[i]
        if i < num_layers:
            hidden_state = model.model.norm(hidden_state)
        exit_logits = model.lm_head(hidden_state)
        exit_logits = exit_logits[torch.arange(hidden_state.size(0)), mask_pos]
        exit_logits = exit_logits[:, label_ids]
        exit_probs.append(torch.softmax(exit_logits, dim=-1))
    exit_probs = torch.stack(exit_probs).permute(1, 0, 2)
    exit_max_probs, exit_token_ids = torch.max(exit_probs, dim=-1)
    exit_preds = [[label_ids[i] for i in exit_token_ids[j]] for j in range(len(data))]
    exit_max_probs = exit_max_probs.detach().cpu().numpy()
    
    # Iterate over samples in the batch
    exit_table = []
    for i in range(len(data)):
        exit_info = [(layer_idx, exit_max_probs[i][layer_idx]) for layer_idx, pred_label in enumerate(exit_preds[i]) if pred_label == gold_labels[i]]
        exit_info = np.array(exit_info)
        exit_table.append(exit_info)

    return {
        "exit_table": exit_table,
        "query_embs": embeddings,
    }

def main():
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="sst-2")
    parser.add_argument("--data_dir", type=str, default="SST-2")
    parser.add_argument('--dataset_split', type=float, default=0.2)
    parser.add_argument("--output_dir", type=str, default="RAEE")
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--model_name_or_path", type=str, default="roberta-large")
    parser.add_argument("--query_encoder_path", type=str, default=None)
    parser.add_argument("--cls_layer", type=int, default=None)
    parser.add_argument('--index_type', type=str, default='HNSW32_PQ64')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--template", type=str, default="*cls**sent_0* It was *mask*.*sep*")
    parser.add_argument("--mapping", type=str, default="{0:\"terrible\", 1:\"great\"}")
    parser.add_argument('--seed', type=int, default=88)
    args = parser.parse_args()

    args.device = torch.device("cuda") if args.use_cuda else torch.device("cpu")
    args.task_name = args.task_name.lower()
    logger.info("Arguments: %s", args)

    set_seed(args.seed)

    # Load the model and tokenizer
    config, tokenizer, model = model_loader(args.model_name_or_path)
    model_name = config.model_type
    model.eval()
    model.to(args.device)
    if args.query_encoder_path != None:
        query_config, query_tokenizer, query_model = model_loader(args.query_encoder_path, True)
        query_model_name = query_config.model_type
        query_model.to(args.device)
    else:
        query_config, query_tokenizer, query_model = None, None, None
    num_layers = 0
    if "num_hidden_layers" in config.__dict__:
        num_layers = config.num_hidden_layers
    elif "num_layers" in config.__dict__:
        num_layers = config.num_layers
    logger.info("Model: %s", model_name)
    logger.info("Encoder: %s", ("model-embdding" if args.query_encoder_path == None else query_model_name))

    # Load the data
    train_data, dev_data, test_data = data_loader(args.task_name, args.data_dir)
    logger.info("Task: %s", args.task_name)
    logger.info("Number of training data: %s", len(train_data))
    logger.info("Number of dev data: %s", len(dev_data))
    logger.info("Number of test data: %s", len(test_data))
    label_word_mapping = json.loads(args.mapping)
    label_ids = [convert_token_to_id_llama(tokenizer, token) for token in label_word_mapping.values()]

    # Merge the train and dev data for building the retrieval database
    build_data = train_data 
    num_data = int(len(build_data) * args.dataset_split)
    build_data = build_data[:num_data]
    logger.info("Number of data for building the retrieval database: %s", num_data)

    # Build the early exit table
    logger.info("Building the Exit table")
    exit_tables = []
    query_embs = []
    time_stat = {"build": [], "query": []}    
    num_batches = int(np.ceil(num_data / args.batch_size))
    for i in tqdm(range(num_batches), total=num_batches):
        batch_data = build_data[i * args.batch_size: (i + 1) * args.batch_size]
        
        # Collect the exiting information
        build_start_time = time.time()  
        results = build_early_exit_table(
            template=args.template, 
            data=batch_data, 
            label_word_mapping=label_word_mapping, 
            tokenizer=tokenizer, 
            max_length=args.max_length, 
            model=model, 
            num_layers=num_layers, 
            cls_layer=args.cls_layer if args.query_encoder_path == None else None,
            label_ids=label_ids
        )
        build_end_time = time.time()  
        build_time = build_end_time - build_start_time
        time_stat["build"].append(build_time)
        exit_tables.extend(results["exit_table"])

        if results["query_embs"] is not None:
            query_embs.append(results["query_embs"])

        query_start_time = time.time()  
        if args.query_encoder_path != None:
            assert args.cls_layer is None, "Only one encoding method can be used"
            query_embs.append(collect_query_embs(
                template=args.template,
                data=batch_data, 
                tokenizer=query_tokenizer, 
                encoder=query_model,
            ))
        query_end_time = time.time()  
        query_time = query_end_time - query_start_time 
        time_stat["query"].append(query_time)
    logger.info("Time for building the table: %s", sum(time_stat["build"]))
    logger.info("Time for encoding the query embeddings: %s", sum(time_stat["query"]))

    build_index_start_time = time.time()
    query_embs = np.concatenate(query_embs, axis=0)
    assert query_embs.shape[0] == num_data
    if 'IVF*num*' in args.index_type:
        if num_data <= 1000000: # 1M
            args.index_type = args.index_type.replace("*num*", str(int(8 * np.sqrt(num_data))))
        elif num_data <= 10000000: # 10M
            args.index_type = args.index_type.replace("*num*", str(65536))
        elif num_data <= 100000000: # 100M
            args.index_type = args.index_type.replace("*num*", str(262144))
        else:
            args.index_type = args.index_type.replace("*num*", str(1048576))
    logger.info(f'Build the index {args.index_type}')
    index = faiss.index_factory(query_embs.shape[1], args.index_type)
    logger.info(f'Train the index with {query_embs.shape[0]} samples')
    index.train(query_embs)
    logger.info(f'Add {query_embs.shape[0]} samples to the index')
    index.add(query_embs)
    build_index_end_time = time.time()
    build_index_time = build_index_end_time - build_index_start_time
    logger.info("Time of building the index: %s", build_index_time)

    args.output_dir = os.path.join(args.output_dir, f"{args.task_name}-{int(args.dataset_split*100)}")
    args.output_dir = os.path.join(args.output_dir, f"{model_name}-{query_model_name if args.query_encoder_path else model_name}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Save the tables  
    save_table_start_time = time.time()
    table_path = os.path.join(args.output_dir, f"exit-table.npy")
    with open(table_path, 'wb') as f:
        for table in exit_tables:
            np.save(f, table)
    save_table_end_time = time.time()
    save_table_time = save_table_end_time - save_table_start_time
    logger.info("Time of saving the table: %s", save_table_time)

    # Save the index
    save_index_start_time = time.time()
    index_path = os.path.join(args.output_dir, f"faiss.index")
    faiss.write_index(index, index_path)
    save_index_end_time = time.time()
    save_index_time = save_index_end_time - save_index_start_time
    logger.info("Time of saving the index: %s", save_index_time)
    logger.info("Total time: %s", sum(time_stat["build"]) + sum(time_stat["query"]) + build_index_time + save_table_time + save_index_time)

if __name__ == "__main__":
    main()