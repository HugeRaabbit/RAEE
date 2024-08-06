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
        collect_embs_from_backbone,
    )
    from processors import compute_metrics_mapping
else:
    raise ValueError("Cannot find the module `utils` in the parent directory")


def compute_inverse_ratio_weights(distances, prompt_ids):
    min_valid_distance = np.min(distances[np.isfinite(distances) & (prompt_ids != -1)])
    weights = np.zeros_like(distances).astype(float)   
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            if prompt_ids[i][j] != -1 and np.isfinite(distances[i][j]):
                weights[i][j] = min_valid_distance / distances[i][j]
            else:
                weights[i][j] = 0
    return weights


def compute_exit_layers(prompt_ids, tables, weights, num_layers, topk, threshold=0.9, top_p=3):
    exit_layers = []
    for j in range(len(prompt_ids)):           
        exit_layer = num_layers
        layer_probability_sum = {} 
        for k in range(topk):
            prompt_id = prompt_ids[j][k]
            weight = weights[j][k]
            if prompt_id != -1:
                exit_info = tables[prompt_id]
                high_prob_exit_info = [item for item in exit_info if item[1] >= threshold]
                if not high_prob_exit_info:
                    high_prob_exit_info = sorted(exit_info, key=lambda x: x[1], reverse=True)[:top_p]
                for exit_item in high_prob_exit_info:
                    layer = exit_item[0]  
                    probability = exit_item[1]  
                    prob_weighted = probability * weight
                    if layer in layer_probability_sum:
                        layer_probability_sum[layer] += prob_weighted
                    else:
                        layer_probability_sum[layer] = prob_weighted                   
        if len(layer_probability_sum) > 0:
            exit_layer = max(layer_probability_sum.items(), key=lambda item: item[1])[0]
        exit_layers.append(exit_layer)
    return exit_layers

def main():
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="sst-2")
    parser.add_argument("--data_dir", type=str, default=f"SST-2")
    parser.add_argument('--dataset_split', type=float, default=0.2)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--do_exit", action="store_true")
    parser.add_argument("--model_name_or_path", type=str, default="roberta-large")
    parser.add_argument("--query_encoder_path", type=str, default= None )
    parser.add_argument("--cls_layer", type=int, default=None)
    parser.add_argument("--table_dir", type=str, default="sst-2/roberta-bert")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--template", type=str, default="*cls**sent_0* It was *mask*.*sep*")
    parser.add_argument("--mapping", type=str, default="{\"0\":\"terrible\",\"1\":\"great\"}")
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--nprobe", type=int, default=512)
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
    if args.do_exit and args.query_encoder_path != None:
        query_config, query_tokenizer, query_model = model_loader(args.query_encoder_path, True)
        query_model_name = query_config.model_type
        query_model.to(args.device)
    else:
        query_config, query_tokenizer, query_model = None, None, None
    num_layers = 0
    if "num_hidden_layers" in config.__dict__:
        num_layers = config.num_hidden_layers
    if "num_layers" in config.__dict__:
        num_layers = config.num_layers
    logger.info("Model: %s", model_name)
    if args.do_exit:
        logger.info("Encoder: %s", ("model-embdding" if args.query_encoder_path == None else query_model_name))

    # Load the data
    train_data, dev_data, test_data = data_loader(args.task_name, args.data_dir)
    logger.info("Task: %s", args.task_name)
    logger.info("Number of training data: %s", len(train_data))
    logger.info("Number of dev data: %s", len(dev_data))
    logger.info("Number of test data: %s", len(test_data))
    label_word_mapping = json.loads(args.mapping)
    label_ids = [convert_token_to_id_llama(tokenizer, token) for token in label_word_mapping.values()]
    query_data = test_data
    num_data = len(query_data)

    if args.do_exit:
        args.table_dir = os.path.join(args.table_dir, f"{args.task_name}-{int(args.dataset_split*100)}")
        args.table_dir = os.path.join(args.table_dir, f"{model_name}-{query_model_name if args.query_encoder_path else model_name}")
        # Load the index
        index_path = os.path.join(args.table_dir, f"faiss.index")
        index = faiss.read_index(index_path)
        # Load the tables
        table_path = os.path.join(args.table_dir, f"exit-table.npy")
        tables = []
        with open(table_path, 'rb') as f:
            for i in range(len(train_data)):
                tables.append(np.load(f, allow_pickle=True))
            logger.info(f"{len(tables)} entries are loaded from {table_path}")

    if args.do_exit:
        logger.info('Performing the inference with early exiting')
        time_stat = {"encoding": [], "retrieving": [], "forwarding": [], "exit_layer": []}
    else:
        logger.info('Performing the inference')
        time_stat = {"forwarding": [], "exit_layer": []}
    gold_labels = [convert_token_to_id_llama(tokenizer, label_word_mapping[str(example.label)]) for example in query_data]
    gold_labels = np.array(gold_labels)

    if args.do_exit:
        all_preds = np.zeros(num_data)
        exit_layers = []
        num_batches = int(np.ceil(num_data / args.batch_size))
        for i in tqdm(range(num_batches), total=num_batches):
            batch_data = query_data[i*args.batch_size:(i+1)*args.batch_size]
            
            start_time = time.time()
            if args.query_encoder_path != None:
                query_embs = collect_query_embs(
                    template=args.template,
                    data=batch_data, 
                    tokenizer=query_tokenizer, 
                    encoder=query_model,
                )
            else:
                query_embs = collect_embs_from_backbone(
                    template=args.template, 
                    data=batch_data, 
                    tokenizer=tokenizer, 
                    max_length=args.max_length, 
                    model=model, 
                    cls_layer=args.cls_layer,
                )
            mid_time = time.time() 
            distances, prompt_ids = index.search(query_embs, args.topk)
            weights = compute_inverse_ratio_weights(distances, prompt_ids)
            exit_layers += compute_exit_layers(prompt_ids, tables, weights, num_layers, args.topk)
            end_time = time.time()  
            time_stat["encoding"].append(mid_time - start_time)
            time_stat["retrieving"].append(end_time - mid_time)
        time_stat["exit_layer"] = exit_layers

        exit_layer_set = set(exit_layers)
        for exit_layer in exit_layer_set:
            idx = [i for i, layer in enumerate(exit_layers) if layer == exit_layer]
            num_batches = int(np.ceil(len(idx) / args.batch_size))
            temp_data = [query_data[i] for i in idx]

            exit_preds = []
            start_time = time.time()
            for i in tqdm(range(num_batches), total=num_batches):
                batch_data = temp_data[i*args.batch_size:(i+1)*args.batch_size]
                
                input_texts = [format_input_text(args.template, tokenizer, example) for example in batch_data]
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token 
                inputs = tokenizer(
                    input_texts, 
                    padding='max_length', 
                    truncation=True if args.max_length > 0 else False, 
                    max_length=args.max_length if args.max_length > 0 else None, 
                    add_special_tokens=False,
                    return_tensors="pt", 
                )
                mask_pos = [mask.tolist().index(0) - 1 if 0 in mask else len(mask) - 1 for mask in inputs["attention_mask"]]
                inputs["exit_layer"] = exit_layer
                inputs = {k: v.to(model.device) if type(v) == torch.Tensor else v for k, v in inputs.items()}
                outputs = model(**inputs)
                logits = outputs.logits
                logits = logits[torch.arange(logits.size(0)), mask_pos]
                logits = logits[:, label_ids]
                probs = torch.softmax(logits, dim=-1)
                max_probs, token_ids = torch.max(probs, axis=-1)
                preds = [label_ids[i] for i in token_ids]
                exit_preds += preds
            end_time = time.time()
            time_stat["forwarding"].append(end_time - start_time)
            logger.info(f"Exit layer {exit_layer}, avg forwarding time: {(end_time - start_time) / len(idx)}")
            for i, pred in enumerate(exit_preds):
                all_preds[idx[i]] = pred
    else:
        start_time = time.time()
        all_preds = []
        num_batches = int(np.ceil(num_data / args.batch_size))
        for i in tqdm(range(num_batches), total=num_batches):
            batch_data = query_data[i*args.batch_size:(i+1)*args.batch_size]
            
            input_texts = [format_input_text(args.template, tokenizer, example) for example in batch_data]
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token 
            inputs = tokenizer(
                input_texts, 
                padding='max_length', 
                truncation=True if args.max_length > 0 else False, 
                max_length=args.max_length if args.max_length > 0 else None, 
                add_special_tokens=False,
                return_tensors="pt", 
            )
            mask_pos = [mask.tolist().index(0) - 1 if 0 in mask else len(mask) - 1 for mask in inputs["attention_mask"]]
            inputs = {k: v.to(model.device) if type(v) == torch.Tensor else v for k, v in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            logits = logits[torch.arange(logits.size(0)), mask_pos]
            logits = logits[:, label_ids]
            probs = torch.softmax(logits, dim=-1)
            max_probs, token_ids = torch.max(probs, axis=-1)
            preds = [label_ids[i] for i in token_ids]
            all_preds.append(preds)
        end_time = time.time()
        time_stat["forwarding"].append(end_time - start_time)
        all_preds = np.concatenate(all_preds, axis=0)

    res = compute_metrics_mapping[args.task_name](args.task_name, all_preds, gold_labels)

    if args.do_exit:
        logger.info(f"avg encoding time: {sum(time_stat['encoding'])/len(query_data)}")
        logger.info(f"avg retrieving time: {sum(time_stat['retrieving'])/len(query_data)}")
    logger.info(f"avg forwarding time: {sum(time_stat['forwarding'])/len(query_data)}")
    if args.do_exit:
        logger.info(f"The average exit layer is : {sum(time_stat['exit_layer']) / len(query_data)}")
        logger.info(f"avg total time: {(sum(time_stat['encoding']) + sum(time_stat['retrieving']) + sum(time_stat['forwarding']))/len(query_data)}")
    else:
        logger.info(f"avg total time: {sum(time_stat['forwarding'])/len(query_data)}")
    logger.info('Results:')
    for k, v in res.items():
        logger.info(f'\t{k}:{v}')

if __name__ == "__main__":
    main()