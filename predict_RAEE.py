import os
import json
import faiss
import torch
import argparse
import numpy as np
import random
import transformers
from tqdm import tqdm

def compute_inverse_ratio_weights(distances, prompt_ids):
    min_valid_distance = np.min(distances[np.isfinite(distances) & (prompt_ids != -1)])
    weights = np.zeros_like(distances).astype(float)
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            if prompt_ids[i][j] != -1 and np.isfinite(distances[i][j]):
                weights[i][j] = max_valid_distance / distances[i][j]
            else:
                weights[i][j] = 0
    return weights

def build_early_exit_table(
    EarlyExit_layer,
    prompt_id_base, 
    template,
    data, 
    label_word_mapping,
    tokenizer, 
    max_length, 
    model, 
    num_layers, 
    mask_pos=False,
    return_embs=False,
):
    input_texts = [format_input_text(template, tokenizer, data)]
    gold_labels = [convert_token_to_id(tokenizer, labels[i]) for i in range(len(labels))]
    inputs = tokenizer(
        input_texts, 
        padding='max_length', 
        max_length=max_length if max_length > 0 else None, 
    )
    outputs = model(**inputs, output_hidden_states=True)
    
    final_hiddens_state = outputs.hidden_states[EarlyExit_layer]
    final_hiddens_state = final_hiddens_state[torch.arange(final_hiddens_state.size(0)), mask_pos]
    final_logits = model.lm_head(final_hiddens_state)
    final_logits = model.masked_logits(final_logits)
    final_probs = torch.softmax(final_logits, dim=-1)
    final_max_probs, final_token_ids = torch.max(final_probs, axis=-1)
    final_predictions = [model.masked_logits.label_word_list[i] for i in final_token_ids]

    origin_hiddens_state = outputs.hidden_states[-1]
    origin_hiddens_state = origin_hiddens_state[torch.arange(origin_hiddens_state.size(0)), mask_pos]
    origin_logits = model.lm_head(origin_hiddens_state)
    origin_logits = model.masked_logits(origin_logits)
    origin_probs = torch.softmax(origin_logits, dim=-1)
    origin_max_probs, origin_token_ids = torch.max(origin_probs, axis=-1)
    origin_predictions = [model.masked_logits.label_word_list[i] for i in origin_token_ids]

    return final_predictions, gold_labels, origin_predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True) ### path of your backbone model
    parser.add_argument("--task_name", type=str, required=True) ### name of Task
    parser.add_argument("--data_dir", type=str, required=True) ### path of your Task's data
    parser.add_argument("--table_dir", type=str, required=True) ### path of your tables from the building process
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--template", type=str, required=True) ### template of prompt
    parser.add_argument("--mapping", type=str, required=True) ### label word mapping
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--query_encoder_path", type=str, required=True) ### path of your encoder model
    parser.add_argument("--query_device", type=str, required=True)
    parser.add_argument('--index_path', type=str, required=True) ### path of your index from the building process
    parser.add_argument('--index_type', type=str, required=True) ### type of your building index
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--nprobe", type=int, required=True)
    parser.add_argument('--dataset_split', type=float, default=1)
    args = parser.parse_args()

    config, tokenizer, model = model_loader(args.model_name_or_path)
    model_name = config.model_type
    model.to(args.device)
    if args.query_encoder_path != None:
        query_config, query_tokenizer, query_model = model_loader(args.query_encoder_path, True)
        query_encoder_name = query_config.model_type
        query_model.to(args.query_device)
        print("Query encoder name: ", query_encoder_name)
    else:
        query_config = None
        query_tokenizer = None
        query_model = None
    train_data, dev_data, test_data = data_loader(args.task_name, args.data_dir)
    query_data = test_data

    index_path = os.path.join(args.index_path, f"{args.task_name}_{args.dataset_split}_{args.index_type}.index")
    index = faiss.read_index(index_path)
    table_path = os.path.join(args.table_dir, f"{args.task_name}_{args.dataset_split}_state_scores.json")
    with open(table_path, 'r', encoding='utf-8') as jsonfile:
        tables = json.load(jsonfile)

    for i in tqdm(range(num_batches), total=num_batches):
        data = query_data[i*args.batch_size:(i+1)*args.batch_size]
        if args.query_encoder_path != None:
            query_embs = collect_query_embs(
                prompt_id_base=i*args.batch_size, 
                template=args.template,
                data=data, 
                tokenizer=query_tokenizer, 
                max_length=args.max_length, 
                encoder=query_model,
            )
        else:
            inputs = prepare_inputs(
                template=args.template,
                data=data, 
                label_word_mapping=label_word_mapping,
                tokenizer=tokenizer, 
                max_length=args.max_length, 
                model=model, 
                mask_pos=(model_name in bert_models_mapping),
            )
            query_embs = model.embeddings(
                input_ids=inputs["input_ids"],
                position_ids=inputs["position_ids"],
                token_type_ids=inputs["token_type_ids"],
                inputs_embeds=None,
                past_key_values_length=None,
            )
            
        distances, prompt_ids = index.search(query_embs, args.topk)
        weights = compute_inverse_ratio_weights(distances, prompt_ids)
        
        for j in range(len(data)):
            EarlyExit_layer = None
            max_probability = -1 
            for k in range(args.topk):
                prompt_id = prompt_ids[j][k]
                distance = distances[j][k]
                weight = weights[j][k] 
                if prompt_id != -1:
                    exit_info = tables[f'prompt-{str(prompt_id)}']
                    for exit_item in exit_info:
                        layer = exit_item[0] 
                        probability = exit_item[1] 
                    high_prob_exit_info = [item for item in exit_info if item[1] >= threshold]
                    if not high_prob_exit_info: 
                        high_prob_exit_info = sorted(exit_info, key=lambda x: x[1], reverse=True)[:top_p]
                    for exit_item in high_prob_exit_info:
                        layer = exit_item[0] 
                        probability = exit_item[1]  
                    for exit_item in high_prob_exit_info:
                        layer = exit_item[0] 
                        probability = exit_item[1] 
                        prob_weighted = probability * weight
                        if layer in layer_probability_sum:
                            layer_probability_sum[layer] += prob_weighted
                        else:
                            layer_probability_sum[layer] = prob_weighted                   
                max_layer = max(layer_probability_sum.items(), key=lambda item: item[1])[0]
            EarlyExit_layer = max_layer                      
            EarlyExit_layers.append(EarlyExit_layer)
        total_EarlyExit_layers.append(EarlyExit_layers)

        for s in range(len(data)):
            results = build_early_exit_table(
                    EarlyExit_layers[s],
                    prompt_id_base=i*args.batch_size, 
                    template=args.template, 
                    data=data[s], 
                    label_word_mapping=label_word_mapping, 
                    tokenizer=tokenizer, 
                    max_length=args.max_length, 
                    model=model, 
                    num_layers=num_layers, 
                    mask_pos=(model_name in bert_models_mapping),
                    return_embs=(args.query_encoder_path == None),
                )
            final_preds, gold_labels, origin_preds = results[:3]
            all_preds_part = [final_preds] if all_preds_part is None else all_preds_part + [final_preds]
            all_labels_part = [gold_labels] if all_labels_part is None else all_labels_part + [gold_labels]
            origin_preds_part = [origin_preds] if origin_preds_part is None else origin_preds_part + [origin_preds]

    all_preds = np.concatenate(all_preds_part, axis=0)
    all_labels = np.concatenate(all_labels_part, axis=0)
    origin_preds = np.concatenate(origin_preds_part, axis=0)

    res = compute_metrics_mapping[args.task_name](args.task_name, all_preds, all_labels)
    origin_res = compute_metrics_mapping[args.task_name](args.task_name, origin_preds, all_labels)
  
    total_sum = sum(num for sublist in total_EarlyExit_layers for num in sublist)
    total_count = sum(len(sublist) for sublist in total_EarlyExit_layers)
    average = total_sum / total_count
    print(f"The average EarlyExit_layer is : {average}")
        
    print('Results of ranked_ensemble model:')
    for k, v in res.items():
        print(f'\t{k}:{v}')

    print('Results of origin model:')
    for k, v in origin_res.items():
        print(f'\t{k}:{v}')



if __name__ == "__main__":
    main()
