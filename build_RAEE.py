import os
import json
import faiss
import torch
import argparse
import numpy as np
import random
import transformers
from tqdm import tqdm

def build_early_exit_table(
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
    input_texts = [format_input_text(template, tokenizer, example) for example in data]
    inputs = tokenizer(
        input_texts, 
        padding='max_length', 
        max_length=max_length if max_length > 0 else None, 
    )
    outputs = model(**inputs, output_hidden_states=True)
    embeddings = hidden_states[0].detach().cpu().numpy()
    embeddings = embeddings[:, 0, :]
    for i in range(num_layers + 1):
        hidden_state = hidden_states[i]
        inter_logits = model.lm_head(hidden_state[torch.arange(hidden_state.size(0))])
        inter_probs.append(torch.softmax(inter_logits, dim=-1))
    inter_probs = torch.stack(inter_probs).permute(1, 0, 2)
    early_exit_max_probs, early_exit_token_ids = torch.max(inter_probs, dim=-1)
    early_exit_predictions = [[model.masked_logits.label_word_list[i] for i in early_exit_token_ids[j]] for j in range(batch_size)]
    early_exit_table = {}
    for j in range(batch_size):
        early_exit_table_i = [
            [
                i,
                early_exit_max_probs[j][i],
            ]
            for i, pred_id in enumerate(early_exit_predictions[j]) if pred_id == label_ids[j]
        ]

    if return_embs:
        return early_exit_table, embeddings
    else:
        return early_exit_table

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
    print("Backbone model name: ", model_name)
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
    build_data = train_data + dev_data
    
    for i in tqdm(range(num_batches), total=num_batches):
        data = build_data_split[i * args.batch_size: (i + 1) * args.batch_size]
        results = build_early_exit_table(
            prompt_id_base=i*args.batch_size, 
            template=args.template, 
            data=data, 
            label_word_mapping=label_word_mapping, 
            tokenizer=tokenizer, 
            max_length=args.max_length, 
            model=model, 
            num_layers=num_layers, 
            mask_pos=(model_name in bert_models_mapping),
            return_embs=(args.query_encoder_path == None),
        )
        table = results[:1]
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
            query_embs = results[1]
        tables.update(table)
        query_embeddings.append(query_embs)
      
    output_path = os.path.join(args.output_dir, f"{args.task_name}_{args.dataset_split}_state_scores.json")
    with open(output_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(tables, jsonfile, ensure_ascii=False, indent=4)
      
    query_embeddings = np.concatenate(query_embeddings, axis=0)
    index = faiss.index_factory(query_embeddings.shape[1], args.index_type)
    index.train(query_embeddings)
    index.add(query_embeddings)
    index_path = os.path.join(args.output_dir, f"{args.task_name}_{args.dataset_split}_{args.index_type}.index")
    faiss.write_index(index, index_path)

if __name__ == "__main__":
    main()
