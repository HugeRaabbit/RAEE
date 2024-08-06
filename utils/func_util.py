import torch
import random
import transformers
import numpy as np
from processors import processors_mapping

def data_loader(task_name, data_dir):
    processor = processors_mapping[task_name]
    train_data = processor.get_train_examples(data_dir)
    dev_data = processor.get_dev_examples(data_dir)    
    test_data = processor.get_test_examples(data_dir)
    return train_data, dev_data, test_data

def format_input_text(template, tokenizer, example):
    input_text = template.replace("*sent_0*", str(example.text_a))
    if example.text_b is not None:
        input_text = input_text.replace("*sent_1*", example.text_b)
    if "*mask*" in input_text:
        input_text = input_text.replace("*mask*", tokenizer.mask_token)
    if "*eos*" in input_text:
        input_text = input_text.replace("*eos*", tokenizer.eos_token)
    if "*cls*" in input_text:
        input_text = input_text.replace("*cls*", tokenizer.cls_token)
    if "*sep*" in input_text:
        input_text = input_text.replace("*sep*", tokenizer.sep_token)
    return input_text

def convert_token_to_id(tokenizer, token):
    if token[0] not in ['<', '[', '.', ',']:
        # Make sure space+word is in the vocabulary
        assert len(tokenizer.tokenize(' ' + token)) == 1
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' ' + token)[0])
    else:
        return tokenizer.convert_tokens_to_ids(token)
    
def convert_token_to_id_llama(tokenizer, token):
    if token[0] not in ['<', '[', '.', ',']:
        # Make sure space+word is in the vocabulary
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token)[0])
    else:
        return tokenizer.convert_tokens_to_ids(token)

def collect_query_embs(
    template,
    data, 
    tokenizer, 
    encoder,
):
    input_texts = [format_input_text(template, tokenizer, example) for example in data]
    query_embeddings = encoder.encode(input_texts, show_progress_bar=False)
    return query_embeddings

def collect_embs_from_backbone(
    template,
    data, 
    tokenizer, 
    max_length,
    model,
    cls_layer,
):
    input_texts = [format_input_text(template, tokenizer, example) for example in data]

    inputs = tokenizer(
        input_texts, 
        padding='max_length', 
        truncation=True if max_length > 0 else False, 
        max_length=max_length if max_length > 0 else None, 
        add_special_tokens=False,
        return_tensors="pt", 
    )
    inputs["exit_layer"] = cls_layer
    inputs = {k: v.to(model.device) if type(v) == torch.Tensor else v for k, v in inputs.items()}
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()

def set_seed(seed):
    transformers.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
