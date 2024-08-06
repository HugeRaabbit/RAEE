import os
import sys
import logging
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if os.path.exists(os.path.join(os.path.dirname(__file__), "../models")):
    logger.info("Importing models from local path")
    sys.path.append(os.path.join(os.path.dirname(__file__), "../models"))
    from prompt_roberta import RobertaForPromptFinetuning 
    from prompt_llama import LlamaForPromptFinetuning
    from prompt_t5 import T5ForPromptFinetuning
   
else:
    raise ValueError("Models not found in the specified path")

bert_models_mapping = {
    "roberta": RobertaForPromptFinetuning,
    "llama": LlamaForPromptFinetuning,
    "t5": T5ForPromptFinetuning
}

def model_loader(model_name, sentence=False, default=False):
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side='left', use_fast=False)
    if sentence:
        model = SentenceTransformer(model_name)
    else:
        if not default and config.model_type in bert_models_mapping:
            model = bert_models_mapping[config.model_type].from_pretrained(model_name, config=config)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
    model.eval()
    return config, tokenizer, model


def block_remove(model, model_name, kill_list):
    print(f"Removing layers... {kill_list}")
    if 'opt' in model_name:
        return block_remove_opt(model, kill_list)
    elif 'llama' in model_name:
        return block_remove_llama(model, kill_list)
    else:
        return None

def block_remove_opt(model, kill_list):

    kill_list.sort()
    
    while(len(kill_list)>0):
        
        del model.model.decoder.layers[kill_list[0]]
        del kill_list[0]
        for i in range(len(kill_list)):
            kill_list[i] -= 1
    
    return model

def block_remove_llama(model, kill_list):

    kill_list.sort()

    while(len(kill_list)>0):
        
        del model.model.layers[kill_list[0]]
        del kill_list[0]
        for i in range(len(kill_list)):
            kill_list[i] -= 1

    for i in range(len(model.model.layers)):
        model.model.layers[i].self_attn.layer_idx = i

    return model
