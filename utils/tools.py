from transformers import BertTokenizer, BertModel
import torch.nn as nn
import os
import glob

ROBERTA_WWM_PATH = "your_path"
BERT_UNCASED_PATH = "your_path"
HUGGINGFACE_OFFLINE = True  

def _check_model_files(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型目录不存在: {model_path}")
    possible_files = [
        "pytorch_model.bin",
        "model.safetensors",
        "pytorch_model.bin.index.json", 
    ]
    
    found_files = []
    for fname in possible_files:
        fpath = os.path.join(model_path, fname)
        if os.path.exists(fpath):
            found_files.append(fname)
    
    
    shard_files = glob.glob(os.path.join(model_path, "pytorch_model-*.bin"))
    if shard_files:
        found_files.append(f"分片文件: {len(shard_files)} 个")
    
    return found_files

def _resolve_model_path(path_or_name: str):
    if os.path.exists(path_or_name):
        found = _check_model_files(path_or_name)
        if not found:
            
        return path_or_name
    else:
        raise FileNotFoundError(f"模型路径不存在: {path_or_name}")
    return path_or_name

def pretrain_bert_wwm_token():
    try:
        tokenizer = BertTokenizer.from_pretrained(_resolve_model_path(ROBERTA_WWM_PATH), local_files_only=HUGGINGFACE_OFFLINE)
        return tokenizer
    except Exception as e:
        raise

def pretrain_bert_wwm_model():
    try:
        model = BertModel.from_pretrained(_resolve_model_path(ROBERTA_WWM_PATH), local_files_only=HUGGINGFACE_OFFLINE).cuda()
        for param in model.parameters():
            param.requires_grad = False
        return model
    except Exception as e:
        raise

def pretrain_bert_uncased_token():
    try:
        tokenizer = BertTokenizer.from_pretrained(_resolve_model_path(BERT_UNCASED_PATH), local_files_only=HUGGINGFACE_OFFLINE)
        return tokenizer
    except Exception as e:
        raise

def pretrain_bert_uncased_model():
    try:
        model = BertModel.from_pretrained(_resolve_model_path(BERT_UNCASED_PATH), local_files_only=HUGGINGFACE_OFFLINE).cuda()
        for param in model.parameters():
            param.requires_grad = False
        return model
    except Exception as e:

        raise

