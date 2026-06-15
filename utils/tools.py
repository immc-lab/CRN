from transformers import BertTokenizer, BertModel
import torch.nn as nn
import os
import glob

# 固定离线权重路径，避免服务器无外网时拉取失败
ROBERTA_WWM_PATH = "/sda/data/caozhiyang/MFSVFND/roberta_wwm"
BERT_UNCASED_PATH = "/sda/data/caozhiyang/MFSVFND/bert-base-uncased"
HUGGINGFACE_OFFLINE = True  # 若需要联网加载，可改为 False

def _check_model_files(model_path: str):
    """
    检查模型目录中的文件，返回找到的模型文件信息
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型目录不存在: {model_path}")
    
    # 检查常见的模型文件名
    possible_files = [
        "pytorch_model.bin",
        "model.safetensors",
        "pytorch_model.bin.index.json",  # 分片模型索引
    ]
    
    found_files = []
    for fname in possible_files:
        fpath = os.path.join(model_path, fname)
        if os.path.exists(fpath):
            found_files.append(fname)
    
    # 检查是否有分片文件
    shard_files = glob.glob(os.path.join(model_path, "pytorch_model-*.bin"))
    if shard_files:
        found_files.append(f"分片文件: {len(shard_files)} 个")
    
    return found_files

def _resolve_model_path(path_or_name: str):
    """
    解析模型路径，如果路径不存在则打印错误信息
    """
    if os.path.exists(path_or_name):
        # 检查模型文件
        found = _check_model_files(path_or_name)
        if not found:
            print(f"[WARN] 在 {path_or_name} 中未找到标准模型文件")
            print(f"      请确认模型文件已正确下载")
            print(f"      需要的文件: pytorch_model.bin 或 model.safetensors 或分片文件")
        return path_or_name
    else:
        print(f"[ERROR] 模型路径不存在: {path_or_name}")
        raise FileNotFoundError(f"模型路径不存在: {path_or_name}")
    return path_or_name

def pretrain_bert_wwm_token():
    try:
        tokenizer = BertTokenizer.from_pretrained(_resolve_model_path(ROBERTA_WWM_PATH), local_files_only=HUGGINGFACE_OFFLINE)
        return tokenizer
    except Exception as e:
        print(f"[ERROR] 加载 BERT-wwm tokenizer 失败")
        print(f"       路径: {ROBERTA_WWM_PATH}")
        print(f"       错误: {e}")
        print(f"\n请检查:")
        print(f"1. 模型目录是否存在: {ROBERTA_WWM_PATH}")
        print(f"2. 目录中是否包含以下文件之一:")
        print(f"   - pytorch_model.bin")
        print(f"   - model.safetensors")
        print(f"   - pytorch_model.bin.index.json (分片模型)")
        print(f"   - config.json")
        print(f"   - vocab.txt")
        raise

def pretrain_bert_wwm_model():
    try:
        model = BertModel.from_pretrained(_resolve_model_path(ROBERTA_WWM_PATH), local_files_only=HUGGINGFACE_OFFLINE).cuda()
        for param in model.parameters():
            param.requires_grad = False
        return model
    except Exception as e:
        print(f"[ERROR] 加载 BERT-wwm 模型失败")
        print(f"       路径: {ROBERTA_WWM_PATH}")
        print(f"       错误: {e}")
        print(f"\n请检查:")
        print(f"1. 模型目录是否存在: {ROBERTA_WWM_PATH}")
        print(f"2. 目录中是否包含以下文件之一:")
        print(f"   - pytorch_model.bin")
        print(f"   - model.safetensors")
        print(f"   - pytorch_model.bin.index.json (分片模型)")
        print(f"   - config.json")
        raise

def pretrain_bert_uncased_token():
    try:
        tokenizer = BertTokenizer.from_pretrained(_resolve_model_path(BERT_UNCASED_PATH), local_files_only=HUGGINGFACE_OFFLINE)
        return tokenizer
    except Exception as e:
        print(f"[ERROR] 加载 BERT-uncased tokenizer 失败")
        print(f"       路径: {BERT_UNCASED_PATH}")
        print(f"       错误: {e}")
        raise

def pretrain_bert_uncased_model():
    try:
        model = BertModel.from_pretrained(_resolve_model_path(BERT_UNCASED_PATH), local_files_only=HUGGINGFACE_OFFLINE).cuda()
        for param in model.parameters():
            param.requires_grad = False
        return model
    except Exception as e:
        print(f"[ERROR] 加载 BERT-uncased 模型失败")
        print(f"       路径: {BERT_UNCASED_PATH}")
        print(f"       错误: {e}")
        raise

