import os
import re


def extract_checkpoint_name_val_loss(filename):
    # 使用正则表达式提取数字部分
    match = re.search(r'val_loss_(\d+\.\d+)', filename)
    if match:
        return float(match.group(1))
    else:
        return None


def get_best_trained_model_path(path):
    pth_files = [file for file in os.listdir(path) if file.endswith('.pth') or file.endswith('pt')]
    pth_files.sort(key=lambda x: extract_checkpoint_name_val_loss(x))
    best_model_pth = os.path.join(path, pth_files[0])
    return best_model_pth
