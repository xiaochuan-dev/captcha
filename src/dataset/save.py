import torch
import numpy as np
import json
from PIL import Image, ImageOps

def resize_keep_ratio(img, target_w=128, target_h=32):
    img.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
    delta_w = target_w - img.width
    delta_h = target_h - img.height
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    return ImageOps.expand(img, padding, fill=255)

def save_predict():
    with open('./model_predict_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open('./a.txt', 'r', encoding='utf-8') as ff:
        num = int(ff.read())

    img_list = []
    label_list = []

    count = 0
    for key, value in data.items():
        if count > num:
            break
        img = Image.open(key).convert('L')
        img = resize_keep_ratio(img)
        img_array = np.array(img)
        img_list.append(img_array)
        label_list.append(value)
        count += 1
    all_images = np.stack(img_list, axis=0)
    torch.save({'images': all_images, 'labels': label_list}, './new_dataset.pt')
    print(f"保存完成！总共 {len(img_list)} 张，内存占用 {all_images.nbytes / 1024 / 1024:.2f} MB")

if __name__ == '__main__':
    save_predict()