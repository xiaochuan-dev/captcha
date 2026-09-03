import torch
import numpy as np
import os
import json
import zipfile

from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download

from .dataset.dataset import CaptchaDataset
from .dataset.save import resize_keep_ratio
from .model import CaptchaViT
from .train import decode_prediction

def download_model():
    if os.path.exists('./best.pth'):
        hf_hub_download(
            repo_id="freexiaochuan/captcha",
            filename="best.pth",
            repo_type="dataset",
            local_dir="./"
        )
        hf_hub_download(
            repo_id="freexiaochuan/captcha",
            filename="captcha_dataset.pt",
            repo_type="dataset",
            local_dir="./"
        )
        if os.path.exists('./captcha_dataset.pt'):
            print("文件下载成功")
        else:
            print("文件下载失败")
    zip_path = hf_hub_download(
        repo_id="freexiaochuan/captcha",
        filename="data.zip",
        repo_type="dataset"
    )

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")
        
    print("解压完成！图片在 . 目录下") 

def predict(filepaths):

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("Using device:", device)

    transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
    ])

    full_dataset = CaptchaDataset(
        './captcha_dataset.pt',
        transform=transform
    )

    num_classes = full_dataset.num_classes
    idx2char = full_dataset.idx2char

    model = CaptchaViT(
        img_h=32,
        img_w=128,
        patch_h=8,
        patch_w=8,
        dim=192,
        depth=6,
        heads=3,
        num_classes=num_classes,
        channels=1,
    ).to(device)

    model.load_state_dict(
        torch.load(
            './best.pth',
            map_location=device
        )
    )

    model.eval()

    images = []
    valid_filepaths = []

    for filepath in filepaths:

        if not os.path.exists(filepath):
            print(f"文件不存在: {filepath}")
            continue

        img = Image.open(filepath).convert('L')
        img = resize_keep_ratio(img)

        img_array = np.array(img)

        img_tensor = torch.from_numpy(img_array)
        img_tensor = img_tensor.unsqueeze(0)

        images.append(img_tensor)
        valid_filepaths.append(filepath)

    if not images:
        return {}

    imgs = torch.stack(images, dim=0)
    imgs = transform(imgs)

    imgs = imgs.to(device)


    with torch.no_grad():
        logits = model(imgs)

    results = {}

    for filepath, logit in zip(valid_filepaths, logits):

        pred_str = decode_prediction(
            logit,
            idx2char
        )

        results[filepath] = pred_str

    return results

if __name__ == '__main__':

    download_model()

    data_file = './model_predict_data.json'
    if os.path.exists(data_file) and os.path.getsize(data_file) > 0:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = {}

    total = 80000

    step = 1000
    i = 0

    while i < total:
        filepaths = [
            f'./data/captcha_{i:05d}.jpg'
            for i in range(i, i + step)
            if os.path.exists(f"./data/captcha_{i:05d}.jpg")
        ]
    
        results = predict(filepaths)

        for filepath, result in results.items():
            data[filepath] = result
        
        with open('./data.json', 'w+', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"{i}-{i + step} done")
        i += step

