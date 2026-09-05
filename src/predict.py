import torch
import numpy as np
import os
import json

from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download

from .dataset.dataset import CaptchaDataset
from .dataset.save import resize_keep_ratio
from .dataset.const import num_classes, idx2char
from .model import CaptchaCNNTransformer
from .train import decode_prediction
from .dataset.download_utils import download_file, download_zip

def predict(filepaths, model_path):

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("Using device:", device)

    transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
    ])


    model = CaptchaCNNTransformer(
        img_h=32,
        img_w=128,
        dim=256,
        depth=6,
        heads=4,
        num_classes=num_classes,
        channels=1,
        dropout=0.2,
    ).to(device)

    model.load_state_dict(
        torch.load(
            model_path,
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

def predict_with_model(model_path, output_path):

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        with open(output_path, 'r', encoding='utf-8') as f:
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
    
        results = predict(filepaths, model_path)

        for filepath, result in results.items():
            data[filepath] = result
        
        with open(output_path, 'w+', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"{i}-{i + step} done")
        i += step


if __name__ == '__main__':

    model_filename = 'best.pth'
    # download_zip()
    # download_file(model_filename)

    # model_path = f'./{model_filename}'
    # output_path = './model_predict_cnn_data.json'

    # predict_with_model(model_path, output_path)
    res = predict(['./captcha.jpg'], model_path=model_filename)
    print(res)