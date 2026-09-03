import torch
import numpy as np

from PIL import Image
from torchvision import transforms

from .dataset.dataset import CaptchaDataset
from .dataset.save import resize_keep_ratio
from .model import CaptchaViT
from .train import decode_prediction

def predict(filepath):
    transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
    ])
    
    full_dataset = CaptchaDataset('./captcha_dataset.pt', transform=transform)
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
    )
    
    model.load_state_dict(torch.load('./best.pth', map_location='cpu'))
    model.eval()
    
    img = Image.open(filepath).convert('L')
    img = resize_keep_ratio(img)

    img_array = np.array(img)
    img_tensor = torch.from_numpy(img_array)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    img_tensor = transform(img_tensor)
    
    with torch.no_grad():
        logits = model(img_tensor)
        pred_str = decode_prediction(logits[0], idx2char)
        print(pred_str)

if __name__ == '__main__':

    for i in range(20):
        predict(f'./data/captcha_0000{i}.jpg')