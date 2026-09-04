import os
import zipfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from .model import CaptchaViT, CaptchaCNNTransformer
from .dataset.dataset import CaptchaDataset, ctc_collate_fn

def download_pt():

    if not os.path.exists('./captcha_dataset.pt'):
        path = hf_hub_download(
            repo_id="freexiaochuan/captcha",
            filename="captcha_dataset.pt",
            repo_type="dataset",
            local_dir="./"
        )

        if os.path.exists('./captcha_dataset.pt'):
            print("文件下载成功")
        else:
            print("文件下载失败")


def decode_prediction(logits, idx2char, blank_idx=0):
    pred = logits.argmax(dim=-1)
    prev = blank_idx
    chars = []
    for p in pred:
        p = p.item()
        if p != prev and p != blank_idx:
            chars.append(idx2char[p])
        prev = p
    return ''.join(chars)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    train_transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
    ])

    val_transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
    ])

    full_dataset = CaptchaDataset(
        './captcha_dataset.pt',
        transform=None
    )

    num_classes = full_dataset.num_classes
    idx2char = full_dataset.idx2char

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    indices = torch.randperm(len(full_dataset)).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = CaptchaDataset(
        './captcha_dataset.pt',
        transform=train_transform
    )

    val_dataset = CaptchaDataset(
        './captcha_dataset.pt',
        transform=val_transform
    )

    train_set = Subset(train_dataset, train_indices)
    val_set = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=ctc_collate_fn
    )

    val_loader = DataLoader(
        val_set,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=ctc_collate_fn
    )

    model = CaptchaCNNTransformer(
        img_h=32,
        img_w=128,
        dim=256,
        depth=6,
        heads=8,
        num_classes=num_classes,
        channels=1,
        dropout=0.1,
    ).to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_acc = 0.0
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(1, 51):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for imgs, labels, label_lengths in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)

            logits = model(imgs)                          # (B, T, C)
            log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)  # (T, B, C)

            input_lengths = torch.full(
                size=(imgs.size(0),),
                fill_value=logits.size(1),
                dtype=torch.long,
                device=device
            )

            loss = criterion(log_probs, labels, input_lengths, label_lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels, label_lengths in val_loader:
                imgs = imgs.to(device)
                logits = model(imgs)

                for i in range(imgs.size(0)):
                    pred_str = decode_prediction(logits[i], idx2char)
                    start = sum(label_lengths[:i].tolist()) if i > 0 else 0
                    end = start + label_lengths[i].item()
                    true_indices = labels[start:end].tolist()
                    true_str = ''.join([idx2char[idx] for idx in true_indices])

                    if pred_str == true_str:
                        correct += 1
                    total += 1

        acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'checkpoints/best.pth')
            print(f"  → 保存最佳模型 (acc={acc:.4f})")

    print("训练完成！最佳准确率:", best_acc)


if __name__ == '__main__':
    download_pt()
    train()