import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CaptchaDataset(Dataset):
    def __init__(self, pt_path, transform=None):
        data = torch.load(pt_path, weights_only=False)
        self.images = data['images']
        self.labels = data['labels']
        self.transform = transform

        self.charset = '2345678abcdefgmnpwxy'
        self.num_classes = len(self.charset) + 1
        self.char2idx = {c: i + 1 for i, c in enumerate(self.charset)}
        self.idx2char = {i + 1: c for i, c in enumerate(self.charset)}
        self.blank = 0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = self.images[idx]
        img = torch.from_numpy(img).unsqueeze(0)

        if self.transform:
            img = self.transform(img)


        label_str = self.labels[idx]
        label = torch.tensor([self.char2idx[c] for c in label_str], dtype=torch.long)

        return img, label, len(label)

def ctc_collate_fn(batch):
    imgs, labels, label_lengths = zip(*batch)
    imgs = torch.stack(imgs, 0)

    labels = torch.cat(labels, dim=0)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)

    return imgs, labels, label_lengths

if __name__ == '__main__':
    dataset = CaptchaDataset('./captcha_dataset.pt')

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=None
    )

    for imgs, labels, label_strs in dataloader:
        print("imgs shape:", imgs.shape)       # (64, 1, 32, 128)
        print("labels shape:", labels.shape)   # (64, 4)
        print("label example:", label_strs[0], "→", labels[0])
        break