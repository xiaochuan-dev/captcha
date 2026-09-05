import torch
from torch.utils.data import Dataset, DataLoader
from .const import charset, num_classes, char2idx, idx2char, blank

class CaptchaDataset(Dataset):
    def __init__(self, pt_path, transform=None):
        data = torch.load(pt_path, weights_only=False)
        self.images = data['images']
        self.labels = data['labels']
        self.transform = transform

        self.charset = charset
        self.num_classes = num_classes
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.blank = blank

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