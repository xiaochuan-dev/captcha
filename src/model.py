import torch
import torch.nn as nn
from einops import rearrange
from x_transformers import Encoder

class CaptchaViT(nn.Module):
    def __init__(
        self,
        img_h=32,
        img_w=128,
        patch_h=8,
        patch_w=8,
        dim=192,
        depth=6,
        heads=3,
        num_classes=37,
        channels=1,
    ):
        super().__init__()
        assert img_h % patch_h == 0 and img_w % patch_w == 0

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.num_patches_h = img_h // patch_h
        self.num_patches_w = img_w // patch_w
        num_patches = self.num_patches_h * self.num_patches_w

        patch_dim = channels * patch_h * patch_w

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        self.encoder = Encoder(
            dim=dim,
            depth=depth,
            heads=heads,
            ff_mult=4,
            attn_dropout=0.1,
            ff_dropout=0.1,
        )

        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_classes)

    def forward(self, x):
        """
        x: (B, C, H, W)   例如 (B, 1, 32, 128)
        return: (B, T, num_classes)   T = 高方向patch数 * 宽方向patch数
        """
        b, c, h, w = x.shape

        x = rearrange(
            x,
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
            p1=self.patch_h,
            p2=self.patch_w
        )

        x = self.to_patch_embedding(x)
        x = x + self.pos_embedding

        x = self.encoder(x)
        x = self.norm(x)

        logits = self.to_logits(x)
        return logits