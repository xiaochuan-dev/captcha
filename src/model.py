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
        emb_dropout=0.1,
        dropout=0.1,
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
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.encoder = Encoder(
            dim=dim,
            depth=depth,
            heads=heads,
            ff_mult=4,
            attn_dropout=0.15,
            ff_dropout=0.15,
        )

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
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
        x = self.emb_dropout(x)

        x = self.encoder(x)
        x = self.norm(x)
        x = self.dropout(x)

        logits = self.to_logits(x)
        return logits


class CaptchaCNNTransformer(nn.Module):
    def __init__(
        self,
        img_h=32,
        img_w=128,
        dim=256,
        depth=6,
        heads=4,
        num_classes=37,
        channels=1,
        dropout=0.1,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        # 原来：8 -> 1
        # 现在：8 -> 2
        self.height_pool = nn.Sequential(
            nn.Conv2d(
                256,
                256,
                kernel_size=(4, 1),
                stride=(4, 1)
            ),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        # 原来输入是 256
        # 现在高度保留 2，因此输入变成 256 * 2 = 512
        self.feature_proj = nn.Sequential(
            nn.Linear(512, dim),
            nn.LayerNorm(dim),
        )

        # 宽度仍然是 32
        self.seq_len = img_w // 4

        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.seq_len, dim) * 0.02
        )

        self.dropout = nn.Dropout(dropout)

        self.encoder = Encoder(
            dim=dim,
            depth=depth,
            heads=heads,
            ff_mult=4,
            attn_dropout=0.1,
            ff_dropout=0.1,
        )

        self.norm = nn.LayerNorm(dim)

        self.to_logits = nn.Linear(
            dim,
            num_classes
        )

    def forward(self, x):
        # 输入:
        # B, 1, 32, 128
        x = self.cnn(x)

        # B, 256, 8, 32
        x = self.height_pool(x)

        # B, 256, 2, 32

        # 把高度维度和 channel 合并
        # B, 256, 2, 32
        # ->
        # B, 32, 512
        x = rearrange(
            x,
            'b c h w -> b w (c h)'
        )

        # B, 32, 512
        # ->
        # B, 32, 256
        x = self.feature_proj(x)

        x = x + self.pos_embedding[:, :x.size(1)]

        x = self.dropout(x)

        x = self.encoder(x)

        x = self.norm(x)

        logits = self.to_logits(x)

        # B, 32, 37
        return logits
# model = CaptchaCNNTransformer(
#     img_h=32,
#     img_w=128,
#     dim=256,
#     depth=6,
#     heads=4,
#     num_classes=num_classes,
#     channels=1,
#     dropout=0.1,
# ).to(device)