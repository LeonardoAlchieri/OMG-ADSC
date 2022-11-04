import math

import torch
from einops import rearrange, repeat
from torch import einsum, nn


class ASP(nn.Module):
    def __init__(self, n_features):
        super(ASP, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(n_features, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, n_features, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        w = self.attention(x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-5))
        x = torch.cat((mu, sg), 1)
        return x


class SAP(nn.Module):
    def __init__(self, n_features):
        super(SAP, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(n_features, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, n_features, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        w = self.attention(x)
        x = torch.sum(x * w, dim=2)
        return x


class Identity(nn.Module):
    def forward(self, x):
        return x


class TAP(nn.Module):
    def forward(self, x):
        return x.mean(1)


class GELU(nn.Module):
    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))
            )
        )


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)
        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            PreNorm(
                                dim,
                                Attention(
                                    dim, heads=heads, dim_head=dim_head, dropout=dropout
                                ),
                            )
                        ),
                        Residual(
                            PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                        ),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


class TFormer(nn.Module):
    def __init__(
        self,
        num_patches=16,
        dim=512,
        depth=3,
        heads=8,
        mlp_dim=1024,
        dim_head=64,
        dropout=0.0,
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.spatial_transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout
        )

    def forward(self, x):
        x = x.contiguous().view(-1, 16, 512)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, : (n + 1)]
        x = self.spatial_transformer(x)
        x = x[:, 1:]
        return x


def temporal_transformer():
    return TFormer()


if __name__ == "__main__":
    img = torch.randn((1, 16, 3, 112, 112))
    model = temporal_transformer()
    model(img)
