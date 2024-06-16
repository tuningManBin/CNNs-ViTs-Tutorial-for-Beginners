"""
Vision Transformer

Reference:
    [1] Dosovitskiy A, Beyer L, Kolesnikov A, et al.
        An image is worth 16x16 words: Transformers for image recognition at scale[J].
        arXiv preprint arXiv:2010.11929, 2020.

    [2] https://github.com/asyml/vision-transformer-pytorch
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class ParchEmbedding(nn.Module):
    """
    This module is used to convert image pixel patches to embeddings according to patch size.
    """
    def __init__(self, in_channels, image_size, patch_size, embedding_dim, drop_rate=0.1):
        """
        Arguments:
            in_channels (int): the number of input tensor channels
            image_size (int): the size of image (h = w)
            patch_size (int): the size of pixel patch (h = w)
            embedding_dim (int): the dimensions of embedding
            drop_rate (float): probability of an element to be zeroed
        """
        super(ParchEmbedding, self).__init__()
        self.project = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim,
                                 kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
        self.num_patches = (image_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embedding_dim))

        self.drop = nn.Dropout(p=drop_rate)

    def forward(self, images):
        # --- Patch Embedding
        tokens = self.project(images)
        # output shape = [batch_size, embedding_dim, image_size // patch_size, image_size // patch_size]
        tokens = tokens.flatten(2).transpose(1, 2)
        # output shape = [batch_size, num_patches, embedding_dim]

        # --- Position Embedding
        tokens = tokens + self.position_embedding[:, 1:, :]

        # --- Add class token
        cls_token = self.cls_token + self.position_embedding[:, :1, :]
        cls_tokens = cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        # output shape = [batch_size, num_patches + 1, embedding_dim]
        tokens = self.drop(tokens)
        return tokens


class SelfAttention(nn.Module):
    """
    This module is used to do Multi-Head Self-Attention.
    """
    def __init__(self, embedding_dim, num_heads, attn_drop_rate=0.1, proj_drop_rate=0.1):
        """
        Arguments:
            embedding_dim (int): the dimensions of embedding
            num_heads (int): the number of self-attention heads
            attn_drop_rate (float): probability of an element to be zeroed in attention map
            proj_drop_rate (float): probability of an element to be zeroed in projection
        """
        super(SelfAttention, self).__init__()
        assert embedding_dim % num_heads == 0, \
            f"embedding_dim {embedding_dim} should be divided by num_heads {num_heads}."
        self.norm = nn.LayerNorm(embedding_dim)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = self.embedding_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv_matrices = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)

        self.project = nn.Linear(embedding_dim, embedding_dim)
        self.proj_drop = nn.Dropout(p=attn_drop_rate)
        self.attn_drop = nn.Dropout(p=proj_drop_rate)

    def forward(self, tokens):
        res = tokens

        batch_size, num_patches = tokens.shape[0], tokens.shape[1]

        tokens = self.norm(tokens)

        qkv = self.qkv_matrices(tokens).reshape(batch_size, num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # output shape = [batch_size, num_heads, num_patches, head_dim]

        scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_map = scores.softmax(dim=-1)
        attn_map = self.attn_drop(attn_map)
        dots = attn_map @ v
        # output shape = [batch_size, num_heads, num_patches, head_dim]
        dots = dots.transpose(1, 2)
        # output shape = [batch_size, num_patches, num_heads, head_dim]
        dots = dots.reshape(batch_size, num_patches, self.embedding_dim)
        # output shape = [batch_size, num_patches, embedding_dim]
        dots = self.project(dots)
        dots = self.proj_drop(dots)

        dots = dots + res
        return dots


class MLP(nn.Module):
    """
    This module is used to define multi-layer perceptron.
    """
    def __init__(self, input_dim, hidden_dim=None, output_dim=None, activation=nn.GELU, drop_rate=0.1):
        """
        Arguments:
            input_dim (int): the dimensions of input tensor
            hidden_dim (int): the dimensions of output tensor in hidden layer
            output_dim (int): the dimensions of output tensor
            activation (nn.modules.activation): activation function
            drop_rate (float): probability of an element to be zeroed
        """
        super(MLP, self).__init__()
        output_dim = output_dim or input_dim
        hidden_dim = hidden_dim or input_dim

        self.fc0 = nn.Linear(input_dim, hidden_dim)
        self.activation = activation()
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(drop_rate)

        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        res = x

        x = self.norm(x)

        x = self.fc0(x)
        x = self.activation(x)
        x = self.drop(x)

        x = self.fc1(x)
        x = self.drop(x)

        x = x + res
        return x


class BasicBlock(nn.Module):
    """
    This module is used to define the Basic block of ViT.
    """
    def __init__(self, embedding_dim, num_heads, mlp_ratio):
        """
        Arguments:
            embedding_dim (int): the dimensions of embedding
            num_heads (int): the number of self-attention heads
            mlp_ratio (int): the ratio of hidden layer dimensions to input dimensions
        """
        super(BasicBlock, self).__init__()
        self.attention = SelfAttention(embedding_dim=embedding_dim, num_heads=num_heads)
        self.project = MLP(input_dim=embedding_dim, hidden_dim=embedding_dim * mlp_ratio)

    def forward(self, tokens):
        tokens = self.attention(tokens)
        tokens = self.project(tokens)
        # output shape = [batch_size, num_patches, embedding_dim]
        return tokens


class VisionTransformer(nn.Module):
    """
    This module is the main body of vision transformer.
    """
    def __init__(self, image_channels, image_size, pool_stride,
                 patch_size, embedding_dim, num_heads, depth, mlp_ratio,
                 dataset, normalize, logits_dim):
        """
        Arguments:
            image_channels (int): the number of image channels
            image_size (int): the size of image (h = w)
            pool_stride (int): the step size of pooling
            patch_size (int): the size of pixel patch (h = w)
            embedding_dim (int): the dimensions of embedding
            num_heads (int): the number of self-attention heads
            depth (int): the number of basic blocks (attention + mlp)
            mlp_ratio (int): the ratio of hidden layer dimensions to input dimensions
            dataset (str): dataset name
            normalize (bool): control variable for normalizing
            logits_dim (int): the dimensions of output tensor
        """
        super(VisionTransformer, self).__init__()
        self.dataset = dataset
        self.image_size_raw = image_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.normalize = normalize
        self.pool_stride = pool_stride
        if pool_stride != 0:
            self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(pool_stride, pool_stride), padding=1)
            self.image_size = image_size // pool_stride
        assert self.image_size % patch_size == 0, \
            f"image_size // pool_stride {self.image_size} should be divided by patch_size {patch_size}."

        self.embedding = ParchEmbedding(in_channels=image_channels, image_size=self.image_size,
                                        embedding_dim=embedding_dim, patch_size=patch_size)

        self.blocks = nn.Sequential(*[
            BasicBlock(embedding_dim=embedding_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        # nn.Sequential(*layers) represents the extraction of elements from layers to form of sequential

        self.cls_project = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, logits_dim)
        )

    def forward(self, images):
        if self.normalize:
            # --- Preprocessing
            if self.dataset == "cifar-10":
                images = F.interpolate(images, size=(32, 32), mode="bilinear", align_corners=True)
                assert images.shape[-1] == images.shape[-2] == 32, f"image_size in cifar-10 should be 32."

                mean = torch.tensor([0.4914, 0.4822, 0.4465], device=images.device)
                mean = mean.expand(images.shape[0], -1)
                mean = mean.unsqueeze(-1).unsqueeze(-1)

                std = torch.tensor([0.2471, 0.2435, 0.2616], device=images.device)
                std = std.expand(images.shape[0], -1)
                std = std.unsqueeze(-1).unsqueeze(-1)

                images = torch.clamp(images, min=0.0, max=1.0)
                images = (images - mean) / std

            if self.dataset == "mnist":
                images = F.interpolate(images, size=(28, 28), mode="bilinear", align_corners=True)
                assert images.shape[-1] == images.shape[-2] == 28, f"image_size in mnist should be 28."

                mean = torch.tensor([0.5], device=images.device)
                mean = mean.expand(images.shape[0], -1)
                mean = mean.unsqueeze(-1).unsqueeze(-1)

                std = torch.tensor([0.5], device=images.device)
                std = std.expand(images.shape[0], -1)
                std = std.unsqueeze(-1).unsqueeze(-1)

                images = torch.clamp(images, min=0.0, max=1.0)
                images = (images - mean) / std

        if self.pool_stride != 0:
            images = self.pool(images)

        tokens = self.embedding(images)
        tokens = self.blocks(tokens)
        # output shape = [batch_size, num_patches, embedding_dim]

        cls_token = tokens[:, 0]
        logits = self.cls_project(cls_token)
        return logits


def ViT_tiny_cifar10(image_channels=3, image_size=32, pool_stride=0,
                     patch_size=2, embedding_dim=192, num_heads=3, depth=12, mlp_ratio=4,
                     dataset="cifar-10", normalize=True, logits_dim=10):
    """
    A version deploying tiny ViT to CIFAR-10.
    """
    return VisionTransformer(image_channels=image_channels, image_size=image_size, pool_stride=pool_stride,
                             patch_size=patch_size, embedding_dim=embedding_dim,
                             num_heads=num_heads, depth=depth, mlp_ratio=mlp_ratio,
                             dataset=dataset, normalize=normalize, logits_dim=logits_dim)


def ViT_tiny_mnist(image_channels=1, image_size=28, pool_stride=0,
                   patch_size=2, embedding_dim=192, num_heads=3, depth=12, mlp_ratio=4,
                   dataset="mnist", normalize=True, logits_dim=10):
    """
    A version deploying tiny ViT to MNIST.
    """
    return VisionTransformer(image_channels=image_channels, image_size=image_size, pool_stride=pool_stride,
                             patch_size=patch_size, embedding_dim=embedding_dim,
                             num_heads=num_heads, depth=depth, mlp_ratio=mlp_ratio,
                             dataset=dataset, normalize=normalize, logits_dim=logits_dim)


if __name__ == "__main__":
    net0 = ViT_tiny_cifar10()
    test_sample0 = torch.rand([2, 3, 32, 32])
    pred0 = net0(test_sample0)
    print(pred0.shape)

    net1 = ViT_tiny_mnist()
    test_sample1 = torch.rand([2, 1, 28, 28])
    pred1 = net1(test_sample1)
    print(pred1.shape)
