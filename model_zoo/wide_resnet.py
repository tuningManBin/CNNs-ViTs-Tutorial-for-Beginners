"""
Wide ResNet

Reference:
    [1] Zagoruyko S, Komodakis N. Wide residual networks[J].
        arXiv preprint arXiv:1605.07146, 2016.

    [2] https://github.com/FreeformRobotics/Divide-and-Co-training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# When importing * from wide_resnet, only the module name specified by __ all__ will be imported
__all__ = ["WideResNet", "wideresnet_28_10_cifar10", "wideresnet_28_10_mnist"]


class BasicBlock(nn.Module):
    """
    This module is used to define the basic block of ResNet.
    """
    def __init__(self, in_channels, out_channels, stride):
        """
        Arguments:
            in_channels (int): the number of input tensor channels
            out_channels (int): the number of output tensor channels
            stride (int): the step size of convolution (only in self.conv0)
        """
        super(BasicBlock, self).__init__()
        # conv + bn --> bias = False
        self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=(3, 3), stride=(stride, stride), padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=(1, 1), stride=(stride, stride)),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv0(x)
        out = self.bn0(out)
        out = self.activation(out)

        out = self.conv1(out)
        out = self.bn1(out)

        identity = self.shortcut(identity)
        out = out + identity
        out = self.activation(out)
        return out


class WideResNet(nn.Module):
    """
    This module is the main body of Wide ResNet.
    """
    def __init__(self, num_blocks, in_channels, widen_factor, num_features, pool_stride,
                 logits_dim, dataset, normalize, feature_size=1, basic_block=BasicBlock, return_feature_map=False):
        """
        Arguments:
            num_blocks (list): the number of basic block in different stages
            in_channels (int): the number of input tensor channels
            widen_factor (int): widening factor for the number of channels in the feature map
            num_features (int): the number of channels in the initial feature map
            pool_stride (int): the step size of pooling
            logits_dim (int): the dimensions of output tensor
            dataset (str): dataset name
            normalize (bool): control variable for normalizing
            feature_size (int): the size of feature map
            basic_block (nn.Module): a convolutional block with shortcut connection
            return_feature_map (bool): control variable for the form of return
        """
        super(WideResNet, self).__init__()
        self.return_feature_map = return_feature_map
        self.dataset = dataset
        self.normalize = normalize
        self.in_channels = num_features
        self.pool_stride = pool_stride
        if pool_stride != 0:
            self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(pool_stride, pool_stride), padding=1)
        self.init_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=self.in_channels,
                      kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True)
        )

        # stage0 layers = 2 + 1 + 2 * (num_blocks[0]-1), output shape = [batch_size, c0, h0, w0]
        self.stage0 = self._make_layer(basic_block=basic_block, num_blocks=num_blocks[0],
                                       out_channels=num_features * widen_factor, stride=1)
        # stage1 layers = 2 + 1 + 2 * (num_blocks[1]-1), output shape = [batch_size, c1, h0 // 2, w0 // 2]
        self.stage1 = self._make_layer(basic_block=basic_block, num_blocks=num_blocks[1],
                                       out_channels=2 * num_features * widen_factor, stride=2)
        # stage2 layers = 2 + 1 + 2 * (num_blocks[2]-1), output shape = [batch_size, c2, h0 // 4, w0 // 4]
        self.stage2 = self._make_layer(basic_block=basic_block, num_blocks=num_blocks[2],
                                       out_channels=4 * num_features * widen_factor, stride=2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((feature_size, feature_size))
        self.project = nn.Linear(feature_size ** 2 * 4 * num_features * widen_factor, logits_dim)

    def _make_layer(self, basic_block, num_blocks, out_channels, stride):
        """
        This function is used to build different stages in ResNet.

        Arguments:
            basic_block (nn.Module): a convolutional block with shortcut connection
            num_blocks (int): the number of basic blocks
            out_channels (int): the number of output tensor channels
            stride (int): step size of convolution (only in the first layer)
        """
        layers = [basic_block(in_channels=self.in_channels, out_channels=out_channels, stride=stride)]
        self.in_channels = out_channels  # Update self.in_channels

        for _ in range(num_blocks - 1):
            layers.append(basic_block(in_channels=out_channels, out_channels=out_channels, stride=1))
        return nn.Sequential(*layers)
        # nn.Sequential(*layers) represents the extraction of elements from layers to form sequential

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
            assert self.pool_stride != 0, f"pool_stride is 0"

        out = self.init_block(images)

        out = self.stage0(out)
        out = self.stage1(out)
        out = self.stage2(out)

        feature_map = self.adaptive_pool(out)
        if self.return_feature_map:
            return feature_map
        else:
            logits = self.project(feature_map.flatten(1))
            return logits


def wideresnet_28_10_cifar10(in_channels=3, widen_factor=10, num_features=16, logits_dim=10, pool_stride=0,
                             dataset="cifar-10", normalize=True, return_feature_map=False):
    """
    A version deploying wide-resnet28-10 to CIFAR-10.
    """
    return WideResNet(num_blocks=[4, 4, 4], in_channels=in_channels, widen_factor=widen_factor, pool_stride=pool_stride,
                      num_features=num_features, logits_dim=logits_dim, dataset=dataset,
                      normalize=normalize, return_feature_map=return_feature_map)


def wideresnet_28_10_mnist(in_channels=1, widen_factor=10, num_features=16, logits_dim=10, pool_stride=0,
                           dataset="mnist", normalize=True, return_feature_map=False):
    """
    A version deploying wide-resnet28-10 to MNIST.
    """
    return WideResNet(num_blocks=[4, 4, 4], in_channels=in_channels, widen_factor=widen_factor, pool_stride=pool_stride,
                      num_features=num_features, logits_dim=logits_dim, dataset=dataset,
                      normalize=normalize, return_feature_map=return_feature_map)


if __name__ == "__main__":
    net0 = wideresnet_28_10_cifar10()
    test_sample0 = torch.rand([2, 3, 32, 32])
    pred0 = net0(test_sample0)
    print(pred0.shape)

    net1 = wideresnet_28_10_mnist()
    test_sample1 = torch.rand([2, 1, 28, 28])
    pred1 = net1(test_sample1)
    print(pred1.shape)
