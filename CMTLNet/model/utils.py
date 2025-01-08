import torch
import torch.nn as nn
from torch.nn import functional as F

# LMCL [CosFace: Large Margin Cosine Loss for Deep Face Recognition]
class MarginLoss(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super(MarginLoss, self).__init__()
        self.embedding_size = in_features
        self.num_classes = out_features
        self.s = s
        self.m = m
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_normal_(self.weights)

    def forward(self, embedding, label):
        assert embedding.size(1) == self.embedding_size, 'embedding size wrong'
        logits = F.linear(F.normalize(embedding), F.normalize(self.weights))
        margin = torch.zeros_like(logits)
        margin.scatter_(1, label.view(-1, 1), self.m)
        m_logits = self.s * (logits - margin)
        return m_logits

    def predict(self, input):
        with torch.no_grad():
            return F.linear(F.normalize(input), F.normalize(self.weights))


class ChannelAttention_3D(nn.Module):
    """
    Implements channel-wise attention for 3D feature maps.
    """
    def __init__(self, channel, reduction=4, min_channel=32, unsqueeze=True):
        """
        Initializes the ChannelAttention_3D module.

        Args:
            channel (int): Number of input channels.
            reduction (int, optional): Reduction ratio for channel dimensions. Default is 4.
            min_channel (int, optional): Minimum number of channels after reduction. Default is 32.
            unsqueeze (bool, optional): Whether to expand the output to match input dimensions. Default is True.
        """
        super().__init__()
        # Calculate the number of channels after reduction, ensuring it doesn't go below min_channel
        attention_channel = max(int(channel // reduction), min_channel)

        self.unsqueeze = unsqueeze
        # Adaptive pooling to generate a single value per channel (global pooling)
        self.maxpool = nn.AdaptiveMaxPool3d(1)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        # Sequential layers for channel attention mechanism
        self.se = nn.Sequential(
            nn.Conv3d(channel, attention_channel, kernel_size=1, bias=False),
            nn.BatchNorm3d(attention_channel),
            nn.ReLU(),
            nn.Conv3d(attention_channel, channel, kernel_size=1, bias=False)
        )
        # Sigmoid activation to obtain attention weights between 0 and 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass for channel attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channel, D, H, W).

        Returns:
            torch.Tensor: Attention weights of shape (batch, channel, D, H, W) if unsqueeze is True,
                          else (batch, channel, 1, 1, 1).
        """
        # Apply max and average pooling
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        # Pass pooled results through the attention network
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        # Combine the outputs and apply sigmoid activation
        output = self.sigmoid(max_out + avg_out)
        # Optionally expand the attention weights to match the input dimensions
        if self.unsqueeze:
            output = output.expand_as(x)
        # Ensure contiguous memory layout
        return output.contiguous()


class SpatialAttention_3D(nn.Module):
    """
    Implements spatial attention for 3D feature maps.
    """
    def __init__(self, kernel_size=7, unsqueeze=True):
        """
        Initializes the SpatialAttention_3D module.

        Args:
            kernel_size (int, optional): Kernel size for the convolutional layer. Default is 7.
            unsqueeze (bool, optional): Whether to expand the output to match input dimensions. Default is True.
        """
        super().__init__()
        self.unsqueeze = unsqueeze
        # Convolution to compute spatial attention; input has 2 channels (max and avg), output has 1 channel
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        # Sigmoid activation to obtain attention weights between 0 and 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass for spatial attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, D, H, W).

        Returns:
            torch.Tensor: Attention weights of shape (batch, channels, D, H, W) if unsqueeze is True,
                          else (batch, 1, D, H, W).
        """
        # Compute the maximum value across the channel dimension
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        # Compute the average value across the channel dimension
        avg_result = torch.mean(x, dim=1, keepdim=True)
        # Concatenate along the channel dimension to form a 2-channel tensor
        result = torch.cat([max_result, avg_result], dim=1)
        # Apply convolution and sigmoid activation to obtain spatial attention
        output = self.conv(result)
        output = self.sigmoid(output)
        # Optionally expand the attention weights to match the input dimensions
        if self.unsqueeze:
            output = output.expand_as(x)
        # Ensure contiguous memory layout
        return output.contiguous()

class UCFCFusion(nn.Module):
    """
    Implements a fusion module that integrates multiple feature maps using channel and spatial attention.
    """
    def __init__(self, channel, ochannel, fusion_num, img_size=[4, 4, 4], L=32, reduction_ca=4, reduction_sa=64):
        """
        Initializes the UCFCFusion module.

        Args:
            channel (int): Number of input channels per feature map.
            ochannel (int): Number of output channels after fusion.
            fusion_num (int): Number of feature maps to fuse.
            img_size (list, optional): Spatial dimensions [D, H, W] of the input feature maps. Default is [4, 4, 4].
            L (int, optional): Base dimension for linear layers. Default is 32.
            reduction_ca (int, optional): Reduction ratio for channel attention. Default is 4.
            reduction_sa (int, optional): Reduction ratio for spatial attention. Default is 64.
        """
        super(UCFCFusion, self).__init__()

        self.img_size = img_size

        in_planes = channel * fusion_num  # Total input channels after concatenation
        out_planes = channel  # Desired output channels

        # Convolution layers for projecting input and fused features
        self.proj = nn.Conv3d(in_planes, ochannel, kernel_size=3, stride=1, padding=1, bias=False)
        self.projf = nn.Conv3d(out_planes, ochannel, kernel_size=3, stride=1, padding=1, bias=False)

        # Channel and Spatial attention mechanisms applied to the summed feature maps
        self.channel_F_W = ChannelAttention_3D(out_planes, unsqueeze=False)
        self.spatial_F_W = SpatialAttention_3D(unsqueeze=False)

        self.fusion_num = fusion_num
        # Dimension for channel attention processing
        self.d = max(L, in_planes // reduction_ca)
        # Dimension for spatial attention processing
        self.dsa = max(L * 2,
                       (self.img_size[0] * self.img_size[1] * self.img_size[2]) // reduction_sa)

        # Fully connected layers for processing attention outputs
        self.fc = nn.Linear(out_planes, self.d)
        self.fc2 = nn.Linear(self.img_size[0] * self.img_size[1] * self.img_size[2], self.dsa)

        # Lists of linear layers for each fusion path
        self.fcas = nn.ModuleList([])
        self.fsas = nn.ModuleList([])
        for _ in range(fusion_num):
            self.fcas.append(nn.Linear(self.d, out_planes))
            self.fsas.append(nn.Linear(self.dsa, self.img_size[0] * self.img_size[1] * self.img_size[2]))

        # Softmax layer to normalize fusion weights across different fusion paths
        self.softmax = nn.Softmax(dim=0)
        # Initialize weights of convolutional and batch normalization layers
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initializes the weights of convolutional and batch normalization layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # Initialize Conv3d weights with Kaiming Normal and biases to zero if present
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                # Initialize BatchNorm3d weights to 1 and biases to zero
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_fusion_w(self, x_F, ca_out, sa_out):
        """
        Computes fusion weights based on channel and spatial attention outputs.

        Args:
            x_F (torch.Tensor): Summed feature maps of shape (batch, channels, D, H, W).
            ca_out (torch.Tensor): Channel attention output of shape (batch, channels, 1, 1, 1).
            sa_out (torch.Tensor): Spatial attention output of shape (batch, 1, D, H, W).

        Returns:
            tuple:
                - weights_ca (torch.Tensor): Channel-wise weights for each fusion path.
                - weights_sa (torch.Tensor): Spatial-wise weights for each fusion path.
        """
        b, c, w, h, l = x_F.size()
        weights_ca = []
        # Process channel attention output through a fully connected layer
        ca_out_p = self.fc(ca_out.view(b, c))  # Shape: (batch, d)
        for fca in self.fcas:
            # Compute channel weights for each fusion path
            weight = fca(ca_out_p)  # Shape: (batch, out_planes)
            weights_ca.append(weight.view(b, c, 1, 1, 1))  # Reshape for broadcasting

        # Stack channel weights across fusion paths
        weights_ca = torch.stack(weights_ca, dim=0)  # Shape: (fusion_num, batch, channels, 1, 1, 1)

        # Process spatial attention output through a fully connected layer
        sa_out_fc = self.fc2(sa_out.view(b, w * h * l))  # Shape: (batch, dsa)
        weights_sa = []
        for fsa in self.fsas:
            # Compute spatial weights for each fusion path
            weight = fsa(sa_out_fc)  # Shape: (batch, D*H*W)
            weights_sa.append(weight.view(b, 1, w, h, l))  # Reshape for broadcasting

        # Stack spatial weights across fusion paths
        weights_sa = torch.stack(weights_sa, dim=0)  # Shape: (fusion_num, batch, 1, D, H, W)
        return weights_ca, weights_sa

    def forward(self, I_IN):
        """
        Forward pass for the UCFCFusion module.

        Args:
            I_IN (torch.Tensor): Input tensor of shape (batch, channel * fusion_num, D, H, W).

        Returns:
            torch.Tensor: Fused output tensor of shape (batch, ochannel, D, H, W).
        """
        # Split the input tensor into 'fusion_num' parts along the channel dimension
        split_I_IN_F = torch.chunk(I_IN, self.fusion_num, dim=1)  # Tuple of tensors
        # Stack the split tensors to form a new dimension for fusion paths
        split_I_IN_F = torch.stack(split_I_IN_F, dim=0)  # Shape: (fusion_num, batch, channel, D, H, W)

        # Sum the feature maps across all fusion paths
        Sum_I_IN_F = split_I_IN_F.sum(dim=0)  # Shape: (batch, channel, D, H, W)

        # Apply channel and spatial attention to the summed feature maps
        channel_F_W = self.channel_F_W(Sum_I_IN_F)  # Shape: (batch, channel, 1, 1, 1)
        spatial_F_W = self.spatial_F_W(Sum_I_IN_F)  # Shape: (batch, 1, D, H, W)

        # Compute fusion weights based on attention outputs
        weights_ca, weights_sa = self.get_fusion_w(Sum_I_IN_F, channel_F_W, spatial_F_W)

        # Combine channel and spatial weights multiplicatively
        weights_all_model = weights_ca * weights_sa  # Shape: (fusion_num, batch, channel, D, H, W)
        # Apply softmax across the fusion paths to normalize the weights
        weights_all_model = self.softmax(weights_all_model)  # Shape: (fusion_num, batch, channel, D, H, W)

        # Apply the weights to each split feature map and sum them
        weighted_sum = (split_I_IN_F * weights_all_model).sum(dim=0)  # Shape: (batch, channel, D, H, W)
        # Project the original input and the weighted sum, then add them together
        F_OUT = self.proj(I_IN) + self.projf(weighted_sum)  # Shape: (batch, ochannel, D, H, W)

        # Ensure contiguous memory layout
        return F_OUT.contiguous()

# 渐进分类
class CFE_refine(nn.Module):
    def __init__(self, in_channels=[64, 128, 256, 512, 1024, 1024], factor=1):
        super(CFE_refine, self).__init__()

        self.block_1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels[0],
                      out_channels=in_channels[1],
                      kernel_size=3,
                      padding=1,
                      stride=1,
                      bias=False),
            nn.BatchNorm3d(in_channels[1]),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        self.block_2 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels[1] * 2,
                      out_channels=in_channels[2],
                      kernel_size=3,
                      padding=1,
                      stride=1,
                      bias=False),
            nn.BatchNorm3d(in_channels[2]),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3,stride=2,padding=1)
        )

        self.block_3 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels[2] * 2,
                      out_channels=in_channels[3],
                      kernel_size=3,
                      padding=1,
                      stride=1,
                      bias=False),
            nn.BatchNorm3d(in_channels[3]),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3,stride=2, padding=1)
        )

        self.block_4 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels[3] * 2,
                      out_channels=in_channels[4],
                      kernel_size=3,
                      padding=1,
                      stride=1,
                      bias=False),
            nn.BatchNorm3d(in_channels[4]),
            nn.ReLU()
        )

        self.block_5 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels[4] * 2,
                      out_channels=in_channels[5]//factor,
                      kernel_size=3,
                      padding=1,
                      stride=1,
                      bias=False),
            nn.BatchNorm3d(in_channels[5]),
            nn.ReLU()
        )

        self.block_6 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels[4] * 2,
                      out_channels=in_channels[5],
                      kernel_size=3,
                      padding=1,
                      stride=1,
                      bias=False),
            nn.BatchNorm3d(in_channels[5]//factor),
            nn.ReLU()
        )

    def forward(self, features_t1, features_t2, features_t3,  features_t4, features_t5):
        x_t1 = features_t1
        x_t1 = self.block_1(x_t1)

        x_t2 = torch.cat((x_t1, features_t2), 1)
        x_t2 = self.block_2(x_t2)

        x_t3 = torch.cat((x_t2, features_t3), 1)
        x_t3 = self.block_3(x_t3)

        x_t4 = torch.cat((x_t3, features_t4), 1)
        x_t4 = self.block_4(x_t4)

        Fc = torch.cat((x_t4, features_t5), 1)

        Fs1 = self.block_5(Fc)
        Fs2 = self.block_6(Fc)
        return Fs1, Fs2

    def getFc(self, features_t1, features_t2, features_t3,  features_t4, features_t5):
        x_t1 = features_t1
        x_t1 = self.block_1(x_t1)

        x_t2 = torch.cat((x_t1, features_t2), 1)
        x_t2 = self.block_2(x_t2)

        x_t3 = torch.cat((x_t2, features_t3), 1)
        x_t3 = self.block_3(x_t3)

        x_t4 = torch.cat((x_t3, features_t4), 1)
        x_t4 = self.block_4(x_t4)

        Fc = torch.cat((x_t4, features_t5), 1)
        return Fc

class UFE_E2(nn.Module):
    def __init__(self, in_channels=[3, 64, 128, 256, 512, 1024], factor=1):
        super(UFE_E2, self).__init__()

        # 160, 192, 160
        # 和E1中的第一个卷积一样
        self.block_1 = nn.Sequential(
            nn.Conv3d(in_channels[0],
                      in_channels[1],
                      kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm3d(in_channels[1]),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        # 40, 46, 40
        self.block_2 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels[1] + 3,
                      out_channels=in_channels[2],
                      kernel_size=3,
                      padding=1,
                      stride=1,
                      bias=False),
            nn.BatchNorm3d(in_channels[2]),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        # 20, 24, 20
        self.block_3 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels[2],
                      out_channels=in_channels[3],
                      kernel_size=3,
                      padding=1,
                      stride=1,
                      bias=False),
            nn.BatchNorm3d(in_channels[3]),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        # 10, 12, 10
        self.block_4 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels[3],
                      out_channels=in_channels[4],
                      kernel_size=3,
                      padding=1,
                      stride=1,
                      bias=False),
            nn.BatchNorm3d(in_channels[4]),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        # 5, 6, 5
        self.block_5 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels[4],
                      out_channels=in_channels[5]//factor,
                      kernel_size=3,
                      padding=1,
                      stride=1,
                      bias=False),
            nn.BatchNorm3d(in_channels[5]//factor),
            nn.ReLU()
        )
        self.activation = nn.ReLU()

    def _Projection3D(self, Fl_prime, Fl):
        # Get dimensions: batch size (b), channels (c), width (w), height (h), depth (l)
        b, c, w, h, l = Fl.size()
        # Flatten the spatial dimensions for both features: [b, c, w*h*l]
        Fl_prime = Fl_prime.view(b, c, w * h * l)
        Fl = Fl.view(b, c, w * h * l)
        # Compute the dot product between Fl' and Fl for each batch and channel: numerator of the projection scalar
        numerator = torch.sum(Fl_prime * Fl, dim=-1, keepdim=True)
        # Compute the dot product of Fl with itself: denominator of the projection scalar
        denominator = torch.sum(Fl * Fl, dim=-1, keepdim=True) + 1e-6  # Add epsilon to prevent division by zero
        # Calculate the projection scalar and multiply by Fl to get the projection vector
        projection = numerator / denominator * Fl
        # Subtract the projection from Fl' to obtain the component of Fl' orthogonal to Fl
        out = Fl_prime - projection
        # Reshape the output back to the original spatial dimensions
        out = out.view(b, c, w, h, l)
        # Apply any additional processing (e.g., activation function)
        out = self.activation(out)
        return out

    def forward(self, input, Fl, condition):
        x_t1 = self.block_1(input)
        x_t1_proj = self._Projection3D(x_t1, Fl)

        condition = F.one_hot(condition.long(), 3)
        conditionAdd1 = condition.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1). \
            repeat(1, 1, x_t1_proj.size(2),  x_t1_proj.size(3), x_t1_proj.size(4))
        x_t1 = torch.cat((x_t1_proj, conditionAdd1), 1)

        x_t2 = self.block_2(x_t1)
        x_t3 = self.block_3(x_t2)
        x_t4 = self.block_4(x_t3)
        Fuhc_prime = self.block_5(x_t4)
        return Fuhc_prime

