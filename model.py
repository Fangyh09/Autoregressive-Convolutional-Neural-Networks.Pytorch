import torch
import torch.nn as nn
import torchvision.models as models
import torchsnooper



class SignificanceCNN(nn.Module):
    def __init__(self):
    # def __init__(self, embed_size, input_channels):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(SignificanceCNN, self).__init__()
        # self.conv0 = nn.Conv2d(1, 1, (1, 3), padding=(0,0))
        self.conv0 = nn.Conv3d(1, 64, (1, 3, 3), padding=(0, 1, 1))
        self.bn0 = nn.BatchNorm3d(64)
        self.lrelu = nn.LeakyReLU()

        self.conv1 = nn.Conv3d(64, 128, (1, 3, 3), padding=(0, 1, 1))
        self.bn1 = nn.BatchNorm3d(128)
        
        self.conv2 = nn.Conv3d(128, 128, (1, 3, 3), padding=(0, 1, 1))
        self.bn2 = nn.BatchNorm3d(128)

        self.conv3 = nn.Conv3d(128, 64, (1, 3, 3), padding=(0, 1, 1))
        self.bn3 = nn.BatchNorm3d(64)

        self.conv4 = nn.Conv3d(64, 1, (1, 3, 3), padding=(0, 1, 1))
        self.bn4 = nn.BatchNorm3d(1)

        self.conv5 = nn.Conv3d(1, 1, (1, 3, 3), padding=(0, 1, 1))
        self.bn5 = nn.BatchNorm3d(1)

    def forward(self, images):
        """Extract feature vectors from input images."""
        x0 = self.lrelu(self.bn0(self.conv0(images)))
        x1 = self.lrelu(self.bn1(self.conv1(x0)))
        x2 = self.lrelu(self.bn2(self.conv2(x1)))
        x3 = self.lrelu(self.bn3(self.conv3(x2)))
        x4 = self.lrelu(self.bn4(self.conv4(x3)))
        x5 = self.lrelu(self.bn5(self.conv5(x4)))

        # x1 = self.lrelu(self.bn1(self.conv1(x0)))
        out = x5
        return out



class OffsetCNN(nn.Module):
    def __init__(self):
    # def __init__(self, embed_size, input_channels):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(OffsetCNN, self).__init__()
        self.conv0 = nn.Conv3d(1, 1, (1, 1, 1), padding=(0,0,0))
        self.bn0 = nn.BatchNorm3d(1)
        self.lrelu = nn.LeakyReLU()

        
    def forward(self, images):
        """Extract feature vectors from input images."""
        x0 = self.lrelu(self.bn0(self.conv0(images)))
        out = x0
        return out


class EncoderCNN(nn.Module):
    def __init__(self):
    # def __init__(self, embed_size, input_channels):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        self.off_model = OffsetCNN()
        self.sig_model = SignificanceCNN()
        self.sigmoid = nn.Sigmoid()
        self.W = nn.Conv3d(1, 1, (5, 1, 1))

    @torchsnooper.snoop()
    def forward(self, images):
        """Extract feature vectors from input images."""
        off = self.off_model(images)
        sig = self.sig_model(images)
        out = off + images
        out = self.sigmoid(off) * out

        out = self.W(out)

        return out



