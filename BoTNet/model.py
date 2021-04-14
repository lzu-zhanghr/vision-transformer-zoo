
import torch
from torch import nn
from torchvision import models
from torchsummaryX import summary

class MHSA(nn.Module):
    def __init__(self, height, width, dim, head):
        super(MHSA, self).__init__()

        self.head = head
        self.r_h = nn.Parameter(data=torch.randn(1, head, dim // head, 1, height), requires_grad=True)
        self.r_w = nn.Parameter(data=torch.randn(1, head, dim // head, width, 1), requires_grad=True)

        self.w_q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, bias=True)
        self.w_k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, bias=True)
        self.w_v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, bias=True)

        self.softmax = nn.Softmax(dim=-1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):

        batch, dim, height, width = x.size()
        q = self.w_q(x).view(batch, self.head, dim // self.head, -1).permute(0, 1, 3, 2)
        k = self.w_k(x).view(batch, self.head, dim // self.head, -1)
        v = self.w_v(x).view(batch, self.head, dim // self.head, -1)
        r = (self.r_h + self.r_w).view(1, self.head, dim // self.head, -1)

        content_position = torch.matmul(q, r)
        content_content = torch.matmul(q, k)
        energy = (content_content + content_position).view(batch, -1)
        attention = self.softmax(energy).view(batch, self.head, height * width, height * width)
        feature = torch.matmul(v, attention).view(batch, dim, height, width)
        out = self.pool(feature)

        return out

class Bottle(nn.Module):
    def __init__(self, height, width, dim, head, mhsa = None):
        super(Bottle, self).__init__()

        self.block0 = nn.Sequential(*[
            nn.Conv2d(in_channels=dim, out_channels=dim // 2, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(num_features=dim // 2, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            MHSA(height=height, width=width, dim=dim // 2, head=head) if mhsa \
                else nn.Conv2d(in_channels=dim // 2, out_channels=dim // 2, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(num_features=dim // 2, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(in_channels=dim // 2, out_channels=dim * 2, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(num_features=dim * 2, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        ])
        self.shortcut = nn.Sequential(*[
            nn.Conv2d(in_channels=dim, out_channels=dim * 2, kernel_size=1, stride=2, bias=True),
            nn.BatchNorm2d(num_features=dim * 2, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        ])

        self.block1 = nn.Sequential(*[
            nn.Conv2d(in_channels=dim * 2 , out_channels=dim // 2, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(num_features=dim // 2, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(in_channels=dim // 2, out_channels=dim // 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=dim // 2, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(in_channels=dim // 2, out_channels=dim * 2, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(num_features=dim * 2, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        ])

        self.block2 = nn.Sequential(*[
            nn.Conv2d(in_channels=dim * 2, out_channels=dim // 2, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(num_features=dim // 2, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(in_channels=dim // 2, out_channels=dim // 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=dim // 2, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(in_channels=dim // 2, out_channels=dim * 2, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(num_features=dim * 2, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        ])

    def forward(self, x):

        feature0 = x
        feature1 = self.block0(feature0)
        feature2 = self.block1(feature1)
        out = self.block2(feature2) + self.shortcut(feature0)

        return out

class BoT50(nn.Module):
    def __init__(self):
        super(BoT50, self).__init__()

        self.head = 4
        self.res = models.resnet50(pretrained=True)
        self.res.layer4 = Bottle(height=14, width=14, dim=1024, head=self.head, mhsa=True)

    def forward(self, x):

        out = self.res(x)
        return out

class BoT101(nn.Module):
    def __init__(self):
        super(BoT101, self).__init__()

        self.head = 4
        self.res = models.resnet101(pretrained=True)
        self.res.layer4 = Bottle(height=14, width=14, dim=1024, head=self.head, mhsa=True)

    def forward(self, x):

        out = self.res(x)
        return out

if __name__ == '__main__':
    model = BoT101()
    summary(model, torch.randn([6, 3, 224, 224]))
