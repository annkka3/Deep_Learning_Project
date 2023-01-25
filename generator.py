
import torch
import torch.nn as nn

class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_norm: bool = True,
        last_layer: bool = False
    ):
        super().__init__()
        stride = 1 if last_layer else 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect" ),
            nn.BatchNorm2d(out_channels) if use_norm else nn.Identity(),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self,x):
        x = self.conv(x)
        return x

class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_norm: bool = True,
        last_layer: bool = False,
        use_dropout: bool = False
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels) if use_norm else nn.Identity(),
            nn.ReLU()
        )
        self.use_dropout = use_dropout
        self.drop = nn.Dropout(0.5)
        
    def forward(self,x):
        x = self.conv(x)
        return self.drop(x) if self.use_dropout else x

class Generator(nn.Module):
    def __init__(self, in_channels: int=3):
        super().__init__()
        self.input_channels = input_channels
        self.down1 = DownBlock(input_channels, 64, use_norm=False)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)
        self.down5 = DownBlock(512, 512)
        self.down6 = DownBlock(512, 512)
        self.down7 = DownBlock(512, 512)
        
        self.bottleneck = nn.Sequential(
                          nn.Conv2d(512, 512, 4, 2, 1),
                          nn.ReLU()
        )
        
        self.up1 = UpBlock(512, 512, use_dropout=True)
        self.up2 = UpBlock(1024, 512, use_dropout=True)
        self.up3 = UpBlock(1024, 512, use_dropout=True)
        self.up4 = UpBlock(1024, 512)
        self.up5 = UpBlock(1024, 256)
        self.up6 = UpBlock(512, 128)
        self.up7 = UpBlock(256, 64)
        self.up8 = nn.Sequential(
                   nn.ConvTranspose2d(128, input_channels, 4, 2, 1),
                   nn.Tanh()
        )
        
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        
        bt = self.bottleneck(d7)
        
        up1 = self.up1(bt)
        up2 = self.up2(torch.concat([up1, d7], dim=1))
        up3 = self.up3(torch.concat([up2, d6], dim=1))
        up4 = self.up4(torch.concat([up3, d5], dim=1))
        up5 = self.up5(torch.concat([up4, d4], dim=1))
        up6 = self.up6(torch.concat([up5, d3], dim=1))
        up7 = self.up7(torch.concat([up6, d2], dim=1))
        up8 = self.up8(torch.concat([up7, d1], dim=1))
        return up8

def test():
    x = torch.randn((1, 3, 256, 256))
    model = Generator()
    preds = model(x)
    print(preds.shape)
