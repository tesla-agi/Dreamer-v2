import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self,out_channels=384):
        super(Encoder,self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=48,kernel_size=4,stride=2,padding=1)
        self.conv2=nn.Conv2d(in_channels=48,out_channels=96,kernel_size=4,stride=2,padding=1)
        self.conv3=nn.Conv2d(in_channels=96,out_channels=192,kernel_size=4,stride=2,padding=1)
        self.conv4=nn.Conv2d(in_channels=192,out_channels=out_channels,kernel_size=4,stride=2,padding=1)

        self.embed_dim=out_channels*4*4

    def forward(self,x):           # x had dim = (B,H,W,C) and also value range from 0-255
        x=x.float()/255.0
        x=x.permute(0,3,1,2)              # x was changed to the dimension (B,C,H,W)
        h=F.relu(self.conv1(x))
        h=F.relu(self.conv2(h))
        h=F.relu(self.conv3(h))
        h=F.relu(self.conv4(h))
        h=h.reshape(h.size(0),-1)
        return h


if __name__ == "__main__":
    print("=" * 60)
    print("ENCODER TEST")
    print("=" * 60)

    enc = Encoder(out_channels=384)
    print(f"embed_dim (flattened output size): {enc.embed_dim}  (expect 6144)")

    # Fake batch of raw uint8 images, channels-last, like the buffer stores.
    B = 8
    x = torch.randint(0, 256, (B, 64, 64, 3), dtype=torch.uint8)
    print(f"\nInput:  {x.shape}  dtype={x.dtype}")

    out = enc(x)
    print(f"Output: {out.shape}  (expect (8, 6144))")
    print(f"Output dtype: {out.dtype}  (expect float32)")
    print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")

    n_params = sum(p.numel() for p in enc.parameters())
    print(f"\nTotal params: {n_params:,}")

    print("\n" + "=" * 60)
    print("ENCODER WORKS")
    print("=" * 60)
