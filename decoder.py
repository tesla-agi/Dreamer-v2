import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self,latent_dim=1624):
        super(Decoder,self).__init__()

        self.fc=nn.Linear(latent_dim,384*4*4)
        self.deconv1=nn.ConvTranspose2d(384,192,kernel_size=4,stride=2,padding=1)
        self.deconv2=nn.ConvTranspose2d(192,96,kernel_size=4,stride=2,padding=1)
        self.deconv3=nn.ConvTranspose2d(96,48,kernel_size=4,stride=2,padding=1)
        self.deconv4=nn.ConvTranspose2d(48,3,kernel_size=4,stride=2,padding=1)


    def forward(self,x):
        h=F.relu(self.fc(x))
        h=h.reshape(-1,384,4,4)
        h=F.relu(self.deconv1(h))
        h=F.relu(self.deconv2(h))
        h=F.relu(self.deconv3(h))
        x_hat=torch.sigmoid(self.deconv4(h))
        return x_hat



if __name__=="__main__":
    dec=Decoder(latent_dim=1624)
    z=torch.randn(8,1624)
    out=dec(z)
    print(f"Input:{z.shape}(expect (8, 1624))")
    print(f"Output:{out.shape}(expect (8, 3, 64, 64))")
    print(f"Range:[{out.min():.3f},{out.max():.3f}](expect within [0,1] from sigmoid)")
    print(f"Params:{sum(p.numel() for p in dec.parameters()):,}")

