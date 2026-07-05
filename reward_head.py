import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardHead(nn.Module):
    def __init__(self,latent_dim=1624,hidden_dim=400):
        super(RewardHead,self).__init__()

        self.fc1=nn.Linear(latent_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)
        self.fc3=nn.Linear(hidden_dim,hidden_dim)
        self.fc4=nn.Linear(hidden_dim,1)

    def forward(self,x):
        h=F.elu(self.fc1(x))
        h=F.elu(self.fc2(h))
        h=F.elu(self.fc3(h))
        h=self.fc4(h)
        return h


if __name__ == "__main__":
    head=RewardHead()
    x=torch.randn(8, 1624)
    out=head(x)
    print(f"Input:{x.shape}")
    print(f"Output:{out.shape}")
    print(f"Params:{sum(p.numel() for p in head.parameters()):,}")