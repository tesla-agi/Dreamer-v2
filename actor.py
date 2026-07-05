import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical


class Actor(nn.Module):
    def __init__(self,latent_dim=1624,hidden_dim=400,action_dim=17):
        super(Actor,self).__init__()
        self.latent_dim=latent_dim
        self.hidden_dim=hidden_dim
        self.action_dim=action_dim

        self.fc1=nn.Linear(latent_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)
        self.fc3=nn.Linear(hidden_dim,action_dim)

    def forward(self,h,s):
        latent=torch.cat([h,s],dim=-1)
        x=F.relu(self.fc1(latent))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        dist=OneHotCategorical(logits=x)
        return dist



if __name__=='__main__':
    actor = Actor()
    h = torch.randn(8, 600)
    s = torch.randn(8, 1024)
    dist = actor(h, s)

    a = dist.sample()
    print("action shape:", a.shape)  # expect (8, 17)
    print("one-hot sums:", a.sum(-1))  # expect all 1.0
    print("log_prob shape:", dist.log_prob(a).shape)  # expect (8,)
    print("entropy shape:", dist.entropy().shape)  # expect (8,)

    # the gradient bridge — does the score-function path reach the actor?
    loss = dist.log_prob(a).sum()
    loss.backward()
    print("fc1 grad exists:", actor.fc1.weight.grad is not None)