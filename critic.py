import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self,latent_dim=1624,hidden_dim=400):
        super(Critic,self).__init__()

        self.latent_dim=latent_dim
        self.hidden_dim=hidden_dim

        self.fc1=nn.Linear(latent_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)
        self.fc3=nn.Linear(hidden_dim,1)

    def forward(self,h,s):
        latent=torch.cat([h,s],dim=-1)
        x=F.relu(self.fc1(latent))
        x=F.relu(self.fc2(x))
        x=self.fc3(x).squeeze(-1)
        return x



def update_target(critic,target_critic,tau):                                #EMA
    for p,p_tgt in zip(critic.parameters(),target_critic.parameters()):
        p_tgt.data.mul_(tau).add_(p.data,alpha=(1-tau))


if __name__=='__main__':
    critic=Critic()
    target_critic=Critic()
    target_critic.load_state_dict(critic.state_dict())  # construct + sync
    for p in target_critic.parameters():
        p.requires_grad=False  # freeze

    # 1. after sync, they must be identical
    w0=critic.fc1.weight.clone()
    t0=target_critic.fc1.weight.clone()
    print("identical after sync:",torch.allclose(w0,t0))  # expect True

    # 2. frozen: target params must not require grad
    print("target frozen:",not any(p.requires_grad for p in target_critic.parameters()))  # True

    # 3. perturb the live critic, then EMA — target should move a LITTLE toward it
    with torch.no_grad():
        critic.fc1.weight.add_(1.0)  # shove live critic away
    update_target(critic,target_critic,tau=0.98)
    moved = (target_critic.fc1.weight-t0).abs().mean().item()
    gap = (critic.fc1.weight-target_critic.fc1.weight).abs().mean().item()
    print(f"target moved by ~{moved:.4f} (small, ~0.02)")  # expect ≈ 0.02, not 1.0
    print(f"still lags live critic by ~{gap:.4f} (large, ~0.98)")  # expect ≈ 0.98