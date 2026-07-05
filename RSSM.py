import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical

from GRU import GRU

class RSSM(nn.Module):
    def __init__(self,s_dim=1024,a_dim=17,hidden_dim=600,emb_dim=384*4*4,groups=32,classes=32):
        super().__init__()
        self.s_dim=s_dim
        self.hidden_dim=hidden_dim
        self.emb_dim=emb_dim
        self.a_dim=a_dim
        self.groups=groups
        self.classes=classes

        self.gru=GRU(s_dim+a_dim,hidden_dim)
        self.prior_net=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,self.s_dim),
        )

        self.posterior_net=nn.Sequential(
            nn.Linear(emb_dim+hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,self.s_dim),
        )

    def sample_ste(self,logits):
        logits=logits.reshape(-1,self.groups,self.classes)
        dist=OneHotCategorical(logits=logits)
        probs=torch.softmax(logits,dim=-1)
        hard=dist.sample()
        sample=hard+probs-probs.detach()
        sample=sample.reshape(sample.size(0),-1)
        return sample

    def init_state(self,batch_size,device='cpu'):
        h_prev=torch.zeros(batch_size,self.hidden_dim,device=device)
        s_prev=torch.zeros(batch_size,self.s_dim,device=device)

        return h_prev,s_prev

    def imagine_step(self,h_prev,s_prev,a_prev):
        gru_input=torch.cat([s_prev,a_prev],dim=-1)
        h_t=self.gru(gru_input,h_prev)
        prior_logits=self.prior_net(h_t)
        s_t=self.sample_ste(prior_logits)
        return h_t,s_t,prior_logits

    def obs_step(self,h_prev,s_prev,a_prev,obs_embed):
        gru_input=torch.cat([s_prev,a_prev],dim=-1)
        h_t=self.gru(gru_input,h_prev)
        posterior_input=torch.cat([h_t,obs_embed],dim=-1)
        posterior_logits=self.posterior_net(posterior_input)
        s_t=self.sample_ste(posterior_logits)
        return h_t,s_t,posterior_logits




if __name__ =="__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    HIDDEN_DIM, GROUPS, CLASSES = 600, 32, 32
    S_DIM, A_DIM, EMBED_DIM = 1024, 17, 6144
    B = 8
    print("=" * 60)
    print("RSSM TEST")
    print("=" * 60)
    rssm = RSSM(
        hidden_dim=HIDDEN_DIM,groups=GROUPS,classes=CLASSES,
        s_dim=S_DIM,a_dim=A_DIM,emb_dim=EMBED_DIM,
    ).to(device)
    # [1] init_state
    h0,s0=rssm.init_state(B,device=device)
    print(f"\n[1] init_state")
    print(f"h0:{tuple(h0.shape)}(expect({B},{HIDDEN_DIM}))")
    print(f"s0:{tuple(s0.shape)}(expect ({B},{S_DIM}))")

    # [2] obs_step
    a=torch.randn(B,A_DIM,device=device)
    e=torch.randn(B,EMBED_DIM,device=device)
    h_t,s_t,post_logits=rssm.obs_step(h0,s0,a,e)
    print(f"\n[2] obs_step")
    print(f"h_t:{tuple(h_t.shape)}(expect({B},{HIDDEN_DIM}))")
    print(f"s_t:{tuple(s_t.shape)}(expect ({B},{S_DIM}))")
    print(f"post_logits:{tuple(post_logits.shape)}(expect({B},{S_DIM}))")
    print(f"per-group sums ~1:{torch.allclose(s_t.view(B,GROUPS,CLASSES).sum(-1),torch.ones(B,GROUPS,device=device),atol=1e-5)}")

    # [3] imagine_step
    h_t2, s_t2, prior_logits = rssm.imagine_step(h0, s0, a)
    print(f"\n[3] imagine_step")
    print(f"  h_t:          {tuple(h_t2.shape)}")
    print(f"  s_t:          {tuple(s_t2.shape)}")
    print(f"  prior_logits: {tuple(prior_logits.shape)}")
    print(f"  h_t matches obs_step h_t: {torch.allclose(h_t, h_t2)}   (must be True)")

    # [4] 50-step mixed rollout
    h, s = rssm.init_state(B, device=device)
    for _ in range(25):
        h, s, _ = rssm.obs_step(h, s, torch.randn(B, A_DIM, device=device), torch.randn(B, EMBED_DIM, device=device))
    for _ in range(25):
        h, s, _ = rssm.imagine_step(h, s, torch.randn(B, A_DIM, device=device))
    print(f"\n[4] 50-step rollout")
    print(f"  h finite: {torch.isfinite(h).all().item()}   s finite: {torch.isfinite(s).all().item()}")
    print(f"  h range:  [{h.min().item():.3f}, {h.max().item():.3f}]")
    print(f"  per-group sums ~1: {torch.allclose(s.view(B, GROUPS, CLASSES).sum(-1), torch.ones(B, GROUPS, device=device), atol=1e-5)}")

    # [5] gradient flow
    rssm.zero_grad()
    h, s = rssm.init_state(B, device=device)
    e_in = torch.randn(B, EMBED_DIM, device=device, requires_grad=True)
    h, s, post_logits = rssm.obs_step(h, s, torch.randn(B, A_DIM, device=device), e_in)
    h, s, prior_logits = rssm.imagine_step(h, s, torch.randn(B, A_DIM, device=device))
    (s.sum() + post_logits.sum() + prior_logits.sum()).backward()
    print(f"\n[5] gradient flow")
    print(f"  all params have grad: {all(p.grad is not None for p in rssm.parameters())}")
    print(f"  obs_embed grad flows: {e_in.grad is not None and torch.isfinite(e_in.grad).all().item()}   (STE → encoder)")

    print(f"\nTotal params: {sum(p.numel() for p in rssm.parameters()):,}")
    print("=" * 60)
    print("RSSM WORKS")
    print("=" * 60)