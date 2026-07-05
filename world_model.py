import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence, OneHotCategorical, Categorical

from encoder import Encoder
from decoder import Decoder
from reward_head import RewardHead
from discount_head import DiscountHead
from RSSM import RSSM

class WorldModel(nn.Module):
    def __init__(self,hidden_dim=600,groups=32,classes=32,
                 a_dim=17,out_channels=384,hidden_head=400):
        super().__init__()
        self.s_dim=groups*classes
        self.latent_dim=hidden_dim+self.s_dim
        self.groups=groups
        self.classes=classes



        self.encoder=Encoder(out_channels=out_channels)
        self.embed_dim=self.encoder.embed_dim
        self.rssm=RSSM(s_dim=self.s_dim,a_dim=a_dim,hidden_dim=hidden_dim,emb_dim=self.embed_dim,groups=groups,classes=classes)
        self.decoder=Decoder(latent_dim=self.latent_dim)
        self.reward_head=RewardHead(latent_dim=self.latent_dim,hidden_dim=hidden_head)
        self.discount_head=DiscountHead(latent_dim=self.latent_dim,hidden_dim=hidden_head)



    def observe(self,obs_seq,action_seq):
        B=obs_seq.shape[0]
        T=action_seq.shape[1]
        device=obs_seq.device

        obs_flat=obs_seq.reshape(B*T,64,64,3)                               #Before obs had (B,T,64,64,3)
        embed_seq=self.encoder(obs_flat).reshape(B,T,self.embed_dim)

        a_dim=action_seq.shape[-1]
        zeros_first=torch.zeros(B,1,a_dim,device=device)
        a_prev_seq=torch.cat([zeros_first,action_seq[:,:-1]],dim=1)

        h,s=self.rssm.init_state(B,device=device)

        h_list=[]
        s_list=[]
        posterior_logits_list=[]
        prior_logits_list=[]
        for t in range(T):
            a_t=a_prev_seq[:,t]
            embed_t=embed_seq[:,t]
            _,_,prior_logits=self.rssm.imagine_step(h_prev=h,s_prev=s,a_prev=a_t)
            h,s,posterior_logits=self.rssm.obs_step(h_prev=h,s_prev=s,a_prev=a_t,obs_embed=embed_t)
            h_list.append(h)
            s_list.append(s)
            prior_logits_list.append(prior_logits)
            posterior_logits_list.append(posterior_logits)

        h_seq=torch.stack(h_list,dim=1)
        s_seq=torch.stack(s_list,dim=1)
        prior_logits_seq=torch.stack(prior_logits_list,dim=1)
        posterior_logits_seq=torch.stack(posterior_logits_list,dim=1)

        latent_seq=torch.cat([h_seq,s_seq],dim=-1)
        latent_flat=latent_seq.reshape(B*T,self.latent_dim)

        obs_recon=self.decoder(latent_flat).reshape(B,T,3,64,64)
        reward_pred=self.reward_head(latent_flat).reshape(B,T,1)
        discount_pred=self.discount_head(latent_flat).reshape(B,T,1)

        return {
            'h_seq':h_seq,
            's_seq':s_seq,
            'prior_logits':prior_logits_seq,
            'posterior_logits':posterior_logits_seq,
            'obs_recon':obs_recon,
            'reward_pred':reward_pred,
            'discount_pred':discount_pred,

        }

    def compute_loss(self,obs_seq,action_seq,reward_seq,discount_seq,alpha=0.8,kl_weight=1.0):
        out=self.observe(obs_seq,action_seq)

        #Reconstruction Loss
        obs_target=obs_seq.float().permute(0,1,4,2,3)/255.0
        recon_loss=F.mse_loss(out['obs_recon'],obs_target)

        #Reward Loss
        reward_loss=F.mse_loss(out['reward_pred'].squeeze(-1),reward_seq)

        #Discount Loss
        discount_loss=F.binary_cross_entropy_with_logits(out['discount_pred'].squeeze(-1),discount_seq)

        #KL Balancing
        B=obs_seq.shape[0]
        T=obs_seq.shape[1]
        prior_logits_reshaped=out['prior_logits'].reshape(B,T,self.groups,self.classes)
        posterior_logits_reshaped=out['posterior_logits'].reshape(B,T,self.groups,self.classes)

        posterior_dist_sg=OneHotCategorical(logits=posterior_logits_reshaped.detach())
        prior_dist=OneHotCategorical(logits=prior_logits_reshaped)
        kl_term1 = alpha * kl_divergence(posterior_dist_sg, prior_dist).sum(dim=-1)

        posterior_dist=OneHotCategorical(logits=posterior_logits_reshaped)
        prior_dist_sg=OneHotCategorical(logits=prior_logits_reshaped.detach())
        kl_term2=(1-alpha)*kl_divergence(posterior_dist,prior_dist_sg).sum(dim=-1)

        total_kl=kl_term1+kl_term2
        kl_loss=total_kl.mean()

        total=recon_loss+reward_loss+discount_loss+kl_loss*kl_weight

        return {
            'recon_loss': recon_loss,
            'reward_loss': reward_loss,
            'discount_loss': discount_loss,
            'kl_loss': kl_loss,
            'total_loss': total,
        }




if __name__ == "__main__":
    print("=" * 60)
    print("WORLD MODEL — FULL CHECK")
    print("=" * 60)

    wm = WorldModel()
    B, T = 4, 50

    # ─── Dummy data ───────────────────────────────────────────
    obs_seq = torch.randint(0, 256, (B, T, 64, 64, 3), dtype=torch.uint8)
    action_seq = torch.randn(B, T, 17)
    reward_seq = torch.randn(B, T)
    discount_seq = torch.ones(B, T)  # all-continue

    # ─── Test 1: observe ──────────────────────────────────────
    print("\n[1] observe() output shapes")
    print("-" * 60)
    out = wm.observe(obs_seq, action_seq)

    expected_shapes = {
        'h_seq':            (B, T, 600),
        's_seq':            (B, T, 1024),
        'prior_logits':     (B, T, 1024),
        'posterior_logits': (B, T, 1024),
        'obs_recon':        (B, T, 3, 64, 64),
        'reward_pred':      (B, T, 1),
        'discount_pred':    (B, T, 1),
    }

    all_shapes_ok = True
    for k, expected in expected_shapes.items():
        actual = tuple(out[k].shape)
        ok = actual == expected
        all_shapes_ok &= ok
        mark = "✓" if ok else "✗"
        print(f"  {mark} {k:18s} {str(actual):25s} expected {expected}")

    # ─── Test 2: all finite ───────────────────────────────────
    print("\n[2] All outputs finite")
    print("-" * 60)
    all_finite = True
    for k, v in out.items():
        finite = torch.isfinite(v).all().item()
        all_finite &= finite
        mark = "✓" if finite else "✗"
        print(f"  {mark} {k:18s} finite={finite}")

    # ─── Test 3: posterior sample is one-hot per group ────────
    print("\n[3] Posterior sample is one-hot per group")
    print("-" * 60)
    s = out['s_seq'].reshape(B, T, 32, 32)
    group_sums = s.sum(dim=-1)             # (B, T, 32)
    ones_ok = torch.allclose(group_sums, torch.ones_like(group_sums))
    print(f"  {'✓' if ones_ok else '✗'} per-group sums ≈ 1.0 (got {group_sums.mean().item():.4f})")

    # ─── Test 4: compute_loss ─────────────────────────────────
    print("\n[4] compute_loss() values")
    print("-" * 60)
    losses = wm.compute_loss(obs_seq, action_seq, reward_seq, discount_seq)
    for k, v in losses.items():
        finite = torch.isfinite(v).item()
        mark = "✓" if finite else "✗"
        print(f"  {mark} {k:15s} {v.item():8.4f}  finite={finite}")

    # ─── Test 5: backward pass ────────────────────────────────
    print("\n[5] Backward pass")
    print("-" * 60)
    losses['total_loss'].backward()

    # Check gradients reached every sub-module
    components = {
        'encoder':        wm.encoder,
        'rssm':           wm.rssm,
        'decoder':        wm.decoder,
        'reward_head':    wm.reward_head,
        'discount_head':  wm.discount_head,
    }
    all_grads_ok = True
    for name, module in components.items():
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in module.parameters())
        all_grads_ok &= has_grad
        mark = "✓" if has_grad else "✗"
        print(f"  {mark} {name:15s} gradients flowing")

    # ─── Test 6: total param count ────────────────────────────
    print("\n[6] Parameter count")
    print("-" * 60)
    total = sum(p.numel() for p in wm.parameters())
    print(f"  Total: {total:,}")
    for name, module in components.items():
        n = sum(p.numel() for p in module.parameters())
        print(f"    {name:15s} {n:>12,}")

    # ─── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    all_ok = all_shapes_ok and all_finite and ones_ok and all_grads_ok
    print(f"  {'✅ WORLD MODEL PASSES' if all_ok else '❌ SOMETHING FAILED'}")
    print("=" * 60)
