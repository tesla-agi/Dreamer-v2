import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from replay_buffer import ReplayBuffer
from world_model import WorldModel

cfg=Config()

@torch.no_grad()
def visualize_dream(wm_path=cfg.wm_path,buffer_path=cfg.buffer_path,seed_steps=5,dream_steps=10,save_path="dream_vs_real.png"):
    device=cfg.device
    buffer=ReplayBuffer(obs_shape=(64,64,3),max_episodes=cfg.num_random_episodes,max_steps=cfg.max_steps)
    buffer.load(buffer_path)
    wm=WorldModel(hidden_dim=cfg.hidden_dim, groups=cfg.groups,classes=cfg.classes,a_dim=cfg.action_dim,out_channels=cfg.out_channels,hidden_head=cfg.hidden_head).to(device)
    wm.load_state_dict(torch.load(wm_path, map_location=device))
    wm.eval()
    total=seed_steps+dream_steps
    obs_np,action_np,_,_=buffer.sample_sequences(batch_size=1,seq_len=total)
    obs=torch.from_numpy(obs_np).to(device)[:,:total]
    action=torch.nn.functional.one_hot(torch.from_numpy(action_np).long(),num_classes=cfg.action_dim).float().to(device)
    B=1
    h,s=wm.rssm.init_state(B,device=device)
    obs_flat=obs.reshape(B*total,64,64,3)
    embed=wm.encoder(obs_flat).reshape(B,total,wm.embed_dim)
    a_prev=torch.cat([torch.zeros(B,1,cfg.action_dim,device=device),action[:, :-1]],dim=1)
    dreamed_latents=[]
    for t in range(seed_steps):
        h,s,_=wm.rssm.obs_step(h_prev=h,s_prev=s,a_prev=a_prev[:,t],obs_embed=embed[:,t])
        dreamed_latents.append(torch.cat([h,s],dim=-1))
    for t in range(seed_steps,total):
        h,s,_=wm.rssm.imagine_step(h_prev=h,s_prev=s,a_prev=a_prev[:,t])
        dreamed_latents.append(torch.cat([h, s],dim=-1))
    dreamed_latents=torch.cat(dreamed_latents,dim=0)
    dreamed_frames=wm.decoder(dreamed_latents).permute(0,2,3,1).cpu().numpy()
    real_frames=obs[0].float().cpu().numpy()/255.0
    fig,axes=plt.subplots(2,total,figsize=(total*1.4,3.2))
    for t in range(total):
        axes[0,t].imshow(np.clip(real_frames[t],0,1))
        axes[0,t].axis("off")
        axes[1,t].imshow(np.clip(dreamed_frames[t],0,1))
        axes[1,t].axis("off")
        title="seed" if t < seed_steps else "dream"
        axes[0,t].set_title(f"{t}\n{title}",fontsize=7)
    fig.suptitle(f"Real vs Dreamed (seed={seed_steps},then open-loop dream)",fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path,dpi=120,bbox_inches="tight")
    print(f"Saved comparison to {save_path}")
    plt.close(fig)

if __name__ == "__main__":
    visualize_dream(seed_steps=5,dream_steps=10)