import torch
import os
from tqdm import tqdm
from world_model import WorldModel
from replay_buffer import ReplayBuffer
from config import *
import torch.nn.functional as F

cfg=Config()
def train_world_model(
        buffer_path=cfg.buffer_path,
        checkpoint_path=cfg.checkpoint_dir,
        num_iteration=cfg.train_steps,
        batch_size=cfg.wm_batch_size,
        seq_len=cfg.seq_len,
        lr=cfg.wm_lr,
        adam_eps=cfg.adam_eps,
        grad_clip=cfg.grad_clip,
):
    device=cfg.device
    print(f"Using device: {device}")
    os.makedirs(checkpoint_path, exist_ok=True)

    print("Loading Buffer...")
    buffer=ReplayBuffer(obs_shape=(64,64,3),max_episodes=cfg.num_random_episodes,max_steps=cfg.max_steps)
    buffer.load(buffer_path)
    print(f"Loaded buffer from {buffer_path}")

    print('Creating World Model...')
    wm=WorldModel(hidden_dim=cfg.hidden_dim,
                  groups=cfg.groups,
                  classes=cfg.classes,
                  a_dim=cfg.action_dim,
                  out_channels=cfg.out_channels,
                  hidden_head=cfg.hidden_head).to(device)
    num_params=sum(p.numel() for p in wm.parameters())
    print(f"Total number of parameters: {num_params}")

    optimizer=torch.optim.Adam(wm.parameters(),lr=lr,eps=adam_eps)
    print("Training World Model...")
    for t in tqdm(range(num_iteration)):
        obs_np,action_np,reward_np,discount_np=buffer.sample_sequences(batch_size=batch_size,seq_len=seq_len)
        obs=torch.from_numpy(obs_np).to(device)
        obs=obs[:,:seq_len]
        action = F.one_hot(torch.from_numpy(action_np).long(),num_classes=cfg.action_dim).float().to(device)
        reward=torch.from_numpy(reward_np).float().to(device)
        discount=torch.from_numpy(discount_np).float().to(device)

        optimizer.zero_grad()
        losses=wm.compute_loss(obs_seq=obs,action_seq=action,reward_seq=reward,discount_seq=discount)
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(),grad_clip)
        optimizer.step()
        if t%10==0:
            tqdm.write(f"Iter{t:5d} | "
                  f"recon:{losses['recon_loss'].item():.4f} | "
                  f"reward:{losses['reward_loss'].item():.4f} | "
                  f"discount: {losses['discount_loss'].item():.4f} | "
                  f"kl: {losses['kl_loss'].item():.4f} | "
                  f"total:{losses['total_loss'].item():.4f}")

        if t%cfg.save_every==0:
            torch.save(wm.state_dict(),os.path.join(checkpoint_path,"wm.pth"))

    torch.save(wm.state_dict(),os.path.join(checkpoint_path,"wm.pth"))
    print(f"\nTraining complete! Final model saved to {checkpoint_path}")

if __name__ == "__main__":
    train_world_model(num_iteration=50)

