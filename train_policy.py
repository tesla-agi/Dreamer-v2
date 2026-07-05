import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

from torch.optim import Adam
from config import *
from replay_buffer import ReplayBuffer
from world_model import WorldModel
from actor import Actor
from critic import Critic,update_target
from imagine import imagine_rollout,compute_loss


cfg=Config()


def train_policy(
        buffer_path=cfg.buffer_path,
        wm_path=cfg.wm_path,
        actor_path='checkpoint/actor.pth',
        critic_path='checkpoint/critic.pth',
        num_iterations=cfg.train_steps,
        batch_size=cfg.wm_batch_size,
        imagine_horizon=cfg.imagine_horizon,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.base_gamma,
        lam=cfg.lam,
        entropy_coef=cfg.entropy_coef,
        tau=cfg.tau,
        grad_clip=cfg.grad_clip,
        log_every=cfg.log_every,
        save_every=cfg.save_every,
):
    device=cfg.device
    print(f"Using device: {device}")

    os.makedirs(os.path.dirname(actor_path),exist_ok=True)
    os.makedirs(os.path.dirname(critic_path),exist_ok=True)

    buffer=ReplayBuffer(obs_shape=(64,64,3),max_episodes=cfg.num_random_episodes,max_steps=cfg.max_steps)
    buffer.load(buffer_path)
    print(f"Loaded{len(buffer)} episodes")

    wm=WorldModel(hidden_dim=cfg.hidden_dim,
                  groups=cfg.groups,
                  classes=cfg.classes,
                  a_dim=cfg.action_dim,
                  out_channels=cfg.out_channels,
                  hidden_head=cfg.hidden_head).to(device)
    wm.load_state_dict(torch.load(wm_path,map_location=device))
    wm.eval()
    for param in wm.parameters():
        param.requires_grad=False

    print("World Model loaded and frozen")

    actor=Actor(latent_dim=cfg.latent_dim,hidden_dim=cfg.ac_hidden_dim,action_dim=cfg.action_dim).to(device)
    critic=Critic(latent_dim=cfg.latent_dim,hidden_dim=cfg.ac_hidden_dim).to(device)
    target_critic=Critic(latent_dim=cfg.latent_dim,hidden_dim=cfg.ac_hidden_dim).to(device)
    target_critic.load_state_dict(critic.state_dict())
    for param in target_critic.parameters():
        param.requires_grad=False

    actor_params=sum(p.numel() for p in actor.parameters())
    critic_params=sum(p.numel() for p in critic.parameters())
    print(f"Actor params: {actor_params}")
    print(f"Critic params: {critic_params}")

    actor_optimizer=Adam(actor.parameters(),lr=actor_lr)
    critic_optimizer=Adam(critic.parameters(),lr=critic_lr)


    for t in tqdm(range(num_iterations)):
        obs_np,action_np,_,_=buffer.sample_sequences(batch_size,seq_len=cfg.seq_len)
        obs=torch.from_numpy(obs_np).to(device)
        obs=obs[:,:cfg.seq_len]
        action=F.one_hot(torch.from_numpy(action_np).long(),num_classes=cfg.action_dim).float().to(device)

        with torch.no_grad():
            wm_out=wm.observe(obs,action)
            h_real=wm_out['h_seq']
            s_real=wm_out['s_seq']

        B,T=h_real.shape[0],h_real.shape[1]
        start_h=h_real.reshape(B*T,-1)
        start_s=s_real.reshape(B*T,-1)

        out=imagine_rollout(
            world_model=wm,
            actor=actor,
            start_h=start_h,
            start_s=start_s,
            horizon=imagine_horizon,
            base_gamma=gamma
        )

        L_actor,L_critic=compute_loss(actor,critic,target_critic,out,entropy_coef=entropy_coef,lam=lam)

        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        (L_actor+L_critic).backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(),grad_clip)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
        actor_optimizer.step()
        critic_optimizer.step()

        update_target(critic=critic,target_critic=target_critic,tau=tau)

        if t%10==0:
            tqdm.write(
                f"Iter {t:5d} |"
                f"L_actor: {L_actor:5.2f} |"
                f"L_critic: {L_critic:5.2f} |"
            )

        if t%save_every==0:
            torch.save(actor.state_dict(),actor_path)
            torch.save(critic.state_dict(),critic_path)

    torch.save(actor.state_dict(),actor_path)
    torch.save(critic.state_dict(),critic_path)
    print(f"Training Complete "
          f"Saved Actor -> {actor_path}"
          f"Saved Critic -> {critic_path}")




if __name__=="__main__":
    train_policy(num_iterations=50)

