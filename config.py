import torch
from dataclasses import dataclass,field

@dataclass
class Config:
    env_name:str='CrafterReward-v1'
    action_dim:int=17
    img_size:int=64
    img_channels:int=3
    max_steps:int=500
    out_channels:int=384
    hidden_head:int=400
    imagine_horizon:int=15
    seq_len:int=50

    #RSSM
    hidden_dim:int=600
    groups:int=32
    classes:int=32
    s_dim:int=1024
    embed_dim:int=6144

    #WMLoss
    kl_alpha:float=0.8
    kl_free_nats:float=0.0
    recon_loss_scale:float=1.0
    discount_loss_scale:float=1.0

    #ActorCritic
    ac_hidden_dim:int=400
    base_gamma:float=0.99
    lam:float=0.95
    entropy_coef:float=3e-4
    tau:float=0.98

    #Optimization
    wm_lr:float=3e-4
    actor_lr:float=8e-5
    critic_lr:float=8e-5
    adam_eps:float=1e-5
    grad_clip:float=100.0

    wm_batch_size:int=16
    buffer_path:str='data/random_buffer.npz'
    num_random_episodes:int=100
    train_steps:int=100000
    checkpoint_dir:str='checkpoint/'
    seed:int=42
    wm_path:str='checkpoint/wm.pth'


    latent_dim:int=field(init=False)
    device:str=field(init=False)
    log_every:int=100
    save_every:int=5000

    def __post_init__(self):
        self.latent_dim=self.hidden_dim+self.s_dim
        self.device="mps" if torch.backends.mps.is_available() else "cpu"

        assert self.s_dim==self.groups*self.classes, \
            f"s_dim ({self.s_dim}) must equal groups*classes ({self.groups * self.classes})"

        assert 0.0<=self.kl_alpha<=1.0,"kl_alpha must be in [0, 1]"
        assert 0.0<=self.tau < 1.0, "tau must be in [0, 1)"




config=Config()


if __name__ == "__main__":
    c=Config()
    print("=== DreamerV2 Config ===")
    for k,v in c.__dict__.items():
        print(f"{k:22s}={v}")
    print("\nDerived latent_dim:", c.latent_dim,"(expect 1624)")
    print("Device:",c.device)
