import torch
from world_model import WorldModel
from actor import Actor
from critic import Critic

def imagine_rollout(world_model,actor,start_h,start_s,horizon=15,base_gamma=0.999):
    h=start_h.detach()                              # seed cut from world model — WM is frozen this phase
    s=start_s.detach()

    h_list=[]
    s_list=[]
    action_list=[]
    discount_list=[]
    reward_list=[]

    for _ in range(horizon):
        h_list.append(h)
        s_list.append(s)

        action_dist=actor(h,s)
        action=action_dist.sample()
        action_list.append(action)

        latent=torch.cat([h,s],dim=-1)
        reward=world_model.reward_head(latent).squeeze(-1)


        discount_logit=world_model.discount_head(latent).squeeze(-1)
        discount=base_gamma*torch.sigmoid(discount_logit)
        reward_list.append(reward)
        discount_list.append(discount)

        h,s,_=world_model.rssm.imagine_step(h,s,action)

    h_seq=torch.stack(h_list,dim=0)
    s_seq=torch.stack(s_list,dim=0)
    action_seq=torch.stack(action_list,dim=0)
    reward_seq=torch.stack(reward_list,dim=0)
    discount_seq=torch.stack(discount_list,dim=0)




    return {
        'h_seq':h_seq,
        's_seq':s_seq,
        'action_seq':action_seq,
        'reward_seq':reward_seq,
        'discount_seq':discount_seq
    }

def lambda_returns(rewards,values,discounts,gamma=0.99,lam=0.95):
    H=rewards.shape[0]
    returns=[None]*H

    returns[H-1]=values[H-1]
    for t in range(H-2,-1,-1):
        returns[t]=rewards[t]+discounts[t]*((1-lam)*values[t+1]+lam*returns[t+1])

    return torch.stack(returns,dim=0)

def compute_loss(actor,critic,target_critic,rollout,entropy_coef=3e-4,lam=0.95):
    h_seq=rollout['h_seq']
    s_seq=rollout['s_seq']
    action_seq=rollout['action_seq']
    reward_seq=rollout['reward_seq']
    discount_seq=rollout['discount_seq']

    v_target=target_critic(h_seq,s_seq)
    v_live=critic(h_seq,s_seq)

    r_lam=lambda_returns(reward_seq,v_target,discount_seq,lam=lam)


    #ACTOR LOSS
    dist=actor(h_seq,s_seq)
    log_prob=dist.log_prob(action_seq)
    entropy=dist.entropy()
    advantage=(r_lam-v_live).detach()
    L_actor=-(log_prob*advantage+entropy_coef*entropy).mean()

    #CRITIC LOSS
    L_critic=0.5*((v_live-r_lam.detach())**2).mean()

    return L_actor,L_critic



if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # --- instantiate (random weights are fine for a structural test) ---
    world_model= WorldModel().to(device)
    actor=Actor().to(device)
    critic=Critic().to(device)
    target_critic=Critic().to(device)
    target_critic.load_state_dict(critic.state_dict())
    for p in target_critic.parameters():
        p.requires_grad=False

    # --- seed + rollout ---
    start_h = torch.randn(8,600,device=device)
    start_s = torch.randn(8,1024,device=device)
    out = imagine_rollout(world_model,actor,start_h,start_s,horizon=15)

    # --- shape sanity: log_prob must reduce the one-hot class dim to (H, B) ---
    dist = actor(out['h_seq'],out['s_seq'])
    print("log_prob shape:",tuple(dist.log_prob(out['action_seq']).shape))   # expect (15, 8)

    # --- losses finite ---
    L_actor,L_critic=compute_loss(actor,critic,target_critic,out)
    print(f"L_actor :{L_actor.item():.4f}finite:{torch.isfinite(L_actor).item()}")
    print(f"L_critic:{L_critic.item():.4f}finite:{torch.isfinite(L_critic).item()}")

    # --- gradient isolation: the two stop-gradients must hold ---
    L_actor.backward(retain_graph=True)
    print("actor  got grad from L_actor :", actor.fc1.weight.grad  is not None)   # True
    print("critic got grad from L_actor :", critic.fc1.weight.grad is not None)   # False

    actor.zero_grad(); critic.zero_grad()
    L_critic.backward()
    print("critic got grad from L_critic:", critic.fc1.weight.grad is not None)   # True
    print("actor  got grad from L_critic:", actor.fc1.weight.grad  is not None)   # False
