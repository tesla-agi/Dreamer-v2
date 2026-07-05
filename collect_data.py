import gym
import numpy as np
import crafter
from replay_buffer import ReplayBuffer
from tqdm import tqdm
import os



def collect_random_episodes(num_episodes=100,save_path='data/random_buffer.npz'):
    env=gym.make("CrafterReward-v1")
    buffer=ReplayBuffer(
        obs_shape=(64,64,3),
        max_episodes=num_episodes,
        max_steps=500
    )
    all_ep_rewards=[]
    os.makedirs(os.path.dirname(save_path),exist_ok=True)

    for episode in tqdm(range(num_episodes)):
        obs=env.reset()
        ep_obs=[obs]
        ep_action=[]
        ep_reward=[]

        done=False
        step=0
        while not done:
            action=env.action_space.sample()
            obs,reward,done,info=env.step(action)
            ep_obs.append(obs)
            ep_action.append(action)
            ep_reward.append(reward)
            step+=1
        buffer.add_episode(ep_obs,ep_action,ep_reward)
        all_ep_rewards.append(sum(ep_reward))

    env.close()
    buffer.save(save_path)
    lengths=buffer.episode_lengths[:buffer.num_episodes]
    print("\n"+"="*60)
    print("COLLECTION COMPLETE")
    print("="*60)
    print(f"Episodes collected: {buffer.num_episodes}")
    print(f"Reward  -> mean: {np.mean(all_ep_rewards):.2f} | "
          f"min: {np.min(all_ep_rewards):.2f} | "
          f"max: {np.max(all_ep_rewards):.2f}")
    print(f"Length -> mean: {lengths.mean():.1f} | "
          f"min: {lengths.min()} | "
          f"max: {lengths.max()}")
    too_short=int((lengths < 50).sum())
    print(f"Episodes shorter than seq_len=50: {too_short} "
          f"(buffer skips these at sample time)")
    print("="*60)



if __name__=="__main__":
    collect_random_episodes()