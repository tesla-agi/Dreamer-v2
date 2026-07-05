import crafter
import gym
import numpy as np
import imageio

def watch_crafter(num_steps=200,out_path="crafter_run.gif", fps=15):
    env=gym.make("CrafterReward-v1")
    obs=env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    frames = [obs]
    for _ in range(num_steps):
        action=env.action_space.sample()
        step_out=env.step(action)
        if len(step_out)==5:
            obs,reward,terminated,truncated,info=step_out
            done=terminated or truncated
        else:
            obs,reward,done,info=step_out
        frames.append(obs)
        if done:
            obs=env.reset()
            if isinstance(obs,tuple):
                obs=obs[0]
            frames.append(obs)
    env.close()
    frames=[np.asarray(f,dtype=np.uint8) for f in frames]
    imageio.mimsave(out_path,frames,fps=fps)
    print(f"Saved {len(frames)} frames to {out_path}")

if __name__=="__main__":
    watch_crafter(num_steps=200)