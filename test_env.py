import gym
import crafter

env=gym.make("CrafterReward-v1")
obs=env.reset()
action=env.action_space.sample()
obs,reward,done,info=env.step(action)

env.close()

print(reward)
print(type(reward))
print(done)
print(type(done))
print(info.keys())

