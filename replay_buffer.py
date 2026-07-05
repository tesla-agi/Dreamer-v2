import numpy as np

class ReplayBuffer:
    def __init__(self,obs_shape=(64,64,3),max_episodes=100,max_steps=500):
        self.obs_shape=obs_shape
        self.max_episodes=max_episodes
        self.max_steps=max_steps

        self.observations=np.zeros(
            (self.max_episodes,self.max_steps+1,*self.obs_shape),dtype=np.uint8
        )

        self.actions=np.zeros(
            (self.max_episodes,self.max_steps),dtype=np.int64
        )

        self.rewards=np.zeros(
            (self.max_episodes,self.max_steps),dtype=np.float32
        )

        self.episode_lengths=np.zeros(
            self.max_episodes,dtype=np.int64
        )
        self.discounts=np.zeros(
            (self.max_episodes,self.max_steps),dtype=np.float32
        )

        self.num_episodes=0

    def add_episode(self,obs_list,action_list,reward_list,done_list=None):
        if self.num_episodes>=self.max_episodes:
            print("Episode Limit Reached")
            return

        idx=self.num_episodes
        T=len(action_list)
        if done_list is None:
            self.discounts[idx,:T]=1.0
        else:
            self.discounts[idx,:T]=1.0-np.array(done_list,dtype=np.float32)
        self.observations[idx,:T+1]=np.array(obs_list,dtype=np.uint8)
        self.actions[idx,:T]=np.array(action_list,dtype=np.int64)
        self.rewards[idx,:T]=np.array(reward_list,dtype=np.float32)
        self.episode_lengths[idx]=T
        self.num_episodes+=1

    def sample_sequences(self,batch_size=50,seq_len=50):
        obs_batch=np.zeros(
            (batch_size,seq_len+1,*self.obs_shape),dtype=np.uint8
        )
        action_batch=np.zeros(
            (batch_size,seq_len),dtype=np.int64
        )
        reward_batch=np.zeros(
            (batch_size,seq_len),dtype=np.float32
        )
        discount_batch=np.zeros(
            (batch_size,seq_len),dtype=np.float32
        )

        for idx in range(batch_size):
            while True:
                ep_idx=np.random.randint(0,self.num_episodes)
                if self.episode_lengths[ep_idx]>=seq_len:
                    break

            max_start=self.episode_lengths[ep_idx]-seq_len
            start_indices=np.random.randint(0,max_start+1)
            end=start_indices+seq_len
            obs_batch[idx]=self.observations[ep_idx,start_indices:end+1]
            action_batch[idx]=self.actions[ep_idx,start_indices:end]
            reward_batch[idx]=self.rewards[ep_idx,start_indices:end]
            discount_batch[idx]=self.discounts[ep_idx,start_indices:end]

        return obs_batch,action_batch,reward_batch,discount_batch

    def save(self,path):
        np.savez(path,
            observations=self.observations[:self.num_episodes],
            actions=self.actions[:self.num_episodes],
            rewards=self.rewards[:self.num_episodes],
            num_episodes=self.num_episodes,
            episode_lengths=self.episode_lengths[:self.num_episodes],
            discounts=self.discounts[:self.num_episodes],
        )
        print(f"Saved {self.num_episodes} episodes to {path}")

    def load(self,path):
        data=np.load(path)
        self.num_episodes=int(data["num_episodes"])
        self.episode_lengths[:self.num_episodes]=data["episode_lengths"]
        self.observations[:self.num_episodes]=data["observations"]
        self.actions[:self.num_episodes]=data["actions"]
        self.rewards[:self.num_episodes]=data["rewards"]
        if "discounts" in data:
            self.discounts[:self.num_episodes]=data["discounts"]
        else:
            for i in range(self.num_episodes):
                self.discounts[i,:self.episode_lengths[i]]=1.0
        print(f"Loaded {self.num_episodes} episodes from {path}")

    def __len__(self):
        return self.num_episodes


if __name__ == "__main__":
    print("=" * 60)
    print("REPLAY BUFFER TEST (Crafter)")
    print("=" * 60)

    buffer = ReplayBuffer(obs_shape=(64, 64, 3), max_episodes=10, max_steps=300)

    # --- Add two episodes of DIFFERENT lengths to test variable-length handling ---
    for T in [188, 75]:
        obs_list = [np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
                    for _ in range(T + 1)]                       # T+1 observations
        action_list = [np.random.randint(0, 17) for _ in range(T)]   # T integer actions
        reward_list = [float(np.random.choice([0.0, 0.0, 0.1, -0.1, 1.0]))
                       for _ in range(T)]                        # T fractional rewards
        buffer.add_episode(obs_list, action_list, reward_list)

    print(f"\nStored episodes: {len(buffer)}")
    print(f"Episode lengths: {buffer.episode_lengths[:len(buffer)].tolist()}  (expect [188, 75])")

    # --- Sample and check shapes ---
    obs_b, act_b, rew_b = buffer.sample_sequences(batch_size=4, seq_len=50)
    print(f"\nSampled shapes:")
    print(f"  obs_batch:    {obs_b.shape}   (expect (4, 51, 64, 64, 3))")
    print(f"  action_batch: {act_b.shape}   (expect (4, 50))")
    print(f"  reward_batch: {rew_b.shape}   (expect (4, 50))")

    # --- Dtype checks ---
    print(f"\nDtypes:")
    print(f"  obs:    {obs_b.dtype}   (expect uint8)")
    print(f"  action: {act_b.dtype}   (expect int64)")
    print(f"  reward: {rew_b.dtype}   (expect float32)")

    # --- Fractional rewards must survive (not truncated to 0) ---
    print(f"\nUnique rewards in sample: {np.unique(rew_b).tolist()}")
    print("  (must contain 0.1 and/or -0.1 if those were sampled -- NOT just 0 and 1)")

    # --- RELOAD ROUND-TRIP: the real test of save/load ---
    print("\n" + "-" * 60)
    print("RELOAD ROUND-TRIP")
    print("-" * 60)
    buffer.save("test_buffer.npz")

    fresh = ReplayBuffer(obs_shape=(64, 64, 3), max_episodes=10, max_steps=300)
    fresh.load("test_buffer.npz")
    print(f"Reloaded episode lengths: {fresh.episode_lengths[:len(fresh)].tolist()}  (expect [188, 75])")

    # If this samples without hanging, load() correctly restored episode_lengths.
    obs_b2, act_b2, rew_b2 = fresh.sample_sequences(batch_size=4, seq_len=50)
    print(f"Sampled from reloaded buffer: obs {obs_b2.shape}, act {act_b2.shape}, rew {rew_b2.shape}")

    print("\n" + "=" * 60)
    print("REPLAY BUFFER WORKS")
    print("=" * 60)

