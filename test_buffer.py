from replay_buffer import ReplayBuffer
b = ReplayBuffer(max_episodes=100, max_steps=500)
b.load("data/random_buffer.npz")
o, a, r, d = b.sample_sequences(batch_size=4, seq_len=50)
print(o.shape, a.shape, r.shape, d.shape)
print("discount range:", d.min(), d.max())