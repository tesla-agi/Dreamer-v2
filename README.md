# DreamerV2 (from scratch)

[One or two sentences: what this is. "A from-scratch implementation of DreamerV2 
(Hafner et al. 2020) trained on Crafter — a world model that learns to dream, and an 
actor-critic that trains entirely inside that dream." Say it in your own words.]

## Idea

[3-4 sentences on the core concept. Prompt yourself: why a world model at all? 
What does "training in imagination" mean? Why is it sample-efficient? This is the 
part that proves you get it.]

## Architecture

[Your ASCII component diagram — you've drawn this a dozen times. World model 
(encoder, RSSM with categorical latents, decoder, reward/discount heads) + agent 
(actor, critic, target critic). Include the ~23.6M param count.]

## How it works

[The two loops, briefly:
- World model: observe real sequences → KL-balanced loss → learns to reconstruct + predict
- Actor-critic: imagine forward from latent states → λ-returns → REINFORCE + entropy
Prompt: what's the V1→V2 change and why? (categorical latents, REINFORCE, target critic)]

## Files

[File map — one line each: config.py, replay_buffer.py, encoder/decoder/RSSM/heads, 
world_model.py, actor.py, critic.py, imagine.py, train_wm.py, train_policy.py, 
visualize_dream.py]

## Run

​```bash
python collect_data.py       # gather random rollouts
python train_wm.py           # train the world model
python train_policy.py       # train actor-critic in imagination
python visualize_dream.py    # decode imagined latents → see the dream
​```

## Status

[Honest note: built + verified end-to-end; undertrained on single-GPU/MPS, so 
dreams are coarse. Structural completeness, not SOTA performance.]

## Dream visualization

![dream](dream_vs_real.png)
[One line: top = real, bottom = dreamed. Note the green-mush = undertrained decoder.]
