# DreamerV2 — from scratch, on Crafter

A from-scratch implementation of **DreamerV2** (Hafner et al., 2020) trained on the
[Crafter](https://github.com/danijar/crafter) environment. Every component — the
convolutional encoder/decoder, the RSSM with categorical latents, the hand-built GRU,
the straight-through categorical sampler, the actor-critic, and both training loops —
is written from the ground up in PyTorch. No high-level RL libraries.

The one idea behind the whole thing: **learn a compressed model of the world, then
train the policy entirely inside that model's imagination** — never touching the real
environment during policy learning. The agent dreams thousands of trajectories in
latent space for free.

---

## The idea

A model-free agent learns by acting in the real environment millions of times. A
**world model** agent instead learns a predictive model of the environment first, then
does its policy learning inside that model — in imagination.

Three moving parts make this work:

1. **Encode** high-dimensional pixel observations into a compact latent state.
2. **Predict** how that latent state evolves given actions (the "dream engine"), plus
   the reward and episode-continuation at each state.
3. **Imagine & learn** — roll the model forward from real states, let an actor choose
   actions and a critic score them, and train both on the imagined trajectories.

Because imagination is cheap (it's just latent-space matrix multiplies, no rendering,
no environment), the agent can train on far more experience than it could ever collect
in the real world. That is the sample-efficiency win.

---

## Architecture

Total world model: **23.62M parameters**. Actor: **0.82M**. Critic: **0.81M**.

```
                          ┌─────────────────────────────────────────────┐
                          │              WORLD MODEL (23.62M)             │
                          └─────────────────────────────────────────────┘

  obs (B,T,64,64,3) uint8
        │
        ▼
  ┌──────────────┐   4 conv layers, stride 2, channels 3→48→96→192→384
  │   ENCODER    │   spatial  64 → 32 → 16 → 8 → 4     flatten
  └──────────────┘
        │  embed  (B, T, 6144)
        ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │                              RSSM                                    │
  │                                                                     │
  │   deterministic path (hand-built GRU):                              │
  │       h_t = GRU(h_{t-1}, [s_{t-1}, a_{t-1}])        h : (B, 600)     │
  │                                                                     │
  │   stochastic path (32 groups × 32 classes categorical):             │
  │       posterior  obs_step(h, s, a, embed) →  s  (uses observation)  │
  │       prior      imagine_step(h, s, a)     →  s  (no observation)   │
  │       straight-through sampler             s : (B, 1024)            │
  └───────────────────────────────────────────────────────────────────┘
        │  latent = [h, s]  (B, T, 1624)
        ├──────────────┬──────────────┬──────────────┐
        ▼              ▼              ▼              
  ┌──────────┐   ┌──────────┐   ┌──────────────┐
  │ DECODER  │   │  REWARD  │   │   DISCOUNT   │
  │ 1624 →   │   │  HEAD    │   │    HEAD      │
  │ (3,64,64)│   │ 1624 → 1 │   │  1624 → 1    │
  └──────────┘   └──────────┘   └──────────────┘
   reconstruct     scalar r      continue-prob (sigmoid)

                          ┌─────────────────────────────────────────────┐
                          │            AGENT (actor-critic)               │
                          └─────────────────────────────────────────────┘

  latent (h, s)  (B, 1624)
        │
        ├──────────────────────────────┐
        ▼                              ▼
  ┌──────────────┐              ┌──────────────┐          ┌──────────────────┐
  │    ACTOR     │              │    CRITIC    │          │  TARGET CRITIC   │
  │ 1624→400→400 │              │ 1624→400→400 │  EMA →   │  frozen EMA copy │
  │   →17        │              │   →1         │ ◄──────  │  of the critic   │
  └──────────────┘              └──────────────┘          └──────────────────┘
  OneHotCategorical             V(s) scalar               V(s) for λ-returns
  over 17 actions               (trained)                 (stable bootstrap)
```

### Shape reference

| Tensor            | Shape                | Notes                                    |
|-------------------|----------------------|------------------------------------------|
| observation       | (B, T, 64, 64, 3)    | uint8, raw Crafter frames                |
| encoder embedding | (B, T, 6144)         | flattened 384 × 4 × 4                     |
| deterministic h   | (B, 600)             | GRU recurrent state                      |
| stochastic s      | (B, 1024)            | 32 groups × 32 classes, one-hot per group|
| latent [h, s]     | (B, 1624)            | input to decoder / heads / actor / critic|
| action            | (B, T, 17)           | one-hot over 17 discrete Crafter actions |
| reward / discount | (B, T)               | scalar per step                          |
| imagined rollout  | (H, B, ...)          | H = imagination horizon (15)             |

---

## How it works — the two training phases

### 1. World-model training (`train_wm.py`)

Sample 50-step sequences from the replay buffer, run them through `observe()` (which
rolls the RSSM computing both posterior and prior at each step), and minimize a
**KL-balanced ELBO**:

```
L_wm = recon_loss + reward_loss + discount_loss + kl_loss

recon    : MSE( decoded frame , real frame )
reward   : MSE( predicted reward , real reward )
discount : BCE( predicted continue-logit , continue-flag )
kl       : KL-balanced ( posterior ‖ prior ),  α = 0.8
```

**KL balancing** trains the prior to match the posterior faster than the reverse, using
stop-gradients on alternating sides:

```
kl = α · KL( sg(posterior) ‖ prior )  +  (1-α) · KL( posterior ‖ sg(prior) )
```

### 2. Actor-critic in imagination (`train_policy.py`)

Seed from the world model's posterior states, then **imagine** forward H steps in the
prior (no observations). Compute **λ-returns**, then update actor and critic:

```
λ-return (backward recursion, bootstrapped by the TARGET critic):
    Rλ_H   = V(s_H)
    Rλ_t   = r_t + γ_t · [ (1-λ)·V(s_{t+1}) + λ·Rλ_{t+1} ]

Actor  (REINFORCE + entropy):
    L_actor  = −[ log π(a_t) · sg(Rλ_t − V(s_t))  +  η · H(π) ]

Critic (regression to the λ-return):
    L_critic = ½ ( V(s_t) − sg(Rλ_t) )²

Target critic: slow EMA copy,  θ_tgt ← τ·θ_tgt + (1-τ)·θ,  τ = 0.98
```

### What changed from DreamerV1 → V2

| | V1 | V2 (this repo) |
|---|---|---|
| latents | Gaussian | categorical (32×32) |
| actions | continuous | discrete (17) |
| sampling | reparameterizable | not differentiable |
| actor gradient | analytic | **REINFORCE** + entropy |
| stability | — | **target critic** (EMA), KL balancing |

Discrete actions break the analytic gradient chain, forcing the switch to REINFORCE —
which is noisier, and therefore needs the target critic and entropy bonus to stay stable.

---

## Files

| File                 | Role                                                            |
|----------------------|-----------------------------------------------------------------|
| `config.py`          | central dataclass of all hyperparameters (derived `latent_dim`) |
| `replay_buffer.py`   | stores episodes, samples 50-step sequences                      |
| `encoder.py`         | conv encoder, obs → 6144 embedding                              |
| `decoder.py`         | conv decoder, latent → 64×64×3 reconstruction                   |
| `RSSM.py`            | hand-built GRU + straight-through categorical sampler; prior + posterior |
| `reward_head.py`     | latent → scalar reward (MSE)                                    |
| `discount_head.py`   | latent → continue-probability (BCE)                             |
| `world_model.py`     | wraps all of the above; `observe()` + KL-balanced `compute_loss`|
| `actor.py`           | latent → OneHotCategorical over 17 actions                      |
| `critic.py`          | latent → V(s); plus `update_target` EMA helper                  |
| `imagine.py`         | `imagine_rollout`, `lambda_returns`, `compute_loss` (actor+critic) |
| `train_wm.py`        | world-model training loop                                       |
| `train_policy.py`    | actor-critic training against the frozen world model            |
| `visualize_dream.py` | decode imagined latents to pixels, compare with real frames     |
| `collect_data.py`    | gather random rollouts into the replay buffer                   |

---

## Running

```bash
pip install -r requirements.txt

python collect_data.py       # 1. gather random rollouts → replay buffer
python train_wm.py           # 2. train the world model (the "dream engine")
python train_policy.py       # 3. train actor-critic inside imagination
python visualize_dream.py    # 4. decode the dream, compare to reality
```

Key pinned dependencies (Crafter requires them): `gym==0.23.1`, `numpy==1.26.4`.

---

## Dream visualization

The world model is seeded with 5 real observations (posterior), then imagines the next
10 steps open-loop (prior only, real actions), decoding each imagined latent back to
pixels.

![Real vs Dreamed](dream_vs_real.png)

*Top row: real frames. Bottom row: the world model's dream.*

In the image above the dream is coarse green mush — this is a **deliberately
undertrained** world model (a short training run on a single MacBook / MPS device). The
decoder has learned the dominant colour statistics of Crafter (grass-green world) but
not yet fine structure. With a full training run the dream row sharpens into recognizable
tiles, and the "seed" columns stay noticeably crisper than the late "dream" columns —
the visible signature of the posterior-vs-prior gap.

---

## Status

Built and verified end-to-end:

- ✅ every component shape-checked and gradient-tested in isolation
- ✅ world model trains (all five loss terms decrease)
- ✅ actor-critic pipeline runs end-to-end with verified gradient isolation
- ✅ dream decoding works (visualization above)

This is a **structural** implementation for understanding DreamerV2, not a
performance-tuned run. Reaching strong Crafter scores needs the full joint
collect→train→imagine loop plus substantial compute — a hardware matter, not a code one.

---

## References

- Hafner et al., 2020 — *Mastering Atari with Discrete World Models* (DreamerV2)
- Hafner et al., 2019 — *Dream to Control* (DreamerV1)
- Ha & Schmidhuber, 2018 — *World Models*
- Crafter: https://github.com/danijar/crafter
- 
