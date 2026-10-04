# RFF-RND

**Mitigating Overgeneralization in RND via Spectral Target Design**

NeurIPS 2026

Minseok Jeong*<sup>1</sup>, Yechan Lee*<sup>1</sup>, Hyewon Choi<sup>1</sup>, Jeongyong Yang<sup>2</sup>, SooJean Han<sup>1,†</sup>

<sup>1</sup> KAIST · <sup>2</sup> University of Washington

<strong>*</strong> Co-first authors

[Paper](https://openreview.net/forum?id=yYvDH6yl2q)

## Abstract

Random Network Distillation (RND) is a scalable novelty signal for RL based on a fixed random target network and a trained predictor network. However, it can overgeneralize: the predictor extrapolates the random target off the data manifold, causing novelty scores to collapse on unfamiliar inputs. This paper analyzes this failure through a kernel-theoretic lens. Under a GP target model and an NTK-regime predictor, we show that the RND novelty score is a cross-kernel residual variance: it is governed jointly by the target covariance kernel and the predictor NTK, rather than by either one alone. This identifies overgeneralization as spectral under-excitation. Motivated by this view, we formulate spectral target design as a principle for shaping RND novelty geometry. As a concrete instance, we replace the implicit random-network target with a bandwidth-controlled Random Fourier Feature (RFF) target. Empirically, this redistribution increases residual signal in modes that are less attenuated by the predictor. Experiments on offline D4RL, online Atari, and Fetch manipulation benchmarks show improved novelty discrimination and downstream RL performance.

![RND and RFF-RND novelty landscapes](assets/novelty_landscapes.png)

## Structure

```text
RFF-RND/
├── README.md
├── environment.yml
├── assets/                     
├── offline/                    
│   ├── config/
│   ├── src/
│   └── train_offline_rl.py
└── online/                     
    ├── configs/
    ├── main.py
    └── run.sh
```

## Environment setup

From the repository root (`RFF-RND/`):

```bash
conda env create -f environment.yml
conda activate rff-rnd
```

Offline RL requires MuJoCo 2.1 installed and connected through `LD_LIBRARY_PATH`.

## Usage

Run from the repository root (`RFF-RND/`). Both commands read experiment settings from the specified config file; edit the config files to change the settings.

### Offline RL

Load the config file and start offline RL training.

Train Offline RL:

```bash
python offline/train_offline_rl.py offline/config/train_default_config.yaml
```

### Online RL

Load the config file and start online RL training.


```bash
python online/main.py online/configs/config_RFF_MontezumaRevenge.conf
```
