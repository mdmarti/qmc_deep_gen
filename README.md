# qmc_deep_gen
# Official PyTorch implementation of "Quasi-Monte Carlo Methods Enable Extremely Low-Dimensional Latent Variable Models" [(ICLR 2026)](https://arxiv.org/abs/2601.18676)
branch for cleaning up code. this will be the final version


<div align="center">
  <a href="https://mdmarti.github.io" target="_blank">Miles&nbsp;Martinez</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://alexhwilliams.info/" target="_blank">Alex&nbsp;Williams</a> 
</div>
<br>
<br>

# TO-DO

- [ ] clean instructions for data loading
- [ ] create type specs and docstrings for all files
  - [ ] analysis
    - [x] geodesics
    - [x] clustering
    - [ ] jacobians
    - [ ] weighted_mean_shift
  - [ ] data
    - [ ] bird_data
    - [ ] dynamics
    - [ ] mocap
    - [ ] toy
    - [ ] utils
  - [ ] models
    - [ ] layers
    - [ ] qmc_base -> qlvm_base (fix imports too)
    - [ ] sampling
    - [ ] utils
    - [ ] vae_base
  - [ ] plotting
    - [ ] visualize (only include loss plots, latent grid vis, and embeddings)
  - [ ] train
    - [ ] losses
    - [ ] model_saving_loading
    - [ ] train (only need normal version)
    - [ ] train_vae
- [ ] remove unnecessary training and eval files; just include an example training script for mnist
- [ ] update README to be more professional lol. use [NVAE](https://github.com/NVlabs/NVAE) as template. 
