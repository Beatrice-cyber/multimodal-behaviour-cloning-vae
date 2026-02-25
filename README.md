# Behaviour Cloning for Robotic Manipulation  
### Multimodal Learning with Self-Supervised Variational Autoencoder (PyTorch)

## Overview

This project explores behaviour cloning for robotic manipulation using multimodal observations.  
The goal is to predict continuous robotic arm control actions from:

- RGB visual inputs  
- Joint positions and velocities  
- End-effector state information  

The system combines self-supervised representation learning with supervised policy learning to improve downstream action prediction performance.

---

## Problem Setting

Given multimodal observations at time step *t*:

- Image frame (RGB)
- Robot joint states
- Positional and velocity vectors

The objective is to learn a policy:

π(aₜ | oₜ)

that predicts continuous control actions from combined sensory inputs.


---

## Methodology

### 1. Self-Supervised Representation Learning (VAE)

A Variational Autoencoder (VAE) was trained on RGB image observations to learn compact latent embeddings.

The objective function optimised:

- Reconstruction loss  
- KL divergence regularisation  

The learned latent representation was then used as visual input to the behaviour cloning model.

---

### 2. Behaviour Cloning Model

The downstream supervised model integrates:

- VAE latent embeddings  
- Proprioceptive robot state features  

The network predicts continuous control actions using Mean Squared Error (MSE) loss.

---

## Baseline Comparison

Performance was evaluated against:

- Fully supervised model trained directly on raw image features  
- State-only baseline  

Evaluation metrics included:

- Mean Squared Error (MSE)  
- Action prediction error distribution  

The VAE-based representation improved generalisation compared to raw image baselines.

---

## Training Pipeline

- Implemented in PyTorch  
- Mixed precision (FP16) training  
- Experiment tracking using Weights & Biases  
- Modular dataset and model structure  
- Configurable training parameters  

---

## Key Observations

- Latent dimensionality significantly influenced downstream performance.
- Self-supervised pretraining improved robustness to visual variation.
- KL weighting required careful tuning to avoid posterior collapse.
- Structured experiment tracking accelerated hyperparameter iteration.

---

## Technologies Used

- Python  
- PyTorch  
- OpenCV  
- Weights & Biases  
- Docker (for containerisation)

---

## Dataset

The dataset (~15GB) is not included due to size constraints.
This is from https://github.com/clvrai/clvr_jaco_play_dataset

The dataset was collected at a frequency of 10Hz. It has the following structure:

- Observations are split into 5 attributes,
    - **front_cam_ob** : observations from 3rd person cam
    - **mount_cam_ob** : observations from mounted camera
    - **ee_cartesian_pos_ob** : end effector cartesian position. ee_cartesian_pos_ob[0:3] corresponds to position and ee_cartesian_pos_ob[3:7] corresponds to orientation in quarternian format
    - **ee_cartesian_vel_ob** : end effector cartesian velocity. ee_cartesian_pos_ob[0:3] corresponds to change in position and ee_cartesian_pos_ob[3:6] corresponds to change in orientation in roll, pitch yaw format
    - **joint_pos_ob** : joint positions of the jaco arm (we only use the last 2 elements of this that correspond to the gripper joints)
- **actions** : first 3 elements are cartesian deltas and 4th element is a label from {0, 1, 2} meaning {open gripper, don't move gripper, close gripper}
- **terminals** : 1 at the end of each skill
- **prompts** : Natural language description of the goal
- **reward** : 1 at the end of each skill

To run training:

1. Update the dataset path in `train.py`
2. Ensure data follows the expected directory structure


---

## Future Improvements

- Temporal modelling (LSTM / Transformer-based policy)
- Contrastive representation learning
- Domain randomisation for improved robustness
- Deployment optimisation for real-time inference

---

## Contact

If you have any questions about this project, feel free to reach out.