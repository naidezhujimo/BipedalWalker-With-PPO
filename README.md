# Proximal Policy Optimization (PPO) for BipedalWalker-v3

This repository contains an implementation of the Proximal Policy Optimization (PPO) algorithm to solve the `BipedalWalker-v3` environment from the Gymnasium library. PPO is a popular reinforcement learning algorithm that balances ease of implementation with strong performance. This project uses a combination of policy and value networks to learn a policy for controlling a bipedal walker.

## Table of Contents
- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Code Structure](#code-structure)
- [How It Works](#how-it-works)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Introduction

Proximal Policy Optimization (PPO) is a policy gradient method that uses a clipped objective function to ensure stable and efficient training. This implementation applies PPO to the `BipedalWalker-v3` environment, where the goal is to train a bipedal robot to walk as far as possible without falling.

## Dependencies

To run this code, you need the following Python libraries:

- `numpy`
- `torch` (PyTorch)
- `gymnasium`
- `tensorboardX`
- `matplotlib`

You can install these dependencies using `pip`:

```bash
pip install numpy torch gymnasium tensorboardX matplotlib
```

## Code Structure

The code is structured as follows:

- **Policy Network (`A2C_policy` class)**: A neural network that outputs the mean action for a given state. It also includes a learnable parameter for the action standard deviation.
- **Value Network (`A2C_value` class)**: A neural network that estimates the value of a given state.
- **Environment Wrapper (`Env` class)**: Handles interactions with the Gymnasium environment, including data collection and advantage estimation using Generalized Advantage Estimation (GAE).
- **PPO Loss (`clipped_PPO_loss` function)**: Computes the PPO loss, which includes a clipped surrogate objective for the policy and a mean squared error loss for the value function.
- **Training Loop**: The main loop that iteratively collects data, computes advantages, and updates the policy and value networks using PPO.

## How It Works

1. **Policy and Value Networks**: The policy network outputs the mean action for a given state, while the value network estimates the state's value.
2. **Data Collection**: The agent interacts with the environment to collect trajectories of states, actions, rewards, and values.
3. **Advantage Estimation**: Generalized Advantage Estimation (GAE) is used to compute advantages, which measure how much better an action is compared to the average action at a given state.
4. **PPO Update**: The policy and value networks are updated using the PPO objective, which includes a clipped surrogate objective to ensure stable updates.
5. **Testing**: Periodically, the trained policy is tested in the environment, and its performance is recorded.

## Usage

To run the training script, simply execute the following command:

```bash
python ppo_bipedalwalker.py
```

### Key Hyperparameters

- `ENV_NAME`: The Gymnasium environment name (`BipedalWalker-v3`).
- `TRAJECTORY_SIZE`: The number of steps to collect before each update.
- `BATCH_SIZE`: The number of samples used for each update.
- `PPO_EPOCHS`: The number of epochs to update the networks for each batch of data.
- `CLIP_EPS`: The clipping parameter for the PPO objective.
- `GAE_LAMBDA`: The lambda parameter for GAE.
- `GAMMA`: The discount factor for future rewards.
- `POLICY_LR`: The learning rate for the policy network.
- `VALUE_LR`: The learning rate for the value network.

### Video Recording

The code includes functionality to record videos of the agent's performance during testing. Videos are saved in the `VIDEOS/TEST_VIDEOS_<experiment_name>` directory.

## Results

The training progress is visualized using TensorBoard and a plot of the training and testing rewards over time. The plot is saved as `reward_curve.png` in the `training_plots` directory.

### Example Output

```plaintext
##### 0 rew: -100 -200 0.0 0
##### 1 rew: -150 -200 0.0 0
...
##### 1199 rew: 250 -200 0.0 0
 > Testing.. 1200 280.5 500
=> Best test!! Reward:280.50  Steps:500
```

### Reward Convergence Plot

![Reward Convergence](training_plots/reward_curve.png)

