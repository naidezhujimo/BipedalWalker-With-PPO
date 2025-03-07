import numpy as np
import gymnasium as gym
from tensorboardX import SummaryWriter
import os

import datetime
from collections import namedtuple
from collections import deque
import math
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from gymnasium.wrappers import RecordVideo

# Policy Network
class A2C_policy(nn.Module):
    def __init__(self, input_shape, n_actions):
        """
        - input_shape: The dimension of the input state.
        - n_actions: The dimension of the action space.
        """
        super(A2C_policy, self).__init__()
        # Two fully connected layers for feature extraction
        self.lp = nn.Sequential(
            nn.Linear(input_shape[0], 32),  # First fully connected layer, output dimension is 32
            nn.ReLU(),                      # ReLU activation function
            nn.Linear(32, 32),              # Second fully connected layer, input and output dimensions are both 32
            nn.ReLU())                      # ReLU activation function
        self.mean_l = nn.Linear(32, n_actions[0]) # Fully connected layer for outputting action means
        self.mean_l.weight.data.mul_(0.1)        # Initialize weights, multiply by 0.1
        self.var_l = nn.Linear(32, n_actions[0]) # Fully connected layer for outputting action variances
        self.var_l.weight.data.mul_(0.1)         # Initialize weights, multiply by 0.1
        self.logstd = nn.Parameter(torch.zeros(n_actions[0])) # Logarithmic form of action standard deviation, initialized to 0

    def forward(self, x):
        ot_n = self.lp(x.float())
        return F.tanh(self.mean_l(ot_n))

# Value Network
class A2C_value(nn.Module):
    def __init__(self, input_shape):
        super(A2C_value, self).__init__()
        self.lp = nn.Sequential(
            nn.Linear(input_shape[0], 32),  # First fully connected layer, output dimension is 32
            nn.ReLU(),                      # ReLU activation function
            nn.Linear(32, 32),              # Second fully connected layer, input and output dimensions are both 32
            nn.ReLU(),                      # ReLU activation function
            nn.Linear(32, 1))               # Fully connected layer for outputting state values

    def forward(self, x):
        return self.lp(x.float())

# Encapsulates the logic of environment interaction
class Env:
    game_rew = 0
    last_game_rew = 0
    game_n = 0
    last_games_rews = [-200]
    n_iter = 0

    def __init__(self, env_name, n_steps, gamma, gae_lambda, save_video=False):
        """
        env_name: Name of the environment
        n_steps: Number of steps for each sampling
        gamma: Discount factor
        gae_lambda: Parameter for Generalized Advantage Estimation (GAE)
        save_video: Whether to save videos
        """
        self.env = gym.make(env_name)
        self.obs, _ = self.env.reset() # Get the initial state
        self.n_steps = n_steps
        self.action_n = self.env.action_space.shape # Get the dimension of the action space
        self.observation_n = self.env.observation_space.shape[0] # Get the dimension of the state space
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    # Interact with the environment, sample data, and calculate rewards and advantage functions
    def steps(self, agent_policy, agent_value):
        memories = []  # Store sampled data
        for s in range(self.n_steps):
            self.n_iter += 1
            ag_mean = agent_policy(torch.tensor(self.obs))  # Get action means using the policy network
            logstd = agent_policy.logstd.data.cpu().numpy()  # Get action standard deviations
            # Generate actions by adding noise
            action = ag_mean.data.cpu().numpy() + np.exp(logstd) * np.random.normal(size=logstd.shape)
            action = np.clip(action, -1, 1)  # Clip actions to the range [-1, 1]
            state_value = float(agent_value(torch.tensor(self.obs)))  # Estimate the value of the current state using the value network
            new_obs, reward, terminated, truncated, _ = self.env.step(action)  # Interact with the environment to get new states and rewards
            done = terminated or truncated  # Check if the episode is done
            if done:
                # If done, set the reward to 0
                memories.append(Memory(obs=self.obs, action=action, new_obs=new_obs, reward=0, done=done, value=state_value, adv=0))
            else:
                # Store sampled data
                memories.append(Memory(obs=self.obs, action=action, new_obs=new_obs, reward=reward, done=done, value=state_value, adv=0))
            self.game_rew += reward  # Accumulate rewards
            self.obs = new_obs  # Update the state
            if done:
                print('#####',self.game_n, 'rew:', int(self.game_rew), int(np.mean(self.last_games_rews[-100:])), np.round(reward,2), self.n_iter)
                self.obs, _ = self.env.reset()  # Reset the environment
                self.last_game_rew = self.game_rew
                self.game_rew = 0
                self.game_n += 1
                self.n_iter = 0
                self.last_games_rews.append(self.last_game_rew) # Record rewards
        return self.generalized_advantage_estimation(memories) # Calculate the advantage function

    # Advantage function calculation
    def generalized_advantage_estimation(self, memories):
        upd_memories = []  # Sampled data containing states, actions, rewards, etc.
        run_add = 0  # Used for cumulative calculation of the advantage function
        for t in reversed(range(len(memories)-1)):
            if memories[t].done:
                run_add = memories[t].reward
            else:
                # Calculate the difference between the current reward and the future state value
                sigma = memories[t].reward + self.gamma * memories[t+1].value - memories[t].value
                # Recursively calculate the advantage function according to the GAE formula
                run_add = sigma + run_add * self.gamma * self.gae_lambda
            # Update the sampled data with the calculated advantage function
            upd_memories.append(Memory(obs=memories[t].obs, action=memories[t].action, new_obs=memories[t].new_obs, reward=run_add + memories[t].value, done=memories[t].done, value=memories[t].value, adv=run_add))
        return upd_memories[::-1]

# Calculate the logarithmic form of the policy's probability density function
def log_policy_prob(mean, std, actions):
    """
    - mean: The mean of the action, output by the policy network.
    - std: The standard deviation of the action (logarithmic form).
    - actions: The actual sampled actions.
    """
    """Calculate the logarithmic form of the Gaussian probability density function
        The first term is the exponential part of the Gaussian distribution
        The second term is the logarithm of the normalization constant
    """
    act_log_softmax = -((mean-actions)**2)/(2*torch.exp(std).clamp(min=1e-4)) - torch.log(torch.sqrt(2*math.pi*torch.exp(std)))
    return act_log_softmax

def compute_log_policy_prob(memories, nn_policy, device):
    n_mean = nn_policy(torch.tensor(np.array([m.obs for m in memories], dtype=np.float32)).to(device))
    n_mean = n_mean.type(torch.DoubleTensor)
    logstd = agent_policy.logstd.type(torch.DoubleTensor)
    actions = torch.DoubleTensor(np.array([m.action for m in memories])).to(device)
    return log_policy_prob(n_mean, logstd, actions)

# PPO loss function for updating the policy network
def clipped_PPO_loss(memories, nn_policy, nn_value, old_log_policy, adv, epsilon, writer, device):
    """
    - memories: Sampled data containing states, actions, rewards, etc.
    - nn_policy: Policy network
    - nn_value: Value network
    - old_log_policy: The logarithmic form of the old policy's probability density function
    - adv: Advantage function
    - epsilon: The clipping parameter in PPO, used to limit the magnitude of policy updates
    """
    rewards = torch.tensor(np.array([m.reward for m in memories], dtype=np.float32)).to(device)
    value = nn_value(torch.tensor(np.array([m.obs for m in memories], dtype=np.float32)).to(device))
    # Loss function for the value network, using mean squared error to calculate the difference between predicted values and true rewards
    vl_loss = F.mse_loss(value.squeeze(-1), rewards)
    # The logarithmic form of the new policy's probability density function
    new_log_policy = compute_log_policy_prob(memories, nn_policy, device)
    # The ratio of new and old policy probabilities
    rt_theta = torch.exp(new_log_policy - old_log_policy.detach())
    adv = adv.unsqueeze(-1)
    # Policy gradient loss function, using the clipping trick in PPO
    pg_loss = -torch.mean(torch.min(rt_theta*adv, torch.clamp(rt_theta, 1-epsilon, 1+epsilon)*adv))
    return pg_loss, vl_loss

# Evaluate the trained policy in the test environment
def test_game(tst_env, agent_policy, test_episodes):
    """
    - tst_env: Test environment
    - agent_policy: Trained policy network
    - test_episodes: Number of test episodes
    """
    reward_games = []
    steps_games = []
    for episode_id in range(test_episodes):
        obs, _ = tst_env.reset()
        rewards = 0
        steps = 0
        while True:
            ag_mean = agent_policy(torch.tensor(obs))
            action = np.clip(ag_mean.data.cpu().numpy().squeeze(), -1, 1)
            next_obs, reward, terminated, truncated, _ = tst_env.step(action)
            done = terminated or truncated
            steps += 1
            obs = next_obs
            rewards += reward
            if done:
                reward_games.append(rewards)
                steps_games.append(steps)
                print(f"Episode {episode_id} finished with reward {rewards}")
                break
    return np.mean(reward_games), np.mean(steps_games)


Memory = namedtuple('Memory', ['obs', 'action', 'new_obs', 'reward', 'done', 'value', 'adv'])

ENV_NAME = 'BipedalWalker-v3' # Name of the environment
PLOT_DIR = "training_plots" # Directory for plotting
MAX_ITER = 1200 # Maximum number of training iterations
BATCH_SIZE = 64 # Batch size for updating the network
PPO_EPOCHS = 7 # Number of times to update the network after each sampling
device = 'cpu' # Device
CLIP_GRADIENT = 0.2 # Clipping parameter for gradients
CLIP_EPS = 0.2 # Clipping parameter for PPO
TRAJECTORY_SIZE = 2049 # Number of steps for each sampling
GAE_LAMBDA = 0.95 # Parameter for GAE
GAMMA = 0.99 # Discount factor
test_episodes = 5 # Number of test episodes
best_test_result = -1e5 # Initialize the best test reward
save_video_test = True
N_ITER_TEST = 100 # Number of iterations between tests
POLICY_LR = 0.0004 # Learning rate for the policy network
VALUE_LR = 0.001 # Learning rate for the value network

if __name__ == '__main__':
    env = Env(ENV_NAME, TRAJECTORY_SIZE, GAMMA, GAE_LAMBDA)
    writer_name = 'PPO_'+ENV_NAME+'_'+datetime.datetime.now().strftime("%d_%H.%M.%S")+'_'+str(POLICY_LR)+'_'+str(VALUE_LR)+'_'+str(TRAJECTORY_SIZE)+'_'+str(BATCH_SIZE)
    writer = SummaryWriter(log_dir='content/runs/'+writer_name)

    # Ensure the video save path exists
    video_dir = "VIDEOS/TEST_VIDEOS_" + writer_name
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    # Ensure the checkpoints directory exists
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Ensure the plotting directory exists
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

    all_train_rewards = []  # Record the raw rewards for each episode
    all_test_rewards = []   # Record the average rewards for each test
    test_iterations = []    # Record the iterations when tests occur

    # Create the test environment and enable video recording
    test_env = gym.make(ENV_NAME, render_mode='rgb_array')
    test_env = RecordVideo(test_env, video_dir, episode_trigger=lambda episode_id: True)  # Save videos for each test

    # Initialize the policy and value networks
    agent_policy = A2C_policy(test_env.observation_space.shape, test_env.action_space.shape).to(device)
    agent_value = A2C_value(test_env.observation_space.shape).to(device)
    # Use Adam optimizer to initialize the optimizers for the policy and value networks
    optimizer_policy = optim.Adam(agent_policy.parameters(), lr=POLICY_LR)
    optimizer_value = optim.Adam(agent_value.parameters(), lr=VALUE_LR)

    experience = []
    n_iter = 0
    while env.game_n < MAX_ITER:
        n_iter += 1
        # Interact with the environment, sample a batch of data (batch), and calculate the advantage function
        batch = env.steps(agent_policy, agent_value)
        # Calculate the logarithmic form of the probability density function for the old policy
        old_log_policy = compute_log_policy_prob(batch, agent_policy, device)
        # Calculate and normalize the advantage function
        batch_adv = np.array([m.adv for m in batch])
        batch_adv = (batch_adv - np.mean(batch_adv)) / (np.std(batch_adv) + 1e-7)
        batch_adv = torch.tensor(batch_adv).to(device)

        pol_loss_acc = []
        val_loss_acc = []
        # Record the reward information during training to the log
        writer.add_scalar('rew', env.last_game_rew, n_iter)
        writer.add_scalar('10rew', np.mean(env.last_games_rews[-100:]), n_iter)
        all_train_rewards.append(env.last_game_rew) # Record the reward for each episode
        for s in range(PPO_EPOCHS):
            # Divide the sampled data into mini-batches and update the network batch by batch
            for mb in range(0, len(batch), BATCH_SIZE):
                mini_batch = batch[mb:mb+BATCH_SIZE]
                minib_old_log_policy = old_log_policy[mb:mb+BATCH_SIZE]
                minib_adv = batch_adv[mb:mb+BATCH_SIZE]
                pol_loss, val_loss = clipped_PPO_loss(mini_batch, agent_policy, agent_value, minib_old_log_policy, minib_adv, CLIP_EPS, writer, device)
                optimizer_policy.zero_grad()
                pol_loss.backward()
                optimizer_policy.step()
                optimizer_value.zero_grad()
                val_loss.backward()
                optimizer_value.step()
                pol_loss_acc.append(float(pol_loss))
                val_loss_acc.append(float(val_loss))

        writer.add_scalar('pg_loss', np.mean(pol_loss_acc), n_iter)
        writer.add_scalar('vl_loss', np.mean(val_loss_acc), n_iter)
        writer.add_scalar('rew', env.last_game_rew, n_iter)
        writer.add_scalar('10rew', np.mean(env.last_games_rews[-100:]), n_iter)

        if env.game_n % N_ITER_TEST == 0:
            test_rews, test_stps = test_game(test_env, agent_policy, test_episodes)
            print(' > Testing..', n_iter, test_rews, test_stps)
            if test_rews > best_test_result:
                torch.save({
                    'agent_policy': agent_policy.state_dict(),
                    'agent_value': agent_value.state_dict(),
                    'optimizer_policy': optimizer_policy.state_dict(),
                    'optimizer_value': optimizer_value.state_dict(),
                    'test_reward': test_rews
                }, os.path.join(checkpoint_dir, 'checkpoint_'+writer_name+'.pth.tar'))
                best_test_result = test_rews
                print('=> Best test!! Reward:{:.2f}  Steps:{}'.format(test_rews, test_stps))
            writer.add_scalar('test_rew', test_rews, n_iter)
    # Plot the reward curve
    def plot_rewards(train_rewards, test_rewards, test_iters, filename):
        plt.figure(figsize=(12, 6))
        plt.plot(train_rewards, label='Training Reward', alpha=0.6, linewidth=1)
        plt.scatter(test_iters, test_rewards, color='red', s=20, label='Test Reward')
        plt.title("Training and Testing Reward Curve")
        plt.xlabel("Training Iterations")
        plt.ylabel("Reward")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.savefig(os.path.join(PLOT_DIR, filename), bbox_inches='tight')
        plt.close()
    
    plot_rewards(all_train_rewards, all_test_rewards, test_iterations, 'reward_curve.png')


    # Close the test environment and video recording
    test_env.close()
    writer.close()