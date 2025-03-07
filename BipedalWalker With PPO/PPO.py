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

# 策略网络
class A2C_policy(nn.Module):
    def __init__(self, input_shape, n_actions):
        """
        - input_shape: 输入状态的维度
        - n_actions: 动作空间的维度
        """
        super(A2C_policy, self).__init__()
        # 两层全连接网络，用于提取特征
        self.lp = nn.Sequential(
            nn.Linear(input_shape[0], 32),  # 第一层全连接层，输入维度为状态空间维度，输出维度为32
            nn.ReLU(),                      # ReLU激活函数
            nn.Linear(32, 32),              # 第二层全连接层，输入输出维度均为32
            nn.ReLU())                      # ReLU激活函数
        self.mean_l = nn.Linear(32, n_actions[0]) # 输出动作均值的全连接层
        self.mean_l.weight.data.mul_(0.1)        # 初始化权重，乘以0.1
        self.var_l = nn.Linear(32, n_actions[0]) # 输出动作方差的全连接层
        self.var_l.weight.data.mul_(0.1)         # 初始化权重，乘以0.1
        self.logstd = nn.Parameter(torch.zeros(n_actions[0])) # 动作的标准差(对数形式)，初始化为0

    def forward(self, x):
        ot_n = self.lp(x.float())
        return F.tanh(self.mean_l(ot_n))

# 价值网络
class A2C_value(nn.Module):
    def __init__(self, input_shape):
        super(A2C_value, self).__init__()
        self.lp = nn.Sequential(
            nn.Linear(input_shape[0], 32),  # 第一层全连接层，输入维度为状态空间维度，输出维度为32
            nn.ReLU(),                      # ReLU激活函数
            nn.Linear(32, 32),              # 第二层全连接层，输入输出维度均为32
            nn.ReLU(),                      # ReLU激活函数
            nn.Linear(32, 1))               # 输出状态价值的全连接层

    def forward(self, x):
        return self.lp(x.float())

# 封装了环境交互的逻辑
class Env:
    game_rew = 0
    last_game_rew = 0
    game_n = 0
    last_games_rews = [-200]
    n_iter = 0

    def __init__(self, env_name, n_steps, gamma, gae_lambda, save_video=False):
        """
        env_name: 环境名称
        n_steps: 每次采样步数
        gamma: 折扣因子
        gae_lambda: GAE(Generalized Advantage Estimation)的参数
        save_video: 是否保存视频
        """
        self.env = gym.make(env_name)
        self.obs, _ = self.env.reset() # 获取初始状态
        self.n_steps = n_steps
        self.action_n = self.env.action_space.shape # 获取动作空间的维度
        self.observation_n = self.env.observation_space.shape[0] # 获取状态空间的维度
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    # 与环境进行交互，采样数据，并计算奖励和优势函数
    def steps(self, agent_policy, agent_value):
        memories = []  # 存储采样数据
        for s in range(self.n_steps):
            self.n_iter += 1
            ag_mean = agent_policy(torch.tensor(self.obs))  # 使用策略网络获取动作均值
            logstd = agent_policy.logstd.data.cpu().numpy()  # 获取动作的标准差
            # 添加噪声生成动作
            action = ag_mean.data.cpu().numpy() + np.exp(logstd) * np.random.normal(size=logstd.shape)
            action = np.clip(action, -1, 1)  # 将动作限制在[-1, 1]范围内
            state_value = float(agent_value(torch.tensor(self.obs)))  # 使用价值网络估计当前状态的价值
            new_obs, reward, terminated, truncated, _ = self.env.step(action)  # 与环境交互，获取新状态和奖励
            done = terminated or truncated  # 判断是否结束
            if done:
                # 如果结束，奖励设为0
                memories.append(Memory(obs=self.obs, action=action, new_obs=new_obs, reward=0, done=done, value=state_value, adv=0))
            else:
                # 存储采样数据
                memories.append(Memory(obs=self.obs, action=action, new_obs=new_obs, reward=reward, done=done, value=state_value, adv=0))
            self.game_rew += reward  # 累加奖励
            self.obs = new_obs  # 更新状态
            if done:
                print('#####',self.game_n, 'rew:', int(self.game_rew), int(np.mean(self.last_games_rews[-100:])), np.round(reward,2), self.n_iter)
                self.obs, _ = self.env.reset()  # 重置环境
                self.last_game_rew = self.game_rew
                self.game_rew = 0
                self.game_n += 1
                self.n_iter = 0
                self.last_games_rews.append(self.last_game_rew) # 记录奖励
        return self.generalized_advantage_estimation(memories) # 计算优势函数

    # 优势函数计算
    def generalized_advantage_estimation(self, memories):
        upd_memories = []  # 采样数据，包含状态、动作、奖励等信息
        run_add = 0  # 用于累积计算优势函数
        for t in reversed(range(len(memories)-1)):
            if memories[t].done:
                run_add = memories[t].reward
            else:
                # 计算当前奖励与未来状态价值的差值
                sigma = memories[t].reward + self.gamma * memories[t+1].value - memories[t].value
                # 根据GAE公式，递归计算优势函数
                run_add = sigma + run_add * self.gamma * self.gae_lambda
            # 更新后的采样数据，包含计算好的优势函数
            upd_memories.append(Memory(obs=memories[t].obs, action=memories[t].action, new_obs=memories[t].new_obs, reward=run_add + memories[t].value, done=memories[t].done, value=memories[t].value, adv=run_add))
        return upd_memories[::-1]

# 计算策略的概率密度函数的对数形式
def log_policy_prob(mean, std, actions):
    """
    - mean: 动作的均值，由策略网络输出
    - std: 动作的标准差(对数形式)
    - actions: 实际采样的动作
    """
    """计算高斯分布的概率密度函数的对数形式
        第一项是高斯分布的指数部分
        第二项是归一化常数的对数
    """
    act_log_softmax = -((mean-actions)**2)/(2*torch.exp(std).clamp(min=1e-4)) - torch.log(torch.sqrt(2*math.pi*torch.exp(std)))
    return act_log_softmax

def compute_log_policy_prob(memories, nn_policy, device):
    n_mean = nn_policy(torch.tensor(np.array([m.obs for m in memories], dtype=np.float32)).to(device))
    n_mean = n_mean.type(torch.DoubleTensor)
    logstd = agent_policy.logstd.type(torch.DoubleTensor)
    actions = torch.DoubleTensor(np.array([m.action for m in memories])).to(device)
    return log_policy_prob(n_mean, logstd, actions)

# PPO损失函数用于更新策略网络
def clipped_PPO_loss(memories, nn_policy, nn_value, old_log_policy, adv, epsilon, writer, device):
    """
    - memories: 采样数据，包含状态、动作、奖励等信息
    - nn_policy: 策略网络
    - nn_value: 价值网络
    - old_log_policy: 旧策略的概率密度函数的对数形式
    - adv: 优势函数
    - epsilon: PPO中的裁剪参数, 用于限制策略更新的幅度
    """
    rewards = torch.tensor(np.array([m.reward for m in memories], dtype=np.float32)).to(device)
    value = nn_value(torch.tensor(np.array([m.obs for m in memories], dtype=np.float32)).to(device))
    # 价值网络的损失函数，使用均方误差计算预测值与真实奖励的差异
    vl_loss = F.mse_loss(value.squeeze(-1), rewards)
    # 新策略的概率密度函数的对数形式
    new_log_policy = compute_log_policy_prob(memories, nn_policy, device)
    # 新旧策略的概率比值
    rt_theta = torch.exp(new_log_policy - old_log_policy.detach())
    adv = adv.unsqueeze(-1)
    # 策略梯度损失函数，使用PPO的裁剪技巧
    pg_loss = -torch.mean(torch.min(rt_theta*adv, torch.clamp(rt_theta, 1-epsilon, 1+epsilon)*adv))
    return pg_loss, vl_loss

# 用于在测试环境中评估训练好的策略
def test_game(tst_env, agent_policy, test_episodes):
    """
    - tst_env: 测试环境
    - agent_policy: 训练好的策略网络
    - test_episodes: 测试的轮数
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

ENV_NAME = 'BipedalWalker-v3' # 环境名称
PLOT_DIR = "training_plots" # 绘图名称
MAX_ITER = 1200 # 训练的最大迭代次数
BATCH_SIZE = 64 # 每次更新网络时使用的批量大小
PPO_EPOCHS = 7 # 每次采样后对网络进行更新的次数
device = 'cpu' # 设备
CLIP_GRADIENT = 0.2 # PPO算法中的裁剪参数
CLIP_EPS = 0.2 # 梯度裁剪的阈值
TRAJECTORY_SIZE = 2049 # 每次采样时与环境交互的步数
GAE_LAMBDA = 0.95 # GAE算法中的参数
GAMMA = 0.99 # 折扣因子
test_episodes = 5 # 每次测试时运行的episode数量
best_test_result = -1e5 # 初始化最佳测试奖励
save_video_test = True
N_ITER_TEST = 100 # 每隔多少次迭代进行一次测试
POLICY_LR = 0.0004 # 策略网络的学习率
VALUE_LR = 0.001 # 价值网络的学习率

if __name__ == '__main__':
    env = Env(ENV_NAME, TRAJECTORY_SIZE, GAMMA, GAE_LAMBDA)
    writer_name = 'PPO_'+ENV_NAME+'_'+datetime.datetime.now().strftime("%d_%H.%M.%S")+'_'+str(POLICY_LR)+'_'+str(VALUE_LR)+'_'+str(TRAJECTORY_SIZE)+'_'+str(BATCH_SIZE)
    writer = SummaryWriter(log_dir='content/runs/'+writer_name)

    # 确保视频保存路径存在
    video_dir = "VIDEOS/TEST_VIDEOS_" + writer_name
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    # 确保 checkpoints 目录存在
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # 确保绘图目录存在
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

    all_train_rewards = []  # 记录每个episode的原始奖励
    all_test_rewards = []   # 记录每次测试的平均奖励
    test_iterations = []    # 记录测试发生的迭代次数

    # 创建测试环境并启用视频录制
    test_env = gym.make(ENV_NAME, render_mode='rgb_array')
    test_env = RecordVideo(test_env, video_dir, episode_trigger=lambda episode_id: True)  # 每次测试都保存视频

    # 初始化策略网络和价值网络
    agent_policy = A2C_policy(test_env.observation_space.shape, test_env.action_space.shape).to(device)
    agent_value = A2C_value(test_env.observation_space.shape).to(device)
    # 使用Adam优化器分别初始化策略网络和价值网络的优化器
    optimizer_policy = optim.Adam(agent_policy.parameters(), lr=POLICY_LR)
    optimizer_value = optim.Adam(agent_value.parameters(), lr=VALUE_LR)

    experience = []
    n_iter = 0
    while env.game_n < MAX_ITER:
        n_iter += 1
        # 与环境进行交互，采样一批数据（batch），并计算优势函数
        batch = env.steps(agent_policy, agent_value)
        # 计算旧策略下采样动作的概率对数
        old_log_policy = compute_log_policy_prob(batch, agent_policy, device)
        # 计算并标准化优势函数
        batch_adv = np.array([m.adv for m in batch])
        batch_adv = (batch_adv - np.mean(batch_adv)) / (np.std(batch_adv) + 1e-7)
        batch_adv = torch.tensor(batch_adv).to(device)

        pol_loss_acc = []
        val_loss_acc = []
        # 记录训练过程中的奖励信息到日志
        writer.add_scalar('rew', env.last_game_rew, n_iter)
        writer.add_scalar('10rew', np.mean(env.last_games_rews[-100:]), n_iter)
        all_train_rewards.append(env.last_game_rew) # 记录每个episode的奖励
        for s in range(PPO_EPOCHS):
            # 将采样数据划分为小批次，逐批次更新网络
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
    # 绘制奖励曲线
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


    # 关闭测试环境和视频录制
    test_env.close()
    writer.close()