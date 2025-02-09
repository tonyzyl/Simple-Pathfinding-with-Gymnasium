"""
dqn_agent.py

modified based on: https://www.kaggle.com/code/auxeno/dqn-on-lunar-lander-rl?scriptVersionId=135585373

This module defines the neural network architectures (ResNet and PureResNet),
the replay buffer, the DQNAgent class, and the training function.
"""

import torch
from torch import nn
import random
import numpy as np
from collections import deque

# ------------------------------------------------------------------------------
# NN definitions
# ------------------------------------------------------------------------------
class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.alpha * x + self.beta    

class ResNetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, time_emb_dim, dropout=0,
                 skip_scale=1, adaptive_scale=True, affine=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.emb_dim = time_emb_dim
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.linear = nn.Linear(in_dim, out_dim)
        self.map_cond = nn.Linear(time_emb_dim, out_dim * (2 if adaptive_scale else 1))

        if affine:
            self.pre_norm = Affine(in_dim)
            self.post_norm = Affine(out_dim)
        else:
            self.pre_norm = nn.Identity()
            self.post_norm = nn.Identity()

    def forward(self, x, time_emb=None):
        orig = x
        params = nn.functional.silu(self.map_cond(time_emb).to(x.dtype))
        x = self.pre_norm(x)
        if self.adaptive_scale:
            scale, shift = params.chunk(2, dim=-1)
            x = nn.functional.silu(torch.addcmul(shift, x, scale + 1))
        else:
            x = nn.functional.silu(x.add_(params))
        x = self.linear(nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = self.post_norm(x)
        x = x.add_(orig)
        x = x * self.skip_scale
        return x

class ResNet(nn.Module):
    def __init__(self, in_dim, out_dim, uav_num_rays,  # note extra parameter here
                 model_dim=128,
                 dim_mult=[1, 1, 1],
                 num_blocks=2,
                 dropout=0.,
                 adaptive_scale=True,
                 skip_scale=1.0,
                 affine=True):
        super().__init__()
        block_kwargs = dict(dropout=dropout, skip_scale=skip_scale, adaptive_scale=adaptive_scale, affine=affine)

        # Use the passed parameter instead of a global constant
        self.uav_num_rays = uav_num_rays
        self.state_dim = in_dim - uav_num_rays
        self.obs_dim = uav_num_rays

        self.first_layer = nn.Linear(self.state_dim, model_dim)
        self.map_obs = nn.Linear(self.obs_dim, model_dim)
        self.blocks = nn.ModuleList()
        cout = model_dim
        self.ini_conv = nn.Conv1d(1, model_dim, kernel_size=4, stride=2, padding=1, padding_mode='circular')
        self.mid_conv = nn.Conv1d(model_dim, model_dim, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.out_conv = nn.ConvTranspose1d(model_dim, 1, kernel_size=4, stride=2, padding=1)
        for level, mult in enumerate(dim_mult):
            for _ in range(num_blocks):
                cin = cout
                cout = model_dim * mult
                self.blocks.append(ResNetBlock(cin, cout, model_dim, **block_kwargs))
        self.final_layer = nn.Linear(cout, out_dim)

    def forward(self, x):
        # Split the state and observation parts based on uav_num_rays
        obs = x[:, -self.uav_num_rays:]
        x = x[:, :-self.uav_num_rays]
        
        obs = self.ini_conv(obs.unsqueeze(1))
        obs = self.mid_conv(nn.functional.silu(obs))
        obs = self.out_conv(nn.functional.silu(obs)).squeeze(1)
        x = self.first_layer(x)
        obs = self.map_obs(obs)
        for block in self.blocks:
            x = block(x, obs)
        x = self.final_layer(nn.functional.silu(x))
        return x

class PureResNetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0,
                 skip_scale=1, adaptive_scale=True, affine=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.linear = nn.Linear(in_dim, out_dim)
        if affine:
            self.pre_norm = Affine(in_dim)
            self.post_norm = Affine(out_dim)
        else:
            self.pre_norm = nn.Identity()
            self.post_norm = nn.Identity()

    def forward(self, x):
        orig = x
        x = self.pre_norm(x)
        x = self.linear(nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = self.post_norm(x)
        x = x.add_(orig)
        x = x * self.skip_scale
        return x

class PureResNet(nn.Module):
    def __init__(self, in_dim, out_dim,
                 model_dim=128,
                 dim_mult=[1, 1, 1],
                 num_blocks=2,
                 dropout=0.,
                 adaptive_scale=True,
                 skip_scale=1.0,
                 affine=True):
        super().__init__()
        block_kwargs = dict(dropout=dropout, skip_scale=skip_scale, adaptive_scale=adaptive_scale, affine=affine)
        self.state_dim = in_dim
        self.first_layer = nn.Linear(self.state_dim, model_dim)
        self.blocks = nn.ModuleList()
        cout = model_dim
        for level, mult in enumerate(dim_mult):
            for _ in range(num_blocks):
                cin = cout
                cout = model_dim * mult
                self.blocks.append(PureResNetBlock(cin, cout, **block_kwargs))
        self.final_layer = nn.Linear(cout, out_dim)

    def forward(self, x):
        x = self.first_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_layer(x)
        x = nn.Softmax(dim=1)(x)
        return x

# ------------------------------------------------------------------------------
# Replay Buffer and DQN Agent
# ------------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, buffer_size=10000):
        self.buffer = deque(maxlen=buffer_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.stack(states), actions, rewards, np.stack(next_states), dones

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size=8, action_size=4,
                 learning_rate=1e-4, gamma=0.99, buffer_size=10000, batch_size=512,
                 type='conv', uav_num_rays=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.batch_size = batch_size
        self.action_size = action_size

        # For the conv-based network, the UAV config (e.g. number of rays) is needed.
        if type == 'conv':
            if uav_num_rays is None:
                raise ValueError("For conv type, uav_num_rays must be provided.")
            self.q_network = ResNet(state_size, action_size, uav_num_rays=uav_num_rays).to(self.device)
            self.target_network = ResNet(state_size, action_size, uav_num_rays=uav_num_rays).to(self.device)
        elif type == 'pure':
            self.q_network = PureResNet(state_size, action_size).to(self.device)
            self.target_network = PureResNet(state_size, action_size).to(self.device)
        else:
            raise NotImplementedError
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)

    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size:
            self.update_model()

    def act(self, state, eps=0.):
        if random.random() > eps:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.q_network.eval()
            with torch.no_grad():
                action_values = self.q_network(state)
            self.q_network.train()
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def update_model(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(np.array(dones).astype(np.uint8)).float().to(self.device)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        loss = torch.nn.MSELoss()(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, filepath):
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'agent_params': {
                'gamma': self.gamma,
                'batch_size': self.batch_size,
                'action_size': self.action_size
            }
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent_params = checkpoint.get('agent_params', {})
        self.gamma = agent_params.get('gamma', self.gamma)
        self.batch_size = agent_params.get('batch_size', self.batch_size)
        self.action_size = agent_params.get('action_size', self.action_size)

def train(agent, env, n_episodes=2000, n_max_step=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, target_update=10, print_every=10):
    mean_scores = []
    mean_steps = []
    mean_success_rate = []
    scores_window = deque(maxlen=100)
    steps_window = deque(maxlen=100)
    success_window = deque(maxlen=100)
    eps = eps_start

    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0 
        for n_step in range(n_max_step):
            action = agent.act(state, eps)
            next_state, reward, terminated, raw_reward = env.step(action)
            done = terminated
            agent.step(state, action, reward, next_state, done)
            state = next_state 
            score += reward
            if done:
                break 
        if n_step == n_max_step - 1:
            score += env.config.PEN_NOT_FINISHED
        if env.game_over:
            success_window.append(0)
        else:
            success_window.append(1)
        scores_window.append(score)
        steps_window.append(n_step)
        mean_scores.append(np.mean(score))
        mean_steps.append(np.mean(n_step))
        mean_success_rate.append(np.mean(success_window))
        eps = max(eps_end, eps_decay * eps)
        print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tSuccess rate: {np.mean(success_window):.2f}", end="")
        if i_episode % target_update == 0:
            agent.update_target_network()
        if i_episode % print_every == 0:
            frame = env.render()
            env.close()
            import matplotlib.pyplot as plt
            plt.imshow(frame)
            plt.show()
            print('\rEpisode {}\tAverage Score: {:.2f}, Eps: {:.3f}, Terminal vel: {:.3f}, angle:{:.2f}, dist2goal: {:.2f}, ang2goal: {:.2f}, step: {}, success rate: {:.2f}'
                  .format(i_episode, np.mean(scores_window), eps, 
                          np.linalg.norm(env.uav.linearVelocity), env.uav.angle, env.dist2goal, env.ang2goal, n_step, np.mean(success_window)))
    return mean_scores, mean_steps
