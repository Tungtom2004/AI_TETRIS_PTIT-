import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class TetrisNet(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(TetrisNet, self).__init__()

        # Separate convolution paths for current player and opponent grids
        self.conv_current = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Calculate the size after convolutions
        conv_out_size = self._get_conv_out(input_shape)

        self.advantage = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv_current(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv_current(x).view(x.size()[0], -1)
        advantage = self.advantage(conv_out)
        value = self.value(conv_out)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class TetrisAgent:
    def __init__(self, env, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.env = env
        self.device = device

        # Handle grid observation type
        if env.game_interface._obs_type == "grid":
            self.state_shape = (1, *env.observation_space.shape[:-1])  # Remove last dimension of shape
        else:  # image type
            self.state_shape = env.observation_space.shape

        self.n_actions = env.action_space.n

        # Initialize networks with Dueling DQN architecture
        self.policy_net = TetrisNet(self.state_shape, self.n_actions).to(device)
        self.target_net = TetrisNet(self.state_shape, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Hyperparameters tuned for Tetris
        self.batch_size = 128
        self.gamma = 0.99
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 0.9995
        self.target_update = 1000
        self.learning_rate = 0.0001
        self.double_dqn = True  # Use Double DQN

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = deque(maxlen=100000)  # Increased replay buffer size
        self.steps_done = 0
        self.epsilon = self.eps_start

        # Tetris-specific reward shaping
        self.reward_weights = {
            'lines_cleared': 1.0,
            'height_penalty': -0.5,
            'holes_penalty': -1.0,
            'bumpiness_penalty': -0.3
        }

    def shape_reward(self, reward, info):
        """Shape the reward based on game state information"""
        shaped_reward = reward

        if info:
            # Add penalties/rewards based on board state
            shaped_reward += (
                    self.reward_weights['height_penalty'] * info.get('max_height', 0) +
                    self.reward_weights['holes_penalty'] * info.get('holes', 0) +
                    self.reward_weights['bumpiness_penalty'] * info.get('diff_sum', 0)
            )

        return shaped_reward

    def process_state(self, state):
        """Process state for network input"""
        if isinstance(state, np.ndarray):
            if state.dtype != np.float32:
                state = state.astype(np.float32)

            # Remove last dimension if it's 1
            if state.shape[-1] == 1:
                state = state.squeeze(-1)

            # Add channel dimension if needed
            if len(state.shape) == 2:
                state = state[np.newaxis, :]

        return state

    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = self.process_state(state)
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                return q_values.max(1)[1].item()
        else:
            return self.env.random_action()

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = random.sample(self.memory, self.batch_size)
        batch = list(zip(*transitions))

        state_batch = torch.FloatTensor(np.array([self.process_state(s) for s in batch[0]])).to(self.device)
        action_batch = torch.LongTensor(batch[1]).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).to(self.device)
        next_state_batch = torch.FloatTensor(np.array([self.process_state(s) for s in batch[3]])).to(self.device)
        done_batch = torch.FloatTensor(batch[4]).to(self.device)

        # Double DQN
        if self.double_dqn:
            next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            next_state_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
        else:
            next_state_values = self.target_net(next_state_batch).max(1)[0]

        expected_state_action_values = reward_batch + (1 - done_batch) * self.gamma * next_state_values

        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # Huber loss for stability
        loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        return loss.item()

    def train(self, num_episodes, render=False):
        rewards_history = []

        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, done, info = self.env.step(action)

                # Shape the reward
                shaped_reward = self.shape_reward(reward, info)

                # Store transition
                self.memory.append((state, action, shaped_reward, next_state, done))

                state = next_state
                episode_reward += reward

                # Optimize model
                loss = self.optimize_model()

                # Update target network
                if self.steps_done % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                self.steps_done += 1

                # Decay epsilon
                self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

                if render:
                    self.env.render()

            rewards_history.append(episode_reward)

            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(rewards_history[-10:])
                print(
                    f"Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")

        return rewards_history

    def save(self, path):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']