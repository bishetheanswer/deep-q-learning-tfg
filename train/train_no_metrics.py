# Implementation based on :
# --> https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# --> https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter06/02_dqn_pong.py

import collections
import numpy as np
import torch
import torch.nn as nn
import argparse
import wrappers
import model
import time
import shutil

# Hyperparameters
N_STEPS = 3000000

DEFAULT_ENV_NAME = 'SonicTheHedgehog-Genesis'
GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 80000
REPLAY_START_SIZE = 5000
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 100000
TARGET_UPDATE_FREQUENCY = 2500

Transition = collections.namedtuple(
    'Transition', field_names=['state', 'action', 'reward', 'done',
                               'new_state'])


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), \
            np.array(rewards, dtype=np.float32), \
            np.array(dones, dtype=np.uint8), \
            np.array(next_states)


class Agent:
    def __init__(self, env, buffer):
        self.env = env
        self.buffer = buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.episode_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, epsilon, device):
        done_reward = None

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_t = torch.tensor(state_a).to(device)
            q_values = net(state_t)
            _, act_idx = torch.max(q_values, dim=1)
            action = int(act_idx.item())

        new_state, reward, done, _ = self.env.step(action)
        self.episode_reward += reward
        tran = Transition(self.state, action, reward, done, new_state)
        self.buffer.append(tran)
        self.state = new_state

        if done:
            done_reward = self.episode_reward
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, device):
    states, actions, rewards, dones, next_states = batch
    states_t = torch.tensor(np.array(states, copy=False)).to(device)
    actions_t = torch.tensor(actions).to(device)
    rewards_t = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)
    next_states_t = torch.tensor(np.array(next_states, copy=False)).to(device)

    state_action_values = net(states_t).gather(
        1, actions_t.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(next_states_t).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_t
    return nn.MSELoss()(state_action_values,
                        expected_state_action_values)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" +
                             DEFAULT_ENV_NAME)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Device:", device)
    print("Environment", args.env)

    # create environment
    env = wrappers.make_retro(args.env)
    
    # initizalize dqn and target
    net = model.DQN(env.observation_space.shape,
                        env.action_space.n).to(device)
    tgt_net = model.DQN(env.observation_space.shape,
                            env.action_space.n).to(device)

    # initialize replay buffer, agent, epsilon and optimizer
    buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPS_START
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    
    total_rewards = []
    step = 0
    ts_step = 0
    ts = time.time()
    
    while True:
        step += 1
        epsilon = max(EPS_END, EPS_START - step / EPS_DECAY)
        reward = agent.play_step(net, epsilon, device)
        
        if reward is not None:
            total_rewards.append(reward)
            speed = (step - ts_step) / (time.time() - ts)
            ts_step = step
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, reward %.3f, m_reward %.3f, "
                  "eps %.2f, speed %.2f steps per second" % (
                      step, len(total_rewards), reward, m_reward, epsilon,
                      speed
                  ))
            
        if step % 250000 == 0:
            torch.save(net.state_dict(), args.env +
                        "-steps_%d.dat" % (step))
            if step == N_STEPS:
                shutil.make_archive('results', 'zip', './')
                print("Training completed!")
                break
            
        if len(buffer) < REPLAY_START_SIZE:
            continue

        if step % TARGET_UPDATE_FREQUENCY == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss = calc_loss(batch, net, tgt_net, device)
        loss.backward()
        optimizer.step()
