# Implementation based on:
# --> https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter06/02_dqn_pong.py
# --> https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import collections
import numpy as np
import argparse
import time
import os

import torch
import torch.nn as nn
import wrappers
import model

from keys import access_key, secret_access_key
import boto3


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
LEARNING_RATE = 1e-4


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
        """Samples a batch of experiences"""
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
        self.state = env.reset()  # initial state
        self.episode_reward = 0.0

    @torch.no_grad()
    def play_step(self, dqn_net, epsilon, device):
        done_reward = None

        # e-greedy policy
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_t = torch.tensor(state_a).to(device)
            q_values = dqn_net(state_t)
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


def calc_loss(batch, dqn_net, tgt_net, device):
    states, actions, rewards, dones, next_states = batch

    # Wraps data into PyTorch tensors
    states_t = torch.tensor(np.array(states, copy=False)).to(device)
    actions_t = torch.tensor(actions).to(device)
    rewards_t = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)
    next_states_t = torch.tensor(np.array(next_states, copy=False)).to(device)

    q_values = dqn_net(states_t).gather(
        1, actions_t.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        next_q_values = tgt_net(next_states_t).max(1)[0]
        next_q_values[done_mask] = 0.0  # Mask for terminal states
        next_q_values = next_q_values.detach()

    target_values = next_q_values * GAMMA + rewards_t
    return nn.MSELoss()(q_values, target_values)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" +
                             DEFAULT_ENV_NAME)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize AWS client
    bucket = 'tfg-miguel-bucket'
    client = boto3.client('s3',
                            aws_access_key_id = access_key,
                            aws_secret_access_key = secret_access_key)
    
    # Create environment
    env = wrappers.make_retro(args.env)
    
    # Initialize neural networks
    dqn_net = model.DQN(env.observation_space.shape,
                        env.action_space.n).to(device)
    tgt_net = model.DQN(env.observation_space.shape,
                            env.action_space.n).to(device)

    buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPS_START
    optimizer = torch.optim.Adam(dqn_net.parameters(), lr=LEARNING_RATE)
    
    total_rewards = []
    total_m_rewards = []
    total_steps = []
    total_duration = []
    step = 0
    ts_step = 0
    ts = time.time()
    print('Training started:')

    while True:
        step += 1
        epsilon = max(EPS_END, EPS_START - step / EPS_DECAY)
        reward = agent.play_step(dqn_net, epsilon, device)
        
        # When the episode is finished
        if reward is not None:
            # Append results and update variables
            total_rewards.append(reward)
            speed = (step - ts_step) / (time.time() - ts)
            total_duration.append(step - ts_step)
            ts_step = step
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])
            total_m_rewards.append(m_reward)
            total_steps.append(step)
            print("%d: done %d games, reward %.3f, m_reward %.3f, "
                  "eps %.2f, speed %.2f steps per second" % (
                      step, len(total_rewards), reward, m_reward, epsilon,
                      speed
                  ))

        # Save agent 
        if step % 250000 == 0:
            torch.save(dqn_net.state_dict(), args.env +
                        "-steps_%d.dat" % (step))
            
            # Upload agent to AWS S3
            for file in os.listdir():
                if str(step) in file:
                    upload_key = args.env + '/' + str(file)
                    client.upload_file(file, bucket, upload_key)
            
            # When training is finished exit the loop
            if step == N_STEPS:
                print("Training completed!")
                break

        # Skip gradient updates until the replay buffer has some transitions    
        if len(buffer) < REPLAY_START_SIZE:
            continue

        # Update the target network with the dqn weights
        if step % TARGET_UPDATE_FREQUENCY == 0:
            tgt_net.load_state_dict(dqn_net.state_dict())

        # Sample experiences, calculate the loss and perform optimization step
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss = calc_loss(batch, dqn_net, tgt_net, device)
        loss.backward()
        optimizer.step()
    
    # Save results
    np.savetxt('total_rewards.csv', total_rewards, delimiter=',')
    np.savetxt('total_m_rewards.csv', total_m_rewards, delimiter=',')
    np.savetxt('total_steps.csv', total_steps, delimiter=',')
    np.savetxt('total_duration.csv', total_duration, delimiter=',')
    
    # Upload results to AWS S3
    for file in os.listdir():
        if '.csv' in file:
            upload_key = args.env + '/' + str(file)
            client.upload_file(file, bucket, upload_key)

