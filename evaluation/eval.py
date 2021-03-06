# Implementation based on:
# --> https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter06/03_dqn_play.py

import argparse
import numpy as np
import os
import torch
import wrappers_eval as wrappers
import model

N_EPISODES = 10
EPSILON = 0.05

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default='SonicTheHedgehog-Genesis',
                        help="Environment name to use, default=SonicTheHedgehog-Genesis")
    parser.add_argument("-r", "--record", default=False, action="store_true", help="Enable recording")
    args = parser.parse_args()

    if args.record:
        if not os.path.exists('./rec'):
            os.mkdir('./rec')

    # create environment
    env = wrappers.make_retro(args.env, args.record)

    # load agent
    dqn_net = model.DQN(env.observation_space.shape, env.action_space.n)
    state = torch.load(args.model, map_location=lambda stg, _: stg)
    dqn_net.load_state_dict(state)
    
    total_reward = 0.0
    
    for i in range(N_EPISODES):
        state = env.reset()  # initial state
        episode_reward = 0.0
        while True:
            env.render()
            
            # e-greedy policy
            if np.random.random() < EPSILON:
                action = env.action_space.sample()
            else:
                state_t = torch.tensor(np.array([state], copy=False))
                q_vals = dqn_net(state_t).data.numpy()[0]
                action = np.argmax(q_vals)
            
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            if done:
                break
        
        print('Reward in episode %d --> %.1f' % (i+1, episode_reward))
        total_reward += episode_reward     
    
    print('Average reward --> %.1f' % (total_reward/N_EPISODES))

