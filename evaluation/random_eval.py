# Implementation based on:
# --> https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter06/03_dqn_play.py

import argparse
import os
import wrappers_eval as wrappers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", default='SonicTheHedgehog-Genesis',
                        help="Environment name to use, default=SonicTheHedgehog-Genesis")
    parser.add_argument("-r", "--record", default=False, action="store_true", help="Enable recording")
    args = parser.parse_args()

    if args.record:
        if not os.path.exists('./rec'):
            os.mkdir('./rec')

    # create environment
    env = wrappers.make_retro(args.env, args.record)
    
    total_reward = 0.0
    
    for i in range(10):
        state = env.reset()
        episode_reward = 0.0
        while True:
            env.render()

            # Random policy
            action = env.action_space.sample()

            state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            if done:
                break
        
        print('Reward in episode %d --> %.1f' % (i+1, episode_reward))
        total_reward += episode_reward     
    
    print('Average reward --> %.1f' % (total_reward/10))

