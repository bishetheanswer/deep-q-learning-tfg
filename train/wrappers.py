import gym
from gym import spaces
import retro
import numpy as np
import cv2
from collections import deque


class ImageToPyTorch(gym.ObservationWrapper):
    """
    Transforms the image format from HWC (Height, Width, Channel)
    to PyTorch format CHW (Channel, Height, Width)

    Implementation: https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter06/lib/wrappers.py
    """

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=new_shape, dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    """
    Converts bytes to float in range [0.0,1.0]
    
    Implementation: https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
    """

    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class WarpFrame(gym.ObservationWrapper):
    """
    Warps frames to 84x84 as done in the Nature paper and later work.

    Implementation: https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
    """

    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(
            original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    """
    Returns only every `skip`-th frame

    Implementation: https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
    """

    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros(
            (2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class FrameStack(gym.Wrapper):
    """
    Stacks k last frames.

    Implementation: https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
    """

    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(
            shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class LazyFrames(object):
    """
    This object ensures that common frames between the observations are only stored once.
    It exists purely to optimize memory usage.

    Implementation: https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
    """

    def __init__(self, frames):
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]


class Discretizer(gym.ActionWrapper):
    """
    Wraps a gym environment and make it use discrete actions.
        
    Implementation: https://github.com/openai/retro/blob/master/retro/examples/discretizer.py
    """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(
            len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()


class DiscretizerV1(Discretizer):
    """
    - Sonic The Hedgehog
    """

    def __init__(self, env):
        super().__init__(env=env, combos=[[], ['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'], ['DOWN', 'B'], ['B']])


class DiscretizerV2(Discretizer):
    """
    - Streets of Rage 2
    """

    def __init__(self, env):
        super().__init__(env=env, combos=[[], ['UP'], ['DOWN'], ['LEFT'], ['RIGHT'], ['UP', 'LEFT'], ['UP', 'RIGHT'], ['DOWN', 'LEFT'], ['DOWN', 'RIGHT'], ['B'], ['LEFT', 'B'], ['RIGHT','B'], ['B', 'C'], ['LEFT', 'B', 'C'], ['RIGHT', 'B', 'C'], ['C', 'DOWN', 'B'], ['A'], ['B', 'A'], ['UP', 'C'], ['C']])


class DiscretizerV3(Discretizer):
    """
    - Columns
    """

    def __init__(self, env):
        super().__init__(env=env, combos=[[], ['LEFT'], ['DOWN'], ['RIGHT'], ['A']])


class DiscretizerV4(Discretizer):
    """
    - Flicky
    """

    def __init__(self, env):
        super().__init__(env=env, combos=[[], ['LEFT'], ['RIGHT'], ['A'], ['LEFT', 'A'], ['RIGHT', 'A']])


class DiscretizerV5(Discretizer):
    """
    - Bio-Hazard Battle
    """

    def __init__(self, env):
        super().__init__(env=env, combos=[[], ['UP'], ['DOWN'], ['LEFT'], ['RIGHT'], ['A'], ['B'], ['UP', 'A'], ['DOWN', 'A'], ['LEFT', 'A'], ['RIGHT', 'A'],
        ['UP', 'B'], ['DOWN', 'B'], ['LEFT', 'B'], ['RIGHT', 'B']])


class LifesWrapperV1(gym.Wrapper):
    """
    Makes end-of-life == end-of-episode for games where lives == 2
    - Streets of Rage 2
    """

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.lives = info['lives']
        if self.lives == 1:
            done = True
        return obs, reward, done, info


class LifesWrapperV2(gym.Wrapper):
    """
    Makes end-of-life == end-of-episode for games where lives == 3
    - Sonic The Hedgehog
    - Flicky
    - Bio-Hazard Battle
    """

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.lives = info['lives']
        if self.lives == 2:
            done = True
        return obs, reward, done, info


class ClipRewardEnv(gym.RewardWrapper):
    """
    Bins reward to {+1, 0, -1} by its sign

    Implementation: https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
    """

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        return np.sign(reward)


def make_retro(env_name, scen="scenario"):
    
    two_lives = ["StreetsOfRage2-Genesis"]
    three_lives = ["SonicTheHedgehog-Genesis", "Flicky-Genesis", "BioHazardBattle-Genesis"]

    # Create environment
    env = retro.make(env_name, scenario=scen)

    # Apply wrappers
    env = MaxAndSkipEnv(env, skip=5)

    if env_name in three_lives:
        env = LifesWrapperV2(env)
    elif env_name in two_lives:
        env = LifesWrapperV1(env)
    
    if env_name == "SonicTheHedgehog-Genesis":
        env = DiscretizerV1(env)
    elif env_name == "StreetsOfRage2-Genesis":
        env = DiscretizerV2(env)
    elif env_name == "Columns-Genesis":
        env = DiscretizerV3(env)
    elif env_name == "Flicky-Genesis":
        env = DiscretizerV4(env)
    elif env_name == "BioHazardBattle-Genesis":
        env = DiscretizerV5(env)

    env = WarpFrame(env)
    env = ImageToPyTorch(env)
    env = FrameStack(env, 4)
    env = ClipRewardEnv(env)
    return ScaledFloatFrame(env)

