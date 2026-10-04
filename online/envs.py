import gym
import cv2
import numpy as np
from abc import abstractmethod
from collections import deque
from copy import copy
from torch.multiprocessing import Pipe, Process

from model import *
from config import *
from PIL import Image

default_config, config_path = load_config()
train_method = default_config['TrainMethod']
max_step_per_episode = int(default_config['MaxStepPerEpisode'])


class Environment(Process):
    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def pre_proc(self, x):
        pass

    @abstractmethod
    def get_init_state(self, x):
        pass


def unwrap(env):
    if hasattr(env, "unwrapped"):
        return env.unwrapped
    elif hasattr(env, "env"):
        return unwrap(env.env)
    elif hasattr(env, "leg_env"):
        return unwrap(env.leg_env)
    else:
        return env


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, is_render, skip=4):
        gym.Wrapper.__init__(self, env)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip
        self.is_render = is_render

    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if self.is_render:
                self.env.render()
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break

        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class MontezumaInfoWrapper(gym.Wrapper):
    def __init__(self, env, room_address):
        super(MontezumaInfoWrapper, self).__init__(env)
        self.room_address = room_address
        self.visited_rooms = set()

    def get_current_room(self):
        ram = unwrap(self.env).ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.visited_rooms.add(self.get_current_room())
        if 'episode' not in info:
            info['episode'] = {}
        info['episode'].update(visited_rooms=copy(self.visited_rooms))

        if done:
            self.visited_rooms.clear()
        return obs, rew, done, info

    def reset(self):
        return self.env.reset()

class GravitarInfoWrapper(gym.Wrapper):
    def __init__(self, env, zone_address):
        super(GravitarInfoWrapper, self).__init__(env)
        self.zone_address = zone_address
        self.visited_zones = set()

    def get_current_zone(self):
        ram = unwrap(self.env).ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.zone_address])

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        self.visited_zones.add(self.get_current_zone())
        if 'episode' not in info:
            info['episode'] = {}
        info['episode'].update(visited_zones=copy(self.visited_zones))

        if done:
            self.visited_zones.clear()
        return obs, rew, done, info

    def reset(self):
        return self.env.reset()

class VentureInfoWrapper(gym.Wrapper):
    def __init__(self, env, chamber_address):
        super(VentureInfoWrapper, self).__init__(env)
        self.chamber_address = chamber_address
        self.visited_chambers = set()
    def get_current_chamber(self):
        ram = unwrap(self.env).ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.chamber_address])
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.visited_chambers.add(self.get_current_chamber())
        if 'episode' not in info:
            info['episode'] = {}
        info['episode'].update(visited_chambers=copy(self.visited_chambers))
        if done:
            self.visited_chambers.clear()
        return obs, rew, done, info
    def reset(self):
        return self.env.reset()

class FreewayInfoWrapper(gym.Wrapper):
    def __init__(self, env, y_address=None, num_bins=32):
        super(FreewayInfoWrapper, self).__init__(env)
        self.y_address = y_address
        self.num_bins = num_bins

        self.max_y = 0
        self.visited_y_bins = set()

    def _get_ram(self):
        ram = unwrap(self.env).ale.getRAM()
        assert len(ram) == 128
        return ram

    def get_chicken_y(self):
        ram = self._get_ram()
        if self.y_address is None:
            return None
        return int(ram[self.y_address])

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        y = self.get_chicken_y()
        if y is not None:
            self.max_y = max(self.max_y, y)
            y_bin = int(y / (256 / self.num_bins))
            self.visited_y_bins.add(y_bin)

        if 'episode' not in info:
            info['episode'] = {}
        info['episode'].update(max_y=self.max_y)
        info['episode'].update(visited_y_bins=copy(self.visited_y_bins))

        if done:
            self.max_y = 0
            self.visited_y_bins.clear()

        return obs, rew, done, info

    def reset(self):
        return self.env.reset()
        
class AtariEnvironment(Environment):
    def __init__(
            self,
            env_id,
            is_render,
            env_idx,
            child_conn,
            history_size=4,
            h=84,
            w=84,
            life_done=True,
            sticky_action=True,
            p=0.25):
        super(AtariEnvironment, self).__init__()
        self.daemon = True
        self.env = MaxAndSkipEnv(gym.make(env_id), is_render)
        if 'Montezuma' in env_id:
            self.env = MontezumaInfoWrapper(self.env, room_address=3 if 'Montezuma' in env_id else 1)
        elif 'Gravitar' in env_id:
            self.env = GravitarInfoWrapper(self.env, zone_address=5)
        elif 'Venture' in env_id:
            self.env = VentureInfoWrapper(self.env, chamber_address=3)
        elif 'Freeway' in env_id:
            self.env = FreewayInfoWrapper(self.env, y_address=None, num_bins=32)

        self.env_id = env_id
        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.child_conn = child_conn

        self.sticky_action = sticky_action
        self.last_action = 0
        self.p = p

        self.history_size = history_size
        self.history = np.zeros([history_size, h, w])
        self.h = h
        self.w = w

        self.reset()

    def run(self):
        super(AtariEnvironment, self).run()
        while True:
            action = self.child_conn.recv()

            if 'Breakout' in self.env_id:
                action += 1

            if self.sticky_action:
                if np.random.rand() <= self.p:
                    action = self.last_action
                self.last_action = action

            s, reward, done, info = self.env.step(action)

            if max_step_per_episode < self.steps:
                done = True

            log_reward = reward
            force_done = done

            self.history[:3, :, :] = self.history[1:, :, :]
            self.history[3, :, :] = self.pre_proc(s)

            self.rall += reward
            self.steps += 1

            if done:
                self.recent_rlist.append(self.rall)
                print("[Episode {}({})] Step: {}  Reward: {}  Recent Reward: {}  Visited Room: [{}]".format(
                    self.episode, self.env_idx, self.steps, self.rall, np.mean(self.recent_rlist),
                    info.get('episode', {}).get('visited_rooms', {})))
                self.history = self.reset()

            self.child_conn.send(
                [self.history[:, :, :], reward, force_done, done, log_reward])

    def reset(self):
        self.last_action = 0
        self.steps = 0
        self.episode += 1
        self.rall = 0
        s = self.env.reset()
        self.get_init_state(
            self.pre_proc(s))
        return self.history[:, :, :]

    def pre_proc(self, X):
        if isinstance(X, tuple):
            X = X[0]
        X = np.array(Image.fromarray(X).convert('L')).astype('float32')
        x = cv2.resize(X, (self.h, self.w))
        return x

    def get_init_state(self, s):
        for i in range(self.history_size):
            self.history[i, :, :] = self.pre_proc(s)