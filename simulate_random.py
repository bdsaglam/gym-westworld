import random
import time

import numpy as np
import torch
from gym.wrappers import Monitor

from gym_miniworld.envs import WestWorld
from gym_miniworld.envs.westworld import DecoreOption


def simulate(
        env,
        agent,
        deterministic=True,
        num_episodes=3,
        render=True,
        wait_after_render=1e-3,
        render_kwargs=None,
        record_video=False
):
    render_kwargs = render_kwargs or dict()

    assert env.max_episode_steps > 0
    if record_video:
        env = Monitor(env, directory='./data')

    episode_info = []
    for _ in range(num_episodes):
        obs = env.reset()
        agent.reset()
        done = False
        episode_return = 0
        t = 0
        while not done:
            if render:
                env.render(**render_kwargs)
                time.sleep(wait_after_render)

            with torch.no_grad():
                action = agent.act(obs, deterministic)
            obs, reward, done, _ = env.step(action)
            episode_return += reward
            t += 1
        episode_info.append((t, episode_return))

    return episode_info


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, *args, **kwargs):
        return self.action_space.sample()

    def reset(self):
        pass


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    env = WestWorld(
        seed=seed,
        obs_width=128,
        obs_height=128,
        decore_option=DecoreOption.ALL,
        max_episode_steps=200
    )
    agent = RandomAgent(env.action_space)
    simulate(
        env,
        agent,
        render=True,
        render_kwargs=dict(mode='human', view='top'),
        wait_after_render=0.05,
        record_video=True
    )
    env.close()
