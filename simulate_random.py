import random
import time

import numpy as np
import torch

from gym_miniworld.envs import WestWorld


def simulate(
        env,
        agent,
        deterministic=True,
        num_episodes=5,
        episode_len_limit=None,
        render=True,
        wait_after_render=1e-3,
        render_kwargs=None
):
    render_kwargs = render_kwargs or dict()
    if episode_len_limit is None:
        if env.unwrapped.spec and env.unwrapped.spec.max_episode_steps:
            episode_len_limit = env.spec.max_episode_steps
        else:
            raise ValueError("Episode length limit must be specified")

    episode_info = []
    for _ in range(num_episodes):
        obs = env.reset()
        agent.reset()
        done = False
        episode_return = 0
        t = 0
        while not done and t != episode_len_limit:
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
        obs_width=200,
        obs_height=200,
    )
    agent = RandomAgent(env.action_space)
    simulate(
        env,
        agent,
        episode_len_limit=1000,
        render=True,
        render_kwargs=dict(mode='human', view='top'),
    )
