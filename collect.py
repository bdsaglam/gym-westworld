import pathlib
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

from gym_miniworld.envs import WestWorld
from gym_miniworld.envs.westworld import DecoreOption


def collect_data(data_dir,
                 seed,
                 obs_size,
                 num_episodes=1000,
                 timesteps_per_episode=100,
                 first_episode=0,
                 decore_option=DecoreOption.PORTRAIT):
    folder = data_dir / f'{obs_size}x{obs_size}-s{seed}-dc'

    random.seed(seed)
    np.random.seed(seed)

    env = WestWorld(
        seed=seed,
        obs_width=obs_size,
        obs_height=obs_size,
        decore_option=decore_option,
        num_chars_on_wall=1,
    )

    image_dir = folder / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)
    action_dir = folder / 'actions'
    action_dir.mkdir(parents=True, exist_ok=True)
    pose_dir = folder / 'poses'
    pose_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(desc='Episode', total=num_episodes)
    episode = first_episode
    while episode < (first_episode + num_episodes):
        episode += 1
        env.reset()
        for i in range(timesteps_per_episode):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)

            x, y, z = env.agent.pos
            orientation = env.agent.dir
            pose_info = f'{x:.2f} {y:.2f} {z:.2f} {orientation:.2f}'

            sub_dir_name = f'{episode:08d}'
            filename = f'{i:08d}'

            ep_image_dir = (image_dir / sub_dir_name)
            ep_image_dir.mkdir(parents=True, exist_ok=True)
            ep_action_dir = (action_dir / sub_dir_name)
            ep_action_dir.mkdir(parents=True, exist_ok=True)
            ep_pose_dir = (pose_dir / sub_dir_name)
            ep_pose_dir.mkdir(parents=True, exist_ok=True)

            image = Image.fromarray(obs, 'RGB')
            image_file = ep_image_dir / (filename + '.jpg')
            image.save(str(image_file))

            action_file = ep_action_dir / (filename + '.txt')
            action_file.write_text(str(action))

            pose_file = ep_pose_dir / (filename + '.txt')
            pose_file.write_text(pose_info)

            if done:
                break
        pbar.update(1)

    pbar.close()

    env.close()
    print("finished")
    return folder


if __name__ == '__main__':
    data_dir = pathlib.Path.home() / 'westworld-data'
    collect_data(data_dir=data_dir, seed=42, obs_size=64,
                 num_episodes=1000, timesteps_per_episode=100)
