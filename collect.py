import json
import pathlib
import random

import numpy as np
from PIL import Image
from gym_miniworld.envs.westworld import DecoreOption, WestWorld
from tqdm import tqdm

from gym_miniworld.envs.oneroom import OneRoom


def collect_data(directory,
                 seed,
                 obs_size,
                 num_episodes=1000,
                 timesteps_per_episode=100,
                 first_episode=0,
                 folder_prefix='',
                 save_images=True,
                 save_actions=True,
                 save_poses=True):
    if not any((save_images, save_actions, save_poses)):
        raise ValueError("At least one of the data must be saved.")

    random.seed(seed)
    np.random.seed(seed)

    env = WestWorld(
        seed=seed,
        obs_width=obs_size,
        obs_height=obs_size,
        room_size=2,
        decore_option=DecoreOption.PORTRAIT,
        non_terminate=True
    )

    # env = OneRoom(size=20, max_episode_steps=100, have_goal=False)

    action_probs = np.array([0.15, 0.15, 0.6, 0.1])
    stuck_action_probs = np.array([0.4, 0.4, 0.0, 0.2])

    data_flags = ''.join(
        c if b else ''
        for b, c in ((save_images, 'i'), (save_actions, 'a'), (save_poses, 'p'))
    )
    folder = directory / f'{folder_prefix}-{data_flags}-{obs_size}x{obs_size}-s{seed}'
    folder.mkdir(parents=True, exist_ok=True)

    config_file = folder / 'config.json'
    config = dict(
        obs_size=obs_size,
        num_episodes=num_episodes,
        timesteps_per_episode=timesteps_per_episode,
        action_space_dim=env.action_space.n,
        width=env.width,
        height=env.height,
    )
    config_file.write_text(json.dumps(config))

    top_image = Image.fromarray(env.render(view='top'))
    top_image.save(folder / 'top_view.png')

    if save_images:
        image_dir = folder / 'images'
        image_dir.mkdir(parents=True, exist_ok=True)
    if save_actions:
        action_dir = folder / 'actions'
        action_dir.mkdir(parents=True, exist_ok=True)
    if save_poses:
        pose_dir = folder / 'poses'
        pose_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(desc='Episode', total=num_episodes)
    episode = first_episode
    prev_pos = None
    while episode < (first_episode + num_episodes):
        episode += 1
        env.reset()

        sub_dir_name = f'{episode:08d}'
        if save_images:
            ep_image_dir = (image_dir / sub_dir_name)
            ep_image_dir.mkdir(parents=True, exist_ok=True)

        if save_actions:
            ep_action_file = action_dir / (sub_dir_name + '.csv')
            fa = ep_action_file.open('w')
            fa.write('action_index')

        if save_poses:
            ep_pose_file = pose_dir / (sub_dir_name + '.csv')
            fp = ep_pose_file.open('w')
            fp.write('x,y,z,phi')

        for i in range(timesteps_per_episode):
            n_actions = env.action_space.n
            action = np.random.choice(np.arange(n_actions), p=action_probs)
            obs, reward, done, info = env.step(action)

            if prev_pos is not None and np.allclose(prev_pos, env.agent.pos):
                action = np.random.choice(np.arange(n_actions), p=stuck_action_probs)
                obs, reward, done, info = env.step(action)

            if save_images:
                filename = f'{i:08d}'
                image = Image.fromarray(obs, 'RGB')
                image_file = ep_image_dir / (filename + '.jpg')
                image.save(str(image_file))

            if save_actions:
                fa.write('\n' + str(action))

            if save_poses:
                x, y, z = env.agent.pos
                orientation = env.agent.dir
                pose_info = f'{x:.2f},{y:.2f},{z:.2f},{orientation:.2f}'
                fp.write('\n' + pose_info)

            prev_pos = env.agent.pos

            if done:
                break

        if save_actions:
            fa.close()
        if save_poses:
            fp.close()

        pbar.update(1)

    pbar.close()
    env.close()

    return folder


if __name__ == '__main__':
    directory = pathlib.Path.home() / 'westworld-data'
    collect_data(directory=directory,
                 seed=42,
                 obs_size=64,
                 num_episodes=1024,
                 timesteps_per_episode=128,
                 first_episode=0,
                 folder_prefix='ww',
                 save_images=True)
