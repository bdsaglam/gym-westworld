import pathlib
import random

import numpy as np
from PIL import Image

from gym_miniworld.envs import WestWorld
from gym_miniworld.envs.westworld import DecoreOption


def collect_data(data_dir, seed, obs_size, num_episodes=1000, timesteps_per_episode=100):
    folder = data_dir / f'{obs_size}x{obs_size}-s{seed}-dc'

    random.seed(seed)
    np.random.seed(seed)

    env = WestWorld(
        seed=seed,
        obs_width=obs_size,
        obs_height=obs_size,
        decore_option=(DecoreOption.DIGIT | DecoreOption.CHARACTER),
        num_chars_on_wall=1,
    )

    image_dir = folder / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)
    pose_dir = folder / 'poses'
    pose_dir.mkdir(parents=True, exist_ok=True)

    episode = 0
    while episode < num_episodes:
        episode += 1
        env.reset()
        for i in range(timesteps_per_episode):
            action = env.action_space.sample()

            obs, reward, done, info = env.step(action)
            for j in range(random.randint(0, 10)):
                obs, reward, done, info = env.step(action)

            x, y, z = env.agent.pos
            orientation = env.agent.dir
            pose_info = f'{x:.2f} {y:.2f} {z:.2f} {orientation:.2f}'

            sub_dir_name = f'{episode:08d}'
            filename = f'{i:08d}'

            ep_image_dir = (image_dir / sub_dir_name)
            ep_image_dir.mkdir(parents=True, exist_ok=True)
            ep_pose_dir = (pose_dir / sub_dir_name)
            ep_pose_dir.mkdir(parents=True, exist_ok=True)

            image = Image.fromarray(obs, 'RGB')
            image_file = ep_image_dir / (filename + '.jpg')
            image.save(str(image_file))

            pose_file = ep_pose_dir / (filename + '.txt')
            pose_file.write_text(pose_info)

            if done:
                break

    env.close()
    print("finished")
    return folder


if __name__ == '__main__':
    data_dir = pathlib.Path('~/westworld-data')
    collect_data(data_dir=data_dir, seed=0, obs_size=64)
