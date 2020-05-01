import pathlib
import random

import numpy as np
from PIL import Image

from gym_miniworld.envs import WestWorld


def collect_data():
    root = pathlib.Path('/Users/bdsaglam/westworld-data/validation')

    image_dir = root / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)
    pose_dir = root / 'poses'
    pose_dir.mkdir(parents=True, exist_ok=True)

    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    env = WestWorld(
        seed=seed,
        obs_width=300,
        obs_height=300,
    )


    episode = 0
    while episode < 30:
        episode += 1
        env.reset()
        for i in range(10):
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



    print("finished")


if __name__ == '__main__':
    collect_data()
