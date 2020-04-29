import random

import gym
import numpy as np

if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    env = gym.make('MiniWorld-WestWorld-v0', seed=seed)

    env.reset()

    for _ in range(10):
        action = env.action_space.sample()
        env.step(action)

    print("finished")
