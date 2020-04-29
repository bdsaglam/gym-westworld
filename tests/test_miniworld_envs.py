import unittest

import gym
from gym_miniworld.wrappers import PyTorchObsWrapper


class WestWorldTestCase(unittest.TestCase):
    def test_env(self):
        env = gym.make('MiniWorld-WestWorld-v0')

        # Try stepping a few times
        for i in range(0, 10):
            obs, _, _, _ = env.step(0)

        # Check that the human rendering resembles the agent's view
        first_obs = env.reset()
        first_render = env.render('rgb_array')
        m0 = first_obs.mean()
        m1 = first_render.mean()
        self.assertTrue(0 < m0 < 255)
        self.assertTrue(abs(m0 - m1) < 5)

        # Check that the observation shapes match in reset and step
        second_obs, _, _, _ = env.step(0)
        self.assertTrue(first_obs.shape == env.observation_space.shape)
        self.assertTrue(first_obs.shape == second_obs.shape)

        # Test the PyTorch observation wrapper
        env = PyTorchObsWrapper(env)
        first_obs = env.reset()
        second_obs, _, _, _ = env.step(0)
        self.assertTrue(first_obs.shape == env.observation_space.shape)
        self.assertTrue(first_obs.shape == second_obs.shape)


if __name__ == '__main__':
    unittest.main()
