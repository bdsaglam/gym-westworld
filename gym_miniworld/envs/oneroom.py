from gym import spaces

from gym_miniworld.entity import Box
from gym_miniworld.miniworld import MiniWorldEnv
from gym_miniworld.params import DEFAULT_PARAMS


class OneRoom(MiniWorldEnv):
    """
    Environment in which the goal is to go to a red box
    placed randomly in one big room.
    """

    def __init__(self, size=10, max_episode_steps=180, have_goal=False, **kwargs):
        assert size >= 2
        self.size = size
        self.have_goal = have_goal
        self.height = size
        self.width = size

        params = DEFAULT_PARAMS
        params.set('turn_step', 5, 3, 7)
        params.set('forward_step', 0.2, 0.15, 0.25)

        super().__init__(
            params=params,
            max_episode_steps=max_episode_steps,
            **kwargs
        )

        # Allow only movement actions (left/right/forward/backward)
        self.action_space = spaces.Discrete(self.actions.move_back + 1)

    def _reset(self):
        self.place_agent()

    def _construct(self):
        room = self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size
        )
        if self.have_goal:
            self.box = self.place_entity(Box(color='red'))

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.have_goal and self.near(self.box):
            reward += self._reward()
            done = True

        return obs, reward, done and self.have_goal, info
