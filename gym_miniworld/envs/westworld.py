import string

import numpy as np
from gym import spaces

from gym_miniworld.entity import Box
from gym_miniworld.miniworld import MiniWorldEnv, TextFrame, MeshEnt
from gym_miniworld.utils import get_portrait_names

TOP = 'T'
RIGHT = 'R'
BOTTOM = 'B'
LEFT = 'L'

PORTRAIT_NAMES = list(get_portrait_names())
# CHARACTERS = list(string.ascii_uppercase) + list(string.digits)
CHARACTERS = list(string.digits)


class WestWorld(MiniWorldEnv):
    """
    Maze environment in which the agent has to reach a red box
    """

    def __init__(
            self,
            seed=None,
            room_size=3,
            gap_size=0.25,
            max_episode_steps=None,
            wall_decore_height=None,
            num_chars_on_wall=1,
            **kwargs
    ):

        self.num_rows = 6
        self.num_cols = 6
        self.room_size = room_size
        self.gap_size = gap_size
        self.wall_decore_height = wall_decore_height
        self.num_chars_on_wall = num_chars_on_wall

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

        self.M = None

        super().__init__(
            seed=seed,
            max_episode_steps=max_episode_steps or self.num_rows * self.num_cols * self.room_size * self.action_space.n,
            **kwargs
        )



    def _reset(self):
        self.place_agent()

    def _construct(self):
        self.create_room_matrix()
        self.create_doors()
        self.decorate()
        self.create_buildings()
        self.box = self.place_entity(Box(color='red'), room=self.M[5][1])

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            done = True

        return obs, reward, done, info

    def create_room_matrix(self):
        wall_textures = [
            'airduct_grate',
            'brick_wall',
            'cinder_blocks',
        ]

        floor_textures = []

        M = []
        for i in range(self.num_rows):
            row = []
            for j in range(self.num_cols):
                min_x = j * (self.room_size + self.gap_size)
                max_x = min_x + self.room_size

                min_z = i * (self.room_size + self.gap_size)
                max_z = min_z + self.room_size

                room = self.add_rect_room(
                    min_x=min_x,
                    max_x=max_x,
                    min_z=min_z,
                    max_z=max_z,
                    wall_tex=self.rand.choice(wall_textures),
                    no_ceiling=True,
                    floor_tex='asphalt'
                )
                row.append(room)

            M.append(row)

        self.M = M

    def index2coord(self, index):
        return index // self.num_cols, index % self.num_cols

    def create_doors(self):
        self.create_vertical_doors()
        self.create_horizontal_doors()

    def create_vertical_doors(self):
        M = self.M

        r = 0
        self.connect_at_vertical_wall(M[r][0], M[r][1])
        self.connect_at_vertical_wall(M[r][1], M[r][2])
        self.connect_at_vertical_wall(M[r][2], M[r][3])
        self.connect_at_vertical_wall(M[r][3], M[r][4])
        self.connect_at_vertical_wall(M[r][4], M[r][5])

        r = 1
        self.connect_at_vertical_wall(M[r][0], M[r][1])
        self.connect_at_vertical_wall(M[r][1], M[r][2])
        self.connect_at_vertical_wall(M[r][2], M[r][3])
        self.connect_at_vertical_wall(M[r][3], M[r][4])
        # self.connect_at_vertical_wall(M[r][4], M[r][5])

        r = 2
        self.connect_at_vertical_wall(M[r][0], M[r][1])
        # self.connect_at_vertical_wall(M[r][1], M[r][2])
        self.connect_at_vertical_wall(M[r][2], M[r][3])
        self.connect_at_vertical_wall(M[r][3], M[r][4])
        # self.connect_at_vertical_wall(M[r][4], M[r][5])

        r = 3
        self.connect_at_vertical_wall(M[r][0], M[r][1])
        self.connect_at_vertical_wall(M[r][1], M[r][2])
        # self.connect_at_vertical_wall(M[r][2], M[r][3])
        self.connect_at_vertical_wall(M[r][3], M[r][4])
        self.connect_at_vertical_wall(M[r][4], M[r][5])

        r = 4
        self.connect_at_vertical_wall(M[r][0], M[r][1])
        # self.connect_at_vertical_wall(M[r][1], M[r][2])
        # self.connect_at_vertical_wall(M[r][2], M[r][3])
        self.connect_at_vertical_wall(M[r][3], M[r][4])
        self.connect_at_vertical_wall(M[r][4], M[r][5])

        r = 5
        self.connect_at_vertical_wall(M[r][0], M[r][1])
        # self.connect_at_vertical_wall(M[r][1], M[r][2])
        self.connect_at_vertical_wall(M[r][2], M[r][3])
        self.connect_at_vertical_wall(M[r][3], M[r][4])
        self.connect_at_vertical_wall(M[r][4], M[r][5])

    def create_horizontal_doors(self):
        M = self.M

        c = 0
        # self.connect_at_vertical_wall(M[0][c], M[1][c])
        self.connect_at_horizontal_wall(M[1][c], M[2][c])
        self.connect_at_horizontal_wall(M[2][c], M[3][c])
        self.connect_at_horizontal_wall(M[3][c], M[4][c])
        self.connect_at_horizontal_wall(M[4][c], M[5][c])

        c = 1
        self.connect_at_horizontal_wall(M[0][c], M[1][c])
        self.connect_at_horizontal_wall(M[1][c], M[2][c])
        # self.connect_at_horizontal_wall(M[2][c], M[3][c])
        self.connect_at_horizontal_wall(M[3][c], M[4][c])
        self.connect_at_horizontal_wall(M[4][c], M[5][c])

        c = 2
        # self.connect_at_horizontal_wall(M[0][c], M[1][c])
        self.connect_at_horizontal_wall(M[1][c], M[2][c])
        self.connect_at_horizontal_wall(M[2][c], M[3][c])
        self.connect_at_horizontal_wall(M[3][c], M[4][c])
        self.connect_at_horizontal_wall(M[4][c], M[5][c])

        c = 3
        # self.connect_at_horizontal_wall(M[0][c], M[1][c])
        self.connect_at_horizontal_wall(M[1][c], M[2][c])
        self.connect_at_horizontal_wall(M[2][c], M[3][c])
        self.connect_at_horizontal_wall(M[3][c], M[4][c])
        self.connect_at_horizontal_wall(M[4][c], M[5][c])

        c = 4
        # self.connect_at_horizontal_wall(M[0][c], M[1][c])
        self.connect_at_horizontal_wall(M[1][c], M[2][c])
        self.connect_at_horizontal_wall(M[2][c], M[3][c])
        self.connect_at_horizontal_wall(M[3][c], M[4][c])
        self.connect_at_horizontal_wall(M[4][c], M[5][c])

        c = 5
        self.connect_at_horizontal_wall(M[0][c], M[1][c])
        self.connect_at_horizontal_wall(M[1][c], M[2][c])
        self.connect_at_horizontal_wall(M[2][c], M[3][c])
        self.connect_at_horizontal_wall(M[3][c], M[4][c])
        self.connect_at_horizontal_wall(M[4][c], M[5][c])

    def connect_at_vertical_wall(self, left, right):
        M = self.M
        self.connect_rooms(left, right,
                           min_z=left.min_z,
                           max_z=left.max_z)

    def connect_at_horizontal_wall(self, top, down):
        M = self.M
        self.connect_rooms(top, down,
                           min_x=down.min_x,
                           max_x=down.max_x)

    def decorate(self):
        r = 0
        self.decorate_room(self.M[r][0], TOP, BOTTOM, LEFT)
        self.decorate_room(self.M[r][1], TOP)
        self.decorate_room(self.M[r][2], TOP, BOTTOM)
        self.decorate_room(self.M[r][3], TOP, BOTTOM)
        self.decorate_room(self.M[r][4], TOP, BOTTOM)
        self.decorate_room(self.M[r][5], TOP, RIGHT)

        r = 1
        self.decorate_room(self.M[r][0], TOP, LEFT)
        # self.decorate_room(self.M[r][1])
        self.decorate_room(self.M[r][2], TOP)
        self.decorate_room(self.M[r][3], TOP)
        self.decorate_room(self.M[r][4], TOP, RIGHT)
        self.decorate_room(self.M[r][5], LEFT, RIGHT)

        r = 2
        self.decorate_room(self.M[r][0], LEFT)
        self.decorate_room(self.M[r][1], RIGHT, BOTTOM)
        self.decorate_room(self.M[r][2], LEFT)
        # self.decorate_room(self.M[r][3])
        self.decorate_room(self.M[r][4], RIGHT)
        self.decorate_room(self.M[r][5], LEFT, RIGHT)

        r = 3
        self.decorate_room(self.M[r][0], LEFT)
        self.decorate_room(self.M[r][1], TOP)
        self.decorate_room(self.M[r][2], RIGHT)
        self.decorate_room(self.M[r][3], LEFT)
        # self.decorate_room(self.M[r][4], )
        self.decorate_room(self.M[r][5], RIGHT)

        r = 4
        self.decorate_room(self.M[r][0], LEFT)
        self.decorate_room(self.M[r][1], RIGHT)
        self.decorate_room(self.M[r][2], LEFT, RIGHT)
        self.decorate_room(self.M[r][3], LEFT)
        # self.decorate_room(self.M[r][4], )
        self.decorate_room(self.M[r][5], RIGHT)

        r = 5
        self.decorate_room(self.M[r][0], BOTTOM, LEFT)
        self.decorate_room(self.M[r][1], BOTTOM, RIGHT)
        self.decorate_room(self.M[r][2], BOTTOM, LEFT)
        self.decorate_room(self.M[r][3], BOTTOM)
        self.decorate_room(self.M[r][4], BOTTOM)
        self.decorate_room(self.M[r][5], BOTTOM, RIGHT)

    def decorate_room(self, room, *walls):
        y = room.wall_height / 2
        height = self.wall_decore_height or room.wall_height / self.num_chars_on_wall
        for wall in walls:
            x, z = wall_center_xz(room, wall)
            text = ''.join(self.rand.subset(CHARACTERS, k=self.num_chars_on_wall))
            entity = TextFrame(pos=(0, 0), dir=0, str=text, height=height)
            # entity = ImageFrame(pos=(0, 0),
            #                     dir=0,
            #                    tex_name='portraits/' + self.rand.choice(PORTRAIT_NAMES),
            #                     width=1)
            self.place_entity(
                ent=entity,
                pos=(x, y, z),
                dir=plane_normal(wall),
                room=room
            )

    def create_buildings(self):
        maze_width = (self.room_size + self.gap_size) * self.num_cols
        maze_height = (self.room_size + self.gap_size) * self.num_rows

        self.place_entity(
            MeshEnt(
                mesh_name='building',
                height=10
            ),
            pos=np.array([maze_width + 10, 0, maze_height + 10]),
            dir=-np.pi
        )


def wall_center_xz(room, wall):
    min_x = room.min_x
    max_x = room.max_x

    min_z = room.min_z
    max_z = room.max_z

    mid_x = (min_x + max_x) / 2
    mid_z = (min_z + max_z) / 2
    if wall == TOP:
        return mid_x, min_z
    if wall == RIGHT:
        return max_x, mid_z
    if wall == BOTTOM:
        return mid_x, max_z
    if wall == LEFT:
        return min_x, mid_z


def plane_normal(wall):
    if wall == TOP:
        return 3 * np.pi / 2
    if wall == RIGHT:
        return np.pi
    if wall == BOTTOM:
        return np.pi / 2
    if wall == LEFT:
        return 0.0
