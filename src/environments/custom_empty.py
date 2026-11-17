from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv


class VariableSizeEmptyEnv(MiniGridEnv):
    def __init__(self, size=5, max_steps=None, **kwargs):
        self.size = size
        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * (size ** 2)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=True,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "reach the goal"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Random goal placement
        self.place_obj(Goal(), top=(1, 1), size=(width - 2, height - 2))

        # Random agent placement
        self.place_agent()
        self.mission = "reach the goal"