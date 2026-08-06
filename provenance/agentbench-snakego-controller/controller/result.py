from dataclasses import dataclass
from enum import Enum
from typing import Tuple, List, Optional

from logic.infrastructure import Context


class ResultType(Enum):
    NORMAL = 0
    PLAYER_ERROR = 0x10
    ILLEGAL_ACTION = 0x11
    INVALID_FORMAT = 0x12
    INTERNAL_ERROR = 0x20


@dataclass
class Result:
    type: ResultType
    winner: int
    score: List[int]
    err: Optional[str]


def settle_round(ctx: Context) -> Tuple[bool, List[int]]:
    score = [0, 0]
    snake_num = [ctx.get_snake_count(0), ctx.get_snake_count(1)]

    # 计算得分
    now_game_map = ctx.get_map()
    for x in range(ctx.game_map.length):
        for y in range(ctx.game_map.width):
            # 判断该位置是否有蛇
            if now_game_map.snake_map[x][y] != -1:
                this_snake = ctx.get_snake(now_game_map.snake_map[x][y])
                score[this_snake.camp] += 2
            # 判断该位置是否有墙
            elif now_game_map.wall_map[x][y] == 0:
                score[0] += 2
            elif now_game_map.wall_map[x][y] == 1:
                score[1] += 2

    if ctx.first_get_player != -1:
        score[ctx.first_get_player] += 1

    return not (snake_num == [0, 0] or ctx.turn > ctx.max_round), score
