from typing import List
from dataclasses import dataclass
import random
import time
from logic.constants import ITEM_EXPIRE_TIME

@dataclass
class Item:
    id: int
    x: int
    y: int
    time: int
    type: int
    param: int
    gotten_time: int

    item_num = 0

    def __init__(self, x: int, y: int, time: int, type: int, param: int):
        self.x = x
        self.y = y
        self.time = time
        self.type = type
        self.param = param
        self.id = Item.item_num
        self.gotten_time = -1
        Item.item_num += 1

@dataclass(init=False)
class GameConfig:
    length: int
    width: int
    max_round: int
    random_seed: int

    def __init__(self, length=16, width=16, max_round=512):
        self.length = length
        self.width = width
        self.max_round = max_round
        self.random_seed = int(time.time() * 1000)


@dataclass
class SpawnConfig:
    '''
    area: x1,y1,x2,y2
    weight: length, split, fire
    '''
    start_round: int
    end_round: int
    num_per_round_mean: float
    area: (int, int, int, int)
    weight: (int, int, int)
    last: int

    def __init__(self, start_round: int, end_round: int, mean: float, area: (int, int, int, int), weight: (int, int, int)):
        self.start_round = start_round
        self.end_round = end_round
        self.num_per_round_mean = mean
        self.area = area
        self.weight = weight
        self.last = ITEM_EXPIRE_TIME


config_list = [
    SpawnConfig(1, 20, 0.25, (6, 6, 9, 9), (5, 0, 1)),
    SpawnConfig(21, 40, 0.25, (5, 5, 10, 10), (5, 0, 1)),
    SpawnConfig(41, 60, 0.25, (4, 4, 11, 11), (5, 0, 1)),
    SpawnConfig(61, 64, 0.25, (3, 3, 12, 12), (5, 0, 1)),
    SpawnConfig(65, 80, 0.25, (3, 3, 12, 12), (5, 0, 2)),
    SpawnConfig(81, 100, 0.25, (2, 2, 13, 13), (5, 0, 2)),
    SpawnConfig(101, 120, 0.25, (1, 1, 14, 14), (5, 0, 2)),
    SpawnConfig(121, 384, 0.25, (0, 0, 15, 15), (5, 0, 2)),
    SpawnConfig(385, 512, 0.25, (0, 0, 15, 15), (4, 0, 3))
]


def generate_items(game_config: GameConfig) -> List[Item]:
    def random_position(x1: int, y1: int, x2: int, y2: int) -> (int, int):
        '''
         x1 <= x2, y1 <= y2
        '''
        x = random.randint(x1, x2)
        y = random.randint(y1, y2)
        return x, y

    def generate_single(round: int, config: SpawnConfig):
        assert round in range(config.start_round, config.end_round + 1)
        x, y = random_position(*config.area)
        while item_map[x][y] + ITEM_EXPIRE_TIME >= round:
            x, y = random_position(*config.area)
        item_map[x][y] = round
        t = random.randint(0, sum(config.weight) - 1)
        if t in range(config.weight[0]):
            return Item(x, y, round, 0, random.randint(1, 5))
        elif t in range(config.weight[0] + config.weight[1]):
            return Item(x, y, round, 0, random.randint(1, 5))
        else:
            return Item(x, y, round, 2, game_config.max_round)

    random.seed(game_config.random_seed)
    items = []
    item_map = [[-1024 for y in range(game_config.width)] for x in range(game_config.length)]
    current_round = 1
    for config in config_list:
        while current_round in range(config.start_round, config.end_round + 1):
            score = random.uniform(0, 1)
            if score <= config.num_per_round_mean:
                items.append(generate_single(current_round, config))
            current_round += 1

    return items
