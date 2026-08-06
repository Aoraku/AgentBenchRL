import io
import random
import time
import zipfile
from dataclasses import dataclass
from typing import List, Tuple
from .formatter import format_to

from logic.spawn import generate_items, GameConfig, Item

@dataclass
class Snake:
    coor_list: List[Tuple[int, int]]
    item_list: List[Item]
    length_bank: int
    camp: int
    id: int

    snake_num = 0

    def __init__(self, coor_list: List[Tuple[int, int]], item_list: List[Item], camp: int, id=-1):
        self.coor_list = coor_list.copy()
        self.item_list = item_list.copy()
        self.length_bank = 0
        self.camp = camp
        if id == -1:
            self.id = Snake.snake_num
            Snake.snake_num += 1
        else:
            self.id = id

    def get_len(self) -> int:
        return len(self.coor_list)

    def get_coor(self) -> List[Tuple[int, int]]:
        return self.coor_list

    def add_item(self, item: Item) -> None:
        if item.type == 0:
            self.length_bank += item.param
        else:
            fl = False
            for idx in range(len(self.item_list)):
                if self.item_list[idx].type == item.type:
                    self.item_list[idx] = item
                    fl = True
                    break
            if not fl:
                self.item_list.append(item)

    def get_item(self, id: int) -> Item:
        for _item in self.item_list:
            if _item.id == id:
                return _item

    def delete_item(self, id: int) -> None:
        for _item in self.item_list:
            if _item.id == id:
                self.item_list.remove(_item)
                break


@dataclass
class Map:
    item_list: List[Item]
    wall_map: List[List[int]]
    snake_map: List[List[int]]
    item_map: List[List[int]]
    length: int
    width: int

    def __init__(self, item_list: [Item], config: GameConfig):
        self.length = config.length
        self.width = config.width
        self.item_list = item_list
        self.wall_map = [[-1 for y in range(config.width)] for x in range(config.length)]
        self.snake_map = [[-1 for y in range(config.width)] for x in range(config.length)]
        self.snake_map[0][config.width - 1] = 0
        self.snake_map[config.length - 1][0] = 1
        self.item_map = [[-1 for y in range(config.width)] for x in range(config.length)]

    def set_wall(self, coor_list: List, camp: int, type: int) -> None:
        if type == 1:
            for x, y in coor_list:
                self.wall_map[x][y] = camp
        elif type == -1:
            for x, y in coor_list:
                self.wall_map[x][y] = -1

    def get_map_item(self, id: int) -> Item:
        for _item in self.item_list:
            if _item.id == id:
                return _item
        return None

    def add_map_item(self, item: Item) -> None:
        self.item_list.append(item)
        self.item_map[item.x][item.y] = item.id

    def delete_map_item(self, id: int) -> None:
        for _item in self.item_list:
            if _item.id == id:
                self.item_map[_item.x][_item.y] = -1
                self.item_list.remove(_item)
                break

    def add_map_snake(self, coor_list: List, id: int) -> None:
        for x, y in coor_list:
            self.snake_map[x][y] = id

    def delete_map_snake(self, coor_list: List) -> None:
        for x, y in coor_list:
            self.snake_map[x][y] = -1


@dataclass
class Context:
    snake_list: List[Snake]
    game_map: Map
    turn: int
    current_player: int
    auto_growth_round: int
    max_round: int

    def __init__(self, config: GameConfig):
        self.snake_list = [
            Snake([(0, config.width - 1)], [], 0, -1),
            Snake([(config.length - 1, 0)], [], 1, -1)
        ]
        self.game_map = Map(generate_items(config), config)
        self.turn = 0
        self.current_player = 0
        self.auto_growth_round = 8
        self.max_round = config.max_round
        self.first_get_time = -1
        self.first_get_player = -1

    def get_map(self) -> Map:
        return self.game_map

    def get_snake_count(self, camp: int) -> int:
        return sum(x.camp == camp for x in self.snake_list)

    def get_snake(self, id: int) -> Snake:
        for _snake in self.snake_list:
            if _snake.id == id:
                return _snake

    def add_snake(self, snake: Snake, index: int) -> None:
        self.snake_list.insert(index, snake)
        self.game_map.add_map_snake(snake.coor_list, snake.id)

    def delete_snake(self, id: int) -> int:
        index = -1  # prev id of the deleted
        for _snake in self.snake_list:
            if _snake.id == id:
                self.game_map.delete_map_snake(_snake.coor_list)
                self.snake_list.remove(_snake)
                return index
            index = _snake.id


@dataclass
class Operation:
    type: int
    snake: int
    direction: int
    item_id: int


class History:
    def __init__(self):
        self.game_config = {}
        self.item_list = []
        self.round_info = []
        self.operations = []
        self.end_info = {}

    def dump(self, path):
        self.item_list = list(map(lambda item: {
            'x': item.x,
            'y': item.y,
            'time': item.time,
            'type': item.type,
            'param': item.param
        }, self.item_list))

        with io.StringIO() as buf:
            format_to(buf, self)
            with open(path, 'w') as file:
                file.write(buf.getvalue())

