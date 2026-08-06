import json
import logging
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Tuple, List, Optional

from logic.infrastructure import Item, Operation


def default_dump(x):
    if isinstance(x, Enum):
        return x.value
    else:
        return x.__dict__


class ErrorType(Enum):  # 错误信息类别
    RE = 0
    TLE = 1
    OLE = 2
    IA = 3
    PE = 4


@dataclass
class PlayerError(RuntimeError):
    type: ErrorType
    player: int
    err_msg: str


class round_info:  # 回合开始信息，需依托处理部分补完
    gotten_item = [{"item": int, "snake": int}]
    expired_item = [{"item": int, "snake": int}]
    invalid_item = [{"item": int, "snake": int}]
    extra_length = [{"snake": int, "el": int}]


class processed_operation:  # 附加信息，需依托处理部分补完
    basic = [Operation]
    dead_snake = [int]
    should_solid = bool
    extra_solid_area = [{"x": int, "y": int}]
    got_item = int
    extra_length = int
    new_snake_id = int
    pass


def send(target: int, msg: str):  # 与judger通信发送数据
    if not msg.endswith('\n'):
        msg = msg + '\n'
    message_len = len(msg)
    message = message_len.to_bytes(4, byteorder='big', signed=True)
    message += target.to_bytes(4, byteorder='big', signed=True)
    message += bytes(msg, encoding="utf8")
    sys.stdout.buffer.write(message)
    sys.stdout.flush()


def send_bytes(target: int, msg: bytes):
    message_len = len(msg)
    message = message_len.to_bytes(4, byteorder='big', signed=True)
    message += target.to_bytes(4, byteorder='big', signed=True)
    message += msg
    sys.stdout.buffer.write(message)
    sys.stdout.flush()


def listen():  # 与judger通信接收数据
    read_buffer = sys.stdin.buffer
    data_len = int.from_bytes(read_buffer.read(4), byteorder='big', signed=True)
    data = json.loads(read_buffer.read(data_len).decode())
    logging.debug('Reading data: ' + str(data))
    return data


def receive_metadata() -> ([int], int, str):  # 接受初始化信息（玩家列表、玩家数、回放路径）
    v = listen()
    return v['player_list'], v['player_num'], v['reply']


def send_normal_round_msg(state: int, listen_target: [int], player: [int],
                          content: [str], time: int = -1):  # 指定监听玩家、发送信息玩家及信息(开启监听可通过调用其实现，设定player及content为空)
    if time > 0:
        send(-1, json.dumps({'state': 0, 'time': time, 'length': 2048}))
    send(-1, json.dumps({'state': state, 'listen': listen_target, 'player': player, 'content': content}))


def send_direct_msg_to_player(player: int, content: Any):
    send_bytes(player, content)


def send_direct_msg_to_human(player: int, content: Any):
    send(player, json.dumps(content, default=default_dump))


def broadcast_msg(state: int, listen_target: [int], content: Any):
    content = json.dumps(content, default=default_dump)
    send_normal_round_msg(state, listen_target, [0, 1], [content, content])


def broadcast_msg_1(content: Any):
    for i in range(2):
        send_direct_msg_to_player(i, content)


def send_items(items: [Item]):  # 广播道具列表信息
    content = []
    for item in items:
        data = {"x": item.x, "y": item.y, "time": item.time, "type": item.type, "param": item.param}
        content.append(data)
    broadcast_msg(0, [], content)



def send_next_player(state: int, next_player: int):
    broadcast_msg(state, [next_player], {
        "next_player": next_player
    })


def read_operation(current: int, player: int, round: int) -> Operation:  # 运行直到获取监听对象信息，并判断是否有judger传来的异常，若无任何异常或错误最后返回操作信息
    while True:
        v = listen()
        if v['player'] >= 0:
            if v['player'] == current:
                if player == 1:
                    return on_received_operation(v['player'], v['content'])  # 无异常将信息交给处理函数
                else:
                    return on_received_human_operation(v['player'], v['content'])
            else:
                raise PlayerError(ErrorType.OLE, v['player'], "Too much operations!")
        else:
            error_content = json.loads(v['content'])
            if error_content['error'] != 1 or (error_content['error'] == 1 and error_content['state'] == round):
                raise PlayerError(ErrorType(error_content['error']), error_content['player'], error_content['error_log'])


def on_received_human_operation(player: int, content: str) -> Operation:
    op = json.loads(content)
    if op.get('type') != 1 and op.get('type') != 2 and op.get('type') != 3:
        raise PlayerError(ErrorType.PE, player, "Invalid message format: " + content)
    else:
        return Operation(type=op['direction'] + 1 if op['type'] == 1 else op['type'] + 3, snake=-1, direction=-1, item_id=-1)


def on_received_operation(player: int, content: str) -> Operation:
    try:
        op = int.from_bytes(bytes(content, encoding='utf-8'), byteorder='big', signed=False)
        if op < 1 or op > 6:
            raise PlayerError(ErrorType.PE, player, "Invalid message format")
        else:
            return Operation(type=op, snake=-1, direction=-1, item_id=-1)
    except Exception as err:
        raise PlayerError(ErrorType.PE, player, "Invalid message format")


def send_processed_operation(state: int, current_player: int, p_opt: processed_operation):
    broadcast_msg(state, [current_player], p_opt)


def send_game_over(scores: List[int], state: List[int]):
    end_state = ["OK" if i == 0 else "IA" for i in state]
    send(-1, json.dumps({
        "state": -1,
        "end_info": json.dumps({
            "0": scores[0],
            "1": scores[1]
        }),
        "end_state": json.dumps(end_state)
    }))
