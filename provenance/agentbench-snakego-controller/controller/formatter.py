from enum import Enum


def _is_small_dict(value):
    if len(value.keys()) > 5:
        return False
    value_count = 0
    for v in value.values():
        if type(v) == dict:
            return False
        elif type(v) == list:
            value_count = value_count + len(v)
        else:
            value_count = value_count + 1
    return value_count <= 8


def format_to(fout, value, indent=0):
    def _print_indent():
        for _ in range(indent):
            print('  ', file=fout, end='')

    def _println(s=''):
        print(s, file=fout)

    def _print(s=''):
        print(s, file=fout, end='')

    if isinstance(value, Enum):
        value = value.name
    elif hasattr(value, '__dict__'):
        value = value.__dict__
    elif type(value) == tuple:
        value = list(value)

    if type(value) == dict:
        if _is_small_dict(value):
            _print('{ ')
            kv_count = len(value.items())
            for i, (k, v) in enumerate(value.items()):
                _print(f'"{k}": ')
                format_to(fout, v, indent + 1)
                if i + 1 != kv_count:
                    _print(', ')
            _print(' }')
        else:
            _println('{')
            kv_count = len(value.items())
            for i, (k, v) in enumerate(value.items()):
                _print_indent()
                _print(f'  "{k}": ')
                format_to(fout, v, indent + 1)
                if i + 1 != kv_count:
                    _print(',')
                _println()
            _print_indent()
            _print('}')
    elif type(value) == list:
        if len(value) == 0:
            _print('[]')
        elif type(value[0]) == list or type(value[0]) == dict or hasattr(value[0], '__dict__'):
            # Large list
            _println('[')
            for idx, line in enumerate(value):
                _print_indent()
                _print('  ')
                format_to(fout, line, indent + 1)
                if idx + 1 != len(value):
                    _print(',')
                _println()
            _print_indent()
            _print(']')
        else:
            _print('[')
            for idx, item in enumerate(value):
                _print(item)
                if idx + 1 != len(value):
                    _print(', ')
            _print(']')
    elif type(value) == str:
        _print(f'"{repr(value)[1:-1]}"')
    elif type(value) == bool:
        _print('true' if value else 'false')
    elif value is None:
        _print('null')
    else:
        _print(value)
