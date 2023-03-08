import time
from functools import wraps


def timed_print(func):
    """Argument printing and time logging decorator.

    Args:
        func: Any function.
    """

    @wraps(func)
    def dec(*args, **kwargs):
        print('Function called:', func.__name__)
        print(args, **kwargs)
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        print('Time:', round((te - ts) * 1000, 1), 'ms')
        print()
        return result

    return dec


def timed(func):
    """Argument printing and time logging decorator.

    Args:
        func: Any function.
    """

    @wraps(func)
    def dec(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        print('Time:', round((te - ts) * 1000, 1), 'ms')
        print()
        return result

    return dec


if __name__ == '__main__':
    pass