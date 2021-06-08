import time
from utils import log


def singleton(cls):
    """
    Singleton pattern decorator
    How to use:
        @Singleton
        class A(object):
            pass

    :param cls:
    :return:
    """
    _instance = {}

    def _singleton(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return _singleton


def count_time(tag=""):
    def ctime(func):
        def wrapper(*args, **kwargs):
            tic = time.time()  # Program start time
            r = func(*args, **kwargs)
            toc = time.time()  # Program end time
            cost = toc - tic
            log.info("[ cost_time ] {tag} {func_name} > {cost}".format(tag=tag, func_name=func.__name__, cost=cost))
            return r

        return wrapper

    return ctime
