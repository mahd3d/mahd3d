from time import time


def timeit(func):
    def timed(*args, **kw):
        start = time()
        result = func(*args, **kw)
        end = time()
        duration = end - start
        print(f"{func.__name__}() took {duration} seconds")
        return result

    return timed
