from functools import wraps
from time import time

DATA_ROOT = "C:\\Users\\novak\\Desktop\\psiml project\\psiml_code_completion\\data\\"

TRAIN = "python100k"
TEST = "python50k"
SMALL = "python1k"


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"func:{f.__name__} took: {te - ts}s")
        return result

    return wrap
