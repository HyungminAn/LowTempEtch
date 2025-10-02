import functools
import time


def log_function_call(func, indent_level=[0]):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tab = indent_level[0] * "    "
        indent_level[0] += 1
        clock = f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
        print(f"{tab}Function {func.__name__} started at {clock}")

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        clock = f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
        print(f"{tab}Function {func.__name__} ended at {clock}")
        elapsed_time = end_time - start_time
        if elapsed_time > 1e-03:
            print(f"{tab}    Elapsed time: {elapsed_time:.3f} s")
        else:
            print(f"{tab}    Elapsed time: < 1 ms")

        indent_level[0] -= 1
        return result
    return wrapper
