import time
import torch

from functools import wraps
from inspect import signature


def output_cuda_resource_usage(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        # Retrieve device regardless of whether it was passed
        # positionally or as a keyword argument.
        bound_args = signature(func).bind(*args, **kwargs)
        device = bound_args.arguments.get("device")

        if device is None or device.type != "cuda":
            return func(*args, **kwargs)

        # Reseting GPU memory stats
        torch.cuda.reset_peak_memory_stats(device)

        # Starting runtime measurement
        torch.cuda.synchronize(device)
        start_time = time.perf_counter()

        try:
            return func(*args, **kwargs)

        finally:
            torch.cuda.synchronize(device)

            runtime = time.perf_counter() - start_time

            print(f"\nTraining runtime: {runtime:.2f} seconds")

            peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

            print(f"Peak GPU memory allocated: {peak_allocated:.2f} MB")
            print(f"Peak GPU memory reserved:  {peak_reserved:.2f} MB")

    return wrapper
