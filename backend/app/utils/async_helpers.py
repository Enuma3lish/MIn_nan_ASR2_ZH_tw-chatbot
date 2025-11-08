"""
Utility functions for async operations
"""
import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any


# Thread pool for running blocking operations
_thread_pool = ThreadPoolExecutor(max_workers=4)


async def run_in_thread(func: Callable, *args, **kwargs) -> Any:
    """
    Run a blocking function in a thread pool to avoid blocking the event loop.

    Compatible with Python 3.8+ (asyncio.to_thread was added in 3.9)

    Args:
        func: The blocking function to run
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        The result of the function call
    """
    # Python 3.9+ has asyncio.to_thread
    if sys.version_info >= (3, 9):
        return await asyncio.to_thread(func, *args, **kwargs)
    else:
        # Python 3.8 compatibility
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_thread_pool, lambda: func(*args, **kwargs))
