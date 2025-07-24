import functools
import time
from typing import (
    Type,
    Union,
    TypeVar,
    ParamSpec,
    Callable,
    cast,
)
import random

P = ParamSpec("P")
T = TypeVar("T")


def exponential_backoff_agentonly(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Union[Type[Exception], tuple[Type[Exception], ...]] = Exception,
    jitter: bool = True,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    A decorator that implements exponential backoff for retrying failed coroutines.

    Args:
        max_retries (int): Maximum number of retry attempts
        base_delay (float): Initial delay between retries in seconds
        max_delay (float): Maximum delay between retries in seconds
        exceptions (Exception or tuple): Exception(s) to catch and retry on
        jitter (bool): Whether to add random jitter to delay times

    Returns:
        Callable: Decorated coroutine with retry logic
    """

    def decorator(
        func: Callable[P, T],
    ) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            retry_count = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    print(f"caught exception: {e}")
                    retry_count += 1
                    if retry_count > max_retries:
                        raise e

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (2 ** (retry_count - 1)), max_delay)

                    # Add jitter if enabled
                    if jitter:
                        delay *= 0.5 + random.random()

                    # Log retry attempt
                    print(
                        f"{func.__name__} - retry {retry_count}/{max_retries} after {delay:.2f}s delay"
                    )

                    # Wait before retrying
                    time.sleep(delay)

        return wrapper

    return cast(
        "Callable[[Callable[P, T]], Callable[P, T]]",
        decorator,
    )
