"""Observability utilities for timing and logging."""

import functools
import logging
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)


def timer(node_name: str | None = None) -> Callable[[F], F]:
    """Decorator that logs elapsed time for a node/function to the console."""

    def decorator(func: F) -> F:
        name = node_name or getattr(func, "__name__", "unknown")

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                logger.info("[%s] elapsed: %.3f s", name, elapsed)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                logger.info("[%s] elapsed: %.3f s", name, elapsed)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorator
