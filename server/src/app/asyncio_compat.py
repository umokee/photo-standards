import asyncio
import sys


def configure_asyncio_policy() -> None:
    if sys.platform != "win32":
        return

    policy = asyncio.get_event_loop_policy()
    if isinstance(policy, asyncio.WindowsSelectorEventLoopPolicy):
        return

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


__all__ = ["configure_asyncio_policy"]
