"""A proxy node is in the cluster only while its stage's device is there.

A stage goes away when its app quits or crashes, when the cable comes out, and when iPadOS
suspends the app: the iPad locks, or another app comes to the front. Then the proxy node's
runner can't serve its layers, and a runner that keeps failing to start puts its model on hold
until the user resumes it. A node that leaves is handled better: its models end at once, and are
placed again when it comes back (or elsewhere, if it doesn't). So a proxy node whose stage stops
answering leaves the cluster, waits for the stage, and starts over.
"""

import time
from collections.abc import Callable
from typing import Final

import anyio
from anyio import to_thread
from loguru import logger

# How long a stage may not answer before its proxy node leaves: longer than a stall of the app
# while it loads layers, short enough that a model isn't held up for long
STAGE_LOST_SECONDS: Final = 30.0
STAGE_POLL_SECONDS: Final = 3.0


def stage_answers(host: str, port: int) -> bool:
    """Whether the stage says hello now"""
    # Imported here: only a proxy node, which runs MLX anyway, needs the stage's protocol
    from exo.worker.engines.mlx.remote_stage.client import (
        RemoteStageError,
        greet_stage,
    )

    try:
        greet_stage(host, port)
    except (OSError, RemoteStageError):
        return False
    return True


def wait_for_stage(host: str, port: int) -> None:
    """Block until the stage answers"""
    if stage_answers(host, port):
        return
    logger.info(
        f"Waiting for the remote stage at {host}:{port}: is its app open, in front, and "
        f"the device unlocked?"
    )
    while not stage_answers(host, port):
        time.sleep(STAGE_POLL_SECONDS)
    logger.info(f"The remote stage at {host}:{port} is back")


async def watch_stage(
    host: str,
    port: int,
    on_lost: Callable[[], None],
    *,
    lost_after: float = STAGE_LOST_SECONDS,
    poll_interval: float = STAGE_POLL_SECONDS,
    answers: Callable[[str, int], bool] = stage_answers,
) -> None:
    """Call `on_lost` once the stage hasn't answered for `lost_after` seconds"""
    last_answer = anyio.current_time()
    while True:
        if await to_thread.run_sync(answers, host, port):
            last_answer = anyio.current_time()
        elif anyio.current_time() - last_answer >= lost_after:
            logger.warning(
                f"The remote stage at {host}:{port} hasn't answered for "
                f"{lost_after:.0f}s: this proxy node leaves the cluster until it is back"
            )
            on_lost()
            return
        await anyio.sleep(poll_interval)
