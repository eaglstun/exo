"""A proxy node notices its stage going away: the link says so plainly, and the watcher has the
node leave once the stage has been silent long enough."""

import socket
import threading

import anyio
import pytest

from exo.worker.engines.mlx.remote_stage.client import RemoteStageLink, RemoteStageLost
from exo.worker.engines.mlx.remote_stage.presence import watch_stage
from exo.worker.engines.mlx.remote_stage.protocol import (
    PROTOCOL_VERSION,
    Frame,
    receive_frame,
    send_frame,
)


async def _watch(answers: list[bool], lost_after: float) -> bool:
    """Whether the watcher gave the stage up, given its answers to each poll in turn (the last
    one repeats)"""
    polls = iter(answers)
    last = answers[-1]
    lost: list[bool] = []

    def answer(_host: str, _port: int) -> bool:
        return next(polls, last)

    with anyio.move_on_after(1.0):
        await watch_stage(
            "stage",
            1,
            lambda: lost.append(True),
            lost_after=lost_after,
            poll_interval=0.01,
            answers=answer,
        )
    return bool(lost)


async def test_a_silent_stage_is_given_up():
    assert await _watch([True, False], lost_after=0.1)


async def test_a_stage_that_answers_again_is_kept():
    # Silent for a while, but never as long as lost_after
    assert not await _watch([False] * 3 + [True], lost_after=0.2)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])  # pyright: ignore[reportAny]


def test_an_absent_stage_is_lost_plainly():
    with pytest.raises(RemoteStageLost, match="can't reach the remote stage"):
        RemoteStageLink("127.0.0.1", _free_port())


def test_a_stage_that_quits_mid_request_is_lost_plainly():
    listener = socket.create_server(("127.0.0.1", 0))
    port = int(listener.getsockname()[1])  # pyright: ignore[reportAny]

    def stage() -> None:
        # Says hello, then quits as an app does: without answering the next request
        connection = listener.accept()[0]
        with connection:
            receive_frame(connection)
            send_frame(
                connection,
                Frame(operation="welcome", protocol_version=PROTOCOL_VERSION),
            )
            receive_frame(connection)

    thread = threading.Thread(target=stage)
    thread.start()
    link = RemoteStageLink("127.0.0.1", port)
    with pytest.raises(RemoteStageLost, match="went away"):
        link.live_caches()
    thread.join()
    listener.close()
