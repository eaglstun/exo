"""The node's side of a remote stage: a module that stands in for a range of layers, and a
cache whose contents live on the stage.

exo and mlx-lm reach a cache only through `offset`, `trim`, `deepcopy`, `nbytes`, `state` and
`make_mask`, so `RemoteKVCache` turns each of those into an operation on the stage, and the
rest of exo (prefix cache, generators, sampling) runs unchanged.
"""

import socket
import threading
from collections import deque
from collections.abc import Callable
from typing import Final, final

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache, create_attention_mask

from exo.worker.engines.mlx.remote_stage.protocol import (
    PROTOCOL_VERSION,
    Frame,
    array_to_bytes,
    bytes_to_array,
    dtype_name,
    receive_frame,
    send_frame,
)
from exo.worker.engines.mlx.types import Model
from exo.worker.engines.mlx.vision import get_inner_model

CONNECT_TIMEOUT_SECONDS: Final = 10.0


class RemoteStageError(RuntimeError):
    pass


@final
class RemoteStageLink:
    """One connection to a stage. Frees requested by garbage collection wait for the next
    operation, so they never write to the socket from inside another write."""

    def __init__(self, host: str, port: int):
        self._socket = socket.create_connection(
            (host, port), timeout=CONNECT_TIMEOUT_SECONDS
        )
        self._socket.settimeout(None)
        self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._lock = threading.Lock()
        self._next_handle = 1
        # Appended from __del__ on any thread; deque append and popleft need no lock
        self._unfreed: deque[int] = deque()
        # The length of every forward, in order: what a test reads to see prefix-cache hits
        self.forward_lengths: list[int] = []

        welcome = self._request(
            Frame(operation="hello", protocol_version=PROTOCOL_VERSION)
        )[0]
        if welcome.protocol_version != PROTOCOL_VERSION:
            raise RemoteStageError(
                f"the stage speaks protocol {welcome.protocol_version}, not {PROTOCOL_VERSION}"
            )
        self.layer_count: Final = welcome.layer_count
        self.kv_bytes_per_token: Final = welcome.kv_bytes_per_token

    def _send(self, frame: Frame, payload: bytes = b"") -> None:
        while self._unfreed:
            send_frame(
                self._socket, Frame(operation="free", handle=self._unfreed.popleft())
            )
        send_frame(self._socket, frame, payload)

    def _request(self, frame: Frame, payload: bytes = b"") -> tuple[Frame, bytes]:
        with self._lock:
            self._send(frame, payload)
            answer, answer_payload = receive_frame(self._socket)
        if answer.operation == "error":
            raise RemoteStageError(answer.message)
        return answer, answer_payload

    def _post(self, frame: Frame) -> None:
        with self._lock:
            self._send(frame)

    def new_cache(self) -> int:
        with self._lock:
            handle = self._next_handle
            self._next_handle += 1
            self._send(Frame(operation="new", handle=handle))
        return handle

    def copy_cache(self, source: int) -> int:
        with self._lock:
            handle = self._next_handle
            self._next_handle += 1
            self._send(Frame(operation="copy", handle=handle, source=source))
        return handle

    def truncate(self, handle: int, offset: int) -> None:
        self._post(Frame(operation="truncate", handle=handle, offset=offset))

    def release(self, handle: int) -> None:
        # Called from __del__: no socket write here
        self._unfreed.append(handle)

    def forward(self, handle: int, hidden: mx.array) -> mx.array:
        mx.eval(hidden)
        payload = array_to_bytes(hidden)
        answer, answer_payload = self._request(
            Frame(
                operation="forward",
                handle=handle,
                shape=list(hidden.shape),
                dtype=dtype_name(hidden.dtype),
                payload_bytes=len(payload),
            ),
            payload,
        )
        self.forward_lengths.append(hidden.shape[1])
        return bytes_to_array(answer_payload, answer.shape, answer.dtype)

    def live_caches(self) -> int:
        return self._request(Frame(operation="stats"))[0].live_caches

    def close(self) -> None:
        self._socket.close()


@final
class RemoteKVCache:
    """The caches of a remote stage's layers, held on the stage and named by a handle.

    Duck-typed like mlx-lm's caches, through the members exo and mlx-lm use: it isn't one of
    exo's SnapshotCaches (no prefix_cache_snapshot), so the prefix cache trims it."""

    def __init__(
        self, link: RemoteStageLink, handle: int | None = None, offset: int = 0
    ):
        self._link = link
        self.handle = link.new_cache() if handle is None else handle
        self._offset = offset

    @property
    def offset(self) -> int:
        return self._offset

    @offset.setter
    def offset(self, value: int) -> None:
        # exo sets the offset after a trim (cache.py, KVPrefixCache.get_kv_cache): a shorter
        # offset drops the tokens past it, like trim
        if value > self._offset:
            raise RemoteStageError(
                f"can't grow a remote cache from {self._offset} to {value}"
            )
        if value < self._offset:
            self._link.truncate(self.handle, value)
            self._offset = value

    def advance(self, tokens: int) -> None:
        """The stage appended `tokens` in a forward"""
        self._offset += tokens

    @property
    def state(self) -> list[mx.array]:
        # mlx-lm evaluates every cache's state between prefill chunks: nothing to do here
        return []

    def size(self) -> int:
        return self._offset

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self._offset, n)
        self.offset = self._offset - n
        return n

    def empty(self) -> bool:
        return self._offset == 0

    @property
    def nbytes(self) -> int:
        return self._offset * self._link.kv_bytes_per_token

    def make_mask(
        self, n: int, return_array: bool = False, window_size: int | None = None
    ) -> mx.array | str | None:
        return create_attention_mask(n, self._offset, return_array, window_size)

    def __deepcopy__(self, memo: dict[int, object]) -> "RemoteKVCache":
        return RemoteKVCache(
            self._link, self._link.copy_cache(self.handle), self._offset
        )

    def __del__(self) -> None:
        self._link.release(self.handle)


@final
class RemoteStage(nn.Module):
    """Stands in for layers [start_layer, end_layer) of a model, which run on the stage."""

    def __init__(self, link: RemoteStageLink, start_layer: int, end_layer: int):
        super().__init__()
        self._link = link
        self.start_layer = start_layer
        self.end_layer = end_layer
        # mlx-lm's layer loops pick each layer's mask by this; the stage masks causally only
        self.use_sliding = False

    def __call__(
        self, x: mx.array, mask: object = None, cache: RemoteKVCache | None = None
    ) -> mx.array:
        if cache is None:
            raise RemoteStageError("a remote stage runs only with its RemoteKVCache")
        # The stage masks causally whenever more than one token comes, which is what mlx-lm
        # passes these models (`create_attention_mask` gives "causal" or None)
        output = self._link.forward(cache.handle, x)
        cache.advance(x.shape[1])
        return output


def install_remote_stage(
    model: Model, link: RemoteStageLink, start_layer: int, end_layer: int
) -> None:
    """Replace layers [start_layer, end_layer) of `model` with the stage, and give the model a
    `make_cache` that hands that position a RemoteKVCache (exo's make_kv_cache prefers a
    model's own make_cache)."""
    inner = get_inner_model(model)  # pyright: ignore[reportAny]
    layers: list[nn.Module] = list(inner.layers)  # pyright: ignore[reportAny]
    if end_layer - start_layer != link.layer_count:
        raise RemoteStageError(
            f"the stage runs {link.layer_count} layers, not {end_layer - start_layer}"
        )
    if any(
        getattr(layer, "use_sliding", False) for layer in layers[start_layer:end_layer]
    ):
        raise RemoteStageError(
            "a remote stage can't run sliding-window attention layers"
        )
    stage = RemoteStage(link, start_layer, end_layer)
    inner.layers = [*layers[:start_layer], stage, *layers[end_layer:]]

    def make_cache() -> list[KVCache | RemoteKVCache]:
        return [
            RemoteKVCache(link) if layer is stage else KVCache()
            for layer in inner.layers  # pyright: ignore[reportAny]
        ]

    make: Callable[[], list[KVCache | RemoteKVCache]] = make_cache
    setattr(model, "make_cache", make)  # noqa: B010
