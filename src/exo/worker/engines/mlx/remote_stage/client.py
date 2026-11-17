"""The node's side of a remote stage: a module that stands in for a range of layers, and a
cache whose contents live on the stage.

exo and mlx-lm reach a cache only through `offset`, `trim`, `deepcopy`, `nbytes`, `state` and
`make_mask`, so `RemoteKVCache` turns each of those into an operation on the stage, and the
rest of exo (prefix cache, generators, sampling) runs unchanged.
"""

import json
import os
import socket
import tempfile
import threading
from collections import deque
from collections.abc import Callable, Generator, Iterator, Sequence
from pathlib import Path
from typing import Final, cast, final

import mlx.core as mx
import mlx.nn as nn
from loguru import logger
from mlx.utils import tree_flatten
from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.cache import KVCache, create_attention_mask

from exo.worker.engines.mlx.cache import add_memory_pressure_source
from exo.worker.engines.mlx.remote_stage.protocol import (
    PROTOCOL_VERSION,
    Frame,
    array_to_bytes,
    bytes_to_array,
    dtype_name,
    receive_frame,
    send_frame,
)
from exo.worker.engines.mlx.remote_stage.settings import (
    REMOTE_STAGE_ENV,
    REMOTE_STAGE_LAYERS_ENV,
    configured_stage_address,
    is_stage_proxy,
)
from exo.worker.engines.mlx.sharding.blocks import LayerCallable
from exo.worker.engines.mlx.sharding.pipeline import get_layers
from exo.worker.engines.mlx.types import Model
from exo.worker.engines.mlx.vision import get_inner_model

CONNECT_TIMEOUT_SECONDS: Final = 10.0

# The model families whose layers a stage can run: their blocks are all the same
STAGE_MODEL_TYPES: Final = frozenset({"llama"})


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
        # What the stage's device is, and its memory when it said hello
        self.welcome: Final = welcome
        # Set by load
        self.layer_count = 0
        self.kv_bytes_per_token = 0

    def load(
        self, layers: list[nn.Module], config: dict[str, object], rope_freqs: mx.array
    ) -> None:
        """Send the stage these layers' weights to run from now on."""
        for _ in self.load_layers(layers, config, rope_freqs):
            pass

    def load_layers(
        self, layers: list[nn.Module], config: dict[str, object], rope_freqs: mx.array
    ) -> Iterator[int]:
        """Send the stage these layers' weights, one layer per frame, yielding how many it
        has after each. Only one layer at a time is read into memory here: from a lazily
        loaded model, the rest never are."""
        config_json = json.dumps(config)
        for index, layer in enumerate(layers):
            weights: dict[str, mx.array] = {
                f"layers.{index}.{name}": array
                for name, array in cast(
                    list[tuple[str, mx.array]], tree_flatten(layer.parameters())
                )
            }
            if index == 0:
                weights["rope_freqs"] = rope_freqs
            payload = _safetensors_bytes(weights)
            del weights
            loaded = self._request(
                Frame(
                    operation="load",
                    offset=index,
                    layer_count=len(layers),
                    config_json=config_json,
                    payload_bytes=len(payload),
                ),
                payload,
            )[0]
            if loaded.layer_count != index + 1:
                raise RemoteStageError(
                    f"the stage has {loaded.layer_count} layers, not {index + 1}"
                )
            yield loaded.layer_count
            if loaded.layer_count == len(layers):
                self.layer_count = loaded.layer_count
                self.kv_bytes_per_token = loaded.kv_bytes_per_token

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
        return self._create(Frame(operation="copy", source=source))

    def _create(self, frame: Frame) -> int:
        """Send an operation that makes a cache, under a new handle"""
        with self._lock:
            handle = self._next_handle
            self._next_handle += 1
            self._send(frame.model_copy(update={"handle": handle}))
        return handle

    def merge(self, sources: list[int]) -> int:
        return self._create(Frame(operation="merge", sources=sources))

    def extract(self, batch: int, row: int) -> int:
        return self._create(Frame(operation="extract", source=batch, rows=[row]))

    def post(self, frame: Frame) -> None:
        """Send an operation the stage applies without answering"""
        self._post(frame)

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

    def memory_used_fraction(self) -> float:
        """How full the stage's device is, 0 to 1, as its stats say now"""
        counted = self._request(Frame(operation="stats"))[0]
        if counted.memory_total <= 0:
            return 0.0
        return 1 - counted.memory_available / counted.memory_total

    def close(self) -> None:
        self._socket.close()


def _safetensors_bytes(weights: dict[str, mx.array]) -> bytes:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "layer.safetensors"
        mx.save_safetensors(str(path), weights)  # pyright: ignore[reportUnknownMemberType]
        return path.read_bytes()


def greet_stage(host: str, port: int) -> Frame:
    """The stage's welcome: its device and memory now. A connection of its own: the stage
    serves each connection apart, so this can ask while a runner's link is busy."""
    with socket.create_connection(
        (host, port), timeout=CONNECT_TIMEOUT_SECONDS
    ) as connection:
        send_frame(
            connection, Frame(operation="hello", protocol_version=PROTOCOL_VERSION)
        )
        welcome, _ = receive_frame(connection)
    if welcome.protocol_version != PROTOCOL_VERSION:
        raise RemoteStageError(
            f"the stage speaks protocol {welcome.protocol_version}, not {PROTOCOL_VERSION}"
        )
    return welcome


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

    def merge(self, caches: list["RemoteKVCache"]) -> "RemoteBatchKVCache":
        """These sequences' caches as one batch (mlx-lm calls KVCache.merge on its first)"""
        return RemoteBatchKVCache.merged(self._link, caches)

    def __del__(self) -> None:
        self._link.release(self.handle)


@final
class RemoteBatchKVCache:
    """A batch of sequences' caches on the stage: mlx-lm's BatchKVCache, whose keys and
    values stay there. Its bookkeeping (left padding, per-row offsets, length) is kept here
    exactly as BatchKVCache keeps it, since mlx-lm and exo read it, and every change to it
    is sent to the stage, which makes the same change to its BatchKVCaches."""

    def __init__(
        self,
        link: RemoteStageLink,
        handle: int,
        left_padding: mx.array,
        offset: mx.array,
        length: int,
        holds_tokens: bool,
    ):
        self._link = link
        self.handle = handle
        self.left_padding: mx.array = left_padding
        self.offset: mx.array = offset
        self._idx: int = length
        self._right_padding: mx.array | None = None
        # Whether the stage's BatchKVCaches have keys yet (BatchKVCache.empty)
        self._holds_tokens = holds_tokens

    @classmethod
    def merged(
        cls, link: RemoteStageLink, caches: list["RemoteKVCache"]
    ) -> "RemoteBatchKVCache":
        lengths = [cache.offset for cache in caches]
        longest = max(lengths)
        padding = [longest - length for length in lengths]
        handle = link.merge([cache.handle for cache in caches])
        return cls(
            link,
            handle,
            mx.array(padding),
            mx.array([longest - pad for pad in padding]),
            longest,
            holds_tokens=longest > 0,
        )

    def advance(self, tokens: int) -> None:
        """The stage appended `tokens` to every row in a forward"""
        self.offset = self.offset + tokens
        self._idx += tokens
        self._holds_tokens = True

    @property
    def state(self) -> list[mx.array]:
        return []

    def prepare(
        self,
        *,
        left_padding: list[int] | None = None,
        lengths: list[int] | None = None,
        right_padding: list[int] | None = None,
    ) -> None:
        del lengths  # BatchKVCache doesn't use them either
        if left_padding is not None:
            if self._holds_tokens:
                raise RemoteStageError(
                    "Left padding can only be added to an empty BatchKVCache"
                )
            added = mx.array(left_padding)
            self.left_padding = self.left_padding + added
            self.offset = self.offset - added
        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)
        self._link.post(
            Frame(
                operation="prepare",
                handle=self.handle,
                left_padding=left_padding or [],
                right_padding=right_padding or [],
            )
        )

    def finalize(self) -> None:
        if self._right_padding is not None:
            self.offset = self.offset - self._right_padding
            self.left_padding = self.left_padding + self._right_padding
            self._right_padding = None
            self._link.post(Frame(operation="finalize", handle=self.handle))

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self._idx, n)
        self._idx -= n
        self.offset = self.offset - n
        self._link.post(Frame(operation="trim", handle=self.handle, offset=n))
        return n

    def make_mask(
        self, n: int, return_array: bool = False, **kwargs: object
    ) -> mx.array:
        del return_array
        return create_causal_mask(
            n,
            offset=self._idx,
            left_padding=self.left_padding,
            **kwargs,  # pyright: ignore[reportArgumentType]
        )

    def filter(self, batch_indices: list[int] | mx.array) -> None:
        rows = (
            cast(list[int], batch_indices.tolist())
            if isinstance(batch_indices, mx.array)
            else list(batch_indices)
        )
        self.offset = self.offset[mx.array(rows)] if rows else mx.array([], mx.int32)
        self.left_padding = (
            self.left_padding[mx.array(rows)] if rows else mx.array([], mx.int32)
        )
        self._link.post(Frame(operation="filter", handle=self.handle, rows=rows))
        if not rows:
            return
        # BatchKVCache drops the padding every row has
        least_padding = min(cast(list[int], self.left_padding.tolist()))
        if least_padding > 0:
            self._idx -= least_padding
            self.left_padding = self.left_padding - least_padding

    def extend(self, other: "RemoteBatchKVCache") -> None:
        if not self._holds_tokens and not other._holds_tokens:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
        else:
            length = max(self._idx, other._idx)
            self.left_padding = mx.concatenate(
                [
                    self.left_padding + (length - self._idx),
                    other.left_padding + (length - other._idx),
                ]
            )
            self.offset = mx.concatenate([self.offset, other.offset])
            self._idx = length
            self._holds_tokens = True
        self._link.post(
            Frame(operation="extend", handle=self.handle, source=other.handle)
        )

    def extract(self, idx: int) -> "RemoteKVCache":
        padding = cast(list[int], self.left_padding.tolist())[idx]
        return RemoteKVCache(
            self._link, self._link.extract(self.handle, idx), self._idx - padding
        )

    def size(self) -> int:
        return self._idx

    def empty(self) -> bool:
        return not self._holds_tokens

    @property
    def nbytes(self) -> int:
        return self.offset.shape[0] * self._idx * self._link.kv_bytes_per_token

    def __deepcopy__(self, memo: dict[int, object]) -> "RemoteBatchKVCache":
        copied = RemoteBatchKVCache(
            self._link,
            self._link.copy_cache(self.handle),
            self.left_padding,
            self.offset,
            self._idx,
            self._holds_tokens,
        )
        copied._right_padding = self._right_padding
        return copied

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
        self,
        x: mx.array,
        mask: object = None,
        cache: RemoteKVCache | RemoteBatchKVCache | None = None,
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


def rope_frequencies(layer: nn.Module, config: dict[str, object]) -> mx.array:
    """What mx.fast.rope takes as `freqs` for this layer's attention: the model's own when it
    scales RoPE (llama3), else base ** (2i / dims), which is what `base` alone means."""
    rope = cast(object, getattr(cast(object, getattr(layer, "self_attn")), "rope"))  # noqa: B009
    own: object = getattr(rope, "_freqs", None)
    if isinstance(own, mx.array):
        return own.astype(mx.float32)
    heads = cast(int, config["num_attention_heads"])
    head_dim = cast(
        int, config.get("head_dim") or cast(int, config["hidden_size"]) // heads
    )
    base = float(cast(float, config.get("rope_theta", 10000.0)))
    return mx.power(
        mx.array(base), mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim
    )


def attach_remote_stage(
    model: Model,
    link: RemoteStageLink,
    start_layer: int,
    end_layer: int,
    config: dict[str, object],
) -> None:
    """Send layers [start_layer, end_layer) of this node's layers to the stage and run them
    there from now on."""
    if config.get("model_type") not in STAGE_MODEL_TYPES:
        raise RemoteStageError(
            f"a remote stage can't run {config.get('model_type')} layers yet"
        )
    layers = cast(list[nn.Module], list(get_layers(get_inner_model(model))))  # pyright: ignore[reportAny]
    link.load(
        layers[start_layer:end_layer],
        config,
        rope_frequencies(layers[start_layer], config),
    )
    install_remote_stage(model, link, start_layer, end_layer)


def _stage_config(model_path: Path) -> dict[str, object]:
    """The model's config, once checked that a stage can run its layers here"""
    config = cast(
        dict[str, object], json.loads((model_path / "config.json").read_text())
    )
    if config.get("model_type") not in STAGE_MODEL_TYPES:
        raise RemoteStageError(
            f"a remote stage can't run {config.get('model_type')} layers yet"
        )
    return config


def hand_layers_to_stage(
    model: nn.Module, given_layers: Sequence[LayerCallable], model_path: Path
) -> Generator[tuple[int, int], None, list[LayerCallable]]:
    """On a proxy node: send all of `layers` (this node's share of the model, not yet
    evaluated) to the stage, yielding (layers sent, of how many), and return what runs in
    their place: one RemoteStage. Gives `model` a make_cache with its RemoteKVCache."""
    stage_address = configured_stage_address()
    assert stage_address is not None
    config = _stage_config(model_path)
    layers = cast(list[nn.Module], list(given_layers))
    if any(getattr(layer, "use_sliding", False) for layer in layers):
        raise RemoteStageError(
            "a remote stage can't run sliding-window attention layers"
        )
    link = RemoteStageLink(*stage_address)
    add_memory_pressure_source(link.memory_used_fraction)
    rope = rope_frequencies(layers[0], config)
    for sent in link.load_layers(layers, config, rope):
        yield sent, len(layers)
    stage = RemoteStage(link, 0, len(layers))

    def make_cache() -> list[RemoteKVCache]:
        return [RemoteKVCache(link)]

    make: Callable[[], list[RemoteKVCache]] = make_cache
    setattr(model, "make_cache", make)  # noqa: B010
    logger.info(
        f"remote stage {stage_address[0]}:{stage_address[1]} "
        f"({link.welcome.device_model}) runs all {len(layers)} of this node's layers"
    )
    # Its call names mask and cache, which LayerCallable leaves open
    return [cast(LayerCallable, cast(object, stage))]


def attach_configured_remote_stage(model: Model, model_path: Path) -> None:
    """With EXO_REMOTE_STAGE set, hand part of this node's layers to that stage: by default the
    back half, short of the last layer, so a pipeline's first and last layer stay here. A
    proxy node has already handed over all of them (hand_layers_to_stage)."""
    address = os.environ.get(REMOTE_STAGE_ENV)
    if not address or is_stage_proxy():
        return
    config = _stage_config(model_path)
    layer_count = len(get_layers(get_inner_model(model)))  # pyright: ignore[reportAny]
    wanted = int(os.environ.get(REMOTE_STAGE_LAYERS_ENV) or layer_count // 2)
    count = max(0, min(wanted, layer_count - 2))
    if count == 0:
        logger.warning(f"{REMOTE_STAGE_ENV}: no layers to hand over of {layer_count}")
        return
    end_layer = layer_count - 1
    start_layer = end_layer - count
    host, _, port = address.rpartition(":")
    link = RemoteStageLink(host, int(port))
    add_memory_pressure_source(link.memory_used_fraction)
    attach_remote_stage(model, link, start_layer, end_layer, config)
    logger.info(
        f"remote stage {address} runs this node's layers {start_layer}..{end_layer - 1}"
    )
