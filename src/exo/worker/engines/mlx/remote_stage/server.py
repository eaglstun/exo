"""A remote stage in Python: the reference the Swift stage on an iPad follows.

    python -m exo.worker.engines.mlx.remote_stage.server [--host 127.0.0.1] [--port 5560]

It starts with no layers: a node sends them in `load` frames. Serves each connection on its
own thread, with its own layers and caches. Each cache handle is a list of mlx-lm KVCaches,
one per layer.
"""

import argparse
import json
import platform
import socket
import tempfile
import threading
from copy import deepcopy
from pathlib import Path
from typing import Final, cast, final

import mlx.core as mx
import mlx.nn as nn
import psutil
from loguru import logger
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import BatchKVCache, KVCache
from mlx_lm.models.llama import ModelArgs, TransformerBlock

from exo.worker.engines.mlx.remote_stage.protocol import (
    PROTOCOL_VERSION,
    Frame,
    array_to_bytes,
    bytes_to_array,
    receive_frame,
    send_frame,
)

DEFAULT_PORT: Final = 5560


@final
class _Blocks(nn.Module):
    def __init__(self, blocks: list[TransformerBlock]):
        super().__init__()
        self.layers = blocks


# One layer's cache of one sequence, or of a batch
StageCache = KVCache | BatchKVCache


@final
class StageLayers:
    """The layers a node loaded, built as mlx-lm's own Llama blocks."""

    def __init__(self, config: dict[str, object], weights: dict[str, mx.array]):
        weights = dict(weights)
        # mlx-lm's blocks make their RoPE frequencies from the config
        weights.pop("rope_freqs", None)
        layer_count = 1 + max(int(name.split(".")[1]) for name in weights)
        blocks = _Blocks(
            [TransformerBlock(ModelArgs.from_dict(config)) for _ in range(layer_count)]  # pyright: ignore[reportUnknownMemberType]
        )
        quantization = cast(dict[str, int] | None, config.get("quantization"))
        if quantization is not None:
            nn.quantize(
                blocks,
                group_size=quantization["group_size"],
                bits=quantization["bits"],
                class_predicate=lambda path, _: f"{path}.scales" in weights,
            )
        blocks.load_weights(list(weights.items()), strict=True)
        mx.eval(blocks.parameters())
        self.layers: Final = blocks.layers
        heads = cast(int, config["num_attention_heads"])
        kv_heads = cast(int, config.get("num_key_value_heads", heads))
        head_dim = cast(
            int, config.get("head_dim") or cast(int, config["hidden_size"]) // heads
        )
        # Keys and values, per layer, kept in the activations' dtype
        dtype = weights["layers.0.input_layernorm.weight"].dtype
        self.kv_bytes_per_token: Final = (
            layer_count * 2 * kv_heads * head_dim * dtype.size
        )

    def forward(self, hidden: mx.array, caches: list[StageCache]) -> mx.array:
        # As mlx-lm's Llama makes it: "causal", None, or for a batch, a mask of its padding
        mask = cast(mx.array | None, create_attention_mask(hidden, caches[0]))
        for layer, cache in zip(self.layers, caches, strict=True):
            hidden = layer(hidden, mask, cache)
        return hidden


def _load_weights(payload: bytes) -> dict[str, mx.array]:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "stage.safetensors"
        path.write_bytes(payload)
        return cast(dict[str, mx.array], mx.load(str(path)))


def _welcome() -> Frame:
    memory = psutil.virtual_memory()
    return Frame(
        operation="welcome",
        protocol_version=PROTOCOL_VERSION,
        device_model=f"Python stage on {platform.node()}",
        memory_total=memory.total,
        memory_available=memory.available,
    )


def _batch(caches: dict[int, list[StageCache]], handle: int) -> list[BatchKVCache]:
    batch = caches[handle]
    assert all(isinstance(cache, BatchKVCache) for cache in batch)
    return cast(list[BatchKVCache], batch)


def serve_connection(connection: socket.socket) -> None:
    stage: StageLayers | None = None
    caches: dict[int, list[StageCache]] = {}
    # The layers a load has sent so far, and how many it will send
    arriving: dict[str, mx.array] = {}
    expected_layers = 0

    def error(message: str) -> None:
        send_frame(connection, Frame(operation="error", message=message))

    while True:
        try:
            frame, payload = receive_frame(connection)
        except ConnectionError:
            logger.info(f"node disconnected, dropping {len(caches)} caches")
            return
        match frame.operation:
            case "hello":
                send_frame(connection, _welcome())
            case "load":
                if frame.offset == 0:
                    caches.clear()
                    arriving.clear()
                    stage = None
                    expected_layers = frame.layer_count
                weights = _load_weights(payload)
                mx.eval(list(weights.values()))
                arriving.update(weights)
                received = len(
                    {
                        name.split(".")[1]
                        for name in arriving
                        if name.startswith("layers.")
                    }
                )
                if received == expected_layers:
                    stage = StageLayers(
                        cast(dict[str, object], json.loads(frame.config_json)), arriving
                    )
                    arriving = {}
                    logger.info(f"loaded {len(stage.layers)} layers")
                send_frame(
                    connection,
                    Frame(
                        operation="loaded",
                        layer_count=received,
                        kv_bytes_per_token=stage.kv_bytes_per_token if stage else 0,
                    ),
                )
            case "new":
                if stage is not None:
                    caches[frame.handle] = [KVCache() for _ in stage.layers]
            case "copy":
                caches[frame.handle] = deepcopy(caches[frame.source])
            case "free":
                del caches[frame.handle]
            case "truncate":
                for cache in caches[frame.handle]:
                    assert isinstance(cache, KVCache)
                    cache.offset = min(cache.offset, frame.offset)
            case "merge":
                sequences = [caches[source] for source in frame.sources]
                caches[frame.handle] = [
                    BatchKVCache.merge(
                        [cast(KVCache, sequence[layer]) for sequence in sequences]
                    )
                    for layer in range(len(sequences[0]))
                ]
            case "extend":
                for cache, other in zip(
                    _batch(caches, frame.handle),
                    _batch(caches, frame.source),
                    strict=True,
                ):
                    cache.extend(other)
            case "filter":
                for cache in _batch(caches, frame.handle):
                    cache.filter(frame.rows)
            case "extract":
                caches[frame.handle] = [
                    cache.extract(frame.rows[0])
                    for cache in _batch(caches, frame.source)
                ]
            case "prepare":
                for cache in _batch(caches, frame.handle):
                    cache.prepare(
                        left_padding=frame.left_padding or None,
                        right_padding=frame.right_padding or None,
                    )
            case "finalize":
                for cache in _batch(caches, frame.handle):
                    cache.finalize()
            case "trim":
                for cache in caches[frame.handle]:
                    cache.trim(frame.offset)
            case "stats":
                welcome = _welcome()
                send_frame(
                    connection,
                    Frame(
                        operation="counted",
                        live_caches=len(caches),
                        memory_total=welcome.memory_total,
                        memory_available=welcome.memory_available,
                    ),
                )
            case "forward":
                if stage is None or frame.handle not in caches:
                    error(f"no layers loaded or no cache {frame.handle}")
                    continue
                hidden = bytes_to_array(payload, frame.shape, frame.dtype)
                output = stage.forward(hidden, caches[frame.handle]).astype(
                    hidden.dtype
                )
                mx.eval(output)
                data = array_to_bytes(output)
                send_frame(
                    connection,
                    Frame(
                        operation="hidden",
                        handle=frame.handle,
                        shape=list(output.shape),
                        dtype=frame.dtype,
                        payload_bytes=len(data),
                    ),
                    data,
                )
            case "welcome" | "loaded" | "hidden" | "counted" | "error":
                error(f"a node doesn't send {frame.operation}")
                return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    arguments = parser.parse_args()
    host = cast(str, arguments.host)
    port = cast(int, arguments.port)

    listener = socket.create_server((host, port))
    logger.info(f"stage listening on {host}:{port}")
    while True:
        connection, address = cast(tuple[socket.socket, object], listener.accept())
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        logger.info(f"node connected from {address}")
        threading.Thread(target=_serve, args=(connection,), daemon=True).start()


def _serve(connection: socket.socket) -> None:
    with connection:
        serve_connection(connection)


if __name__ == "__main__":
    main()
