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
from mlx_lm.models import qwen3_5, qwen3_moe
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, BatchKVCache, KVCache
from mlx_lm.models.llama import ModelArgs as LlamaArgs
from mlx_lm.models.llama import TransformerBlock as LlamaBlock
from mlx_lm.models.qwen3 import ModelArgs as Qwen3Args
from mlx_lm.models.qwen3 import TransformerBlock as Qwen3Block

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
    def __init__(self, blocks: list[nn.Module]):
        super().__init__()
        self.layers = blocks


def _block(
    config: dict[str, object], weights: dict[str, mx.array], index: int
) -> nn.Module:
    """Layer `index` of those sent, as mlx-lm builds it for the config's model type"""
    match config.get("model_type"):
        case "llama":
            return LlamaBlock(LlamaArgs.from_dict(config))  # pyright: ignore[reportUnknownMemberType]
        case "qwen3":
            return Qwen3Block(Qwen3Args.from_dict(config))  # pyright: ignore[reportUnknownMemberType]
        case "qwen3_moe":
            # Whether a layer has experts follows from its index in the whole model, which
            # the stage doesn't know: its weights say
            has_experts = f"layers.{index}.mlp.gate.weight" in weights
            args = cast(
                qwen3_moe.ModelArgs,
                qwen3_moe.ModelArgs.from_dict(  # pyright: ignore[reportUnknownMemberType]
                    {
                        **config,
                        "mlp_only_layers": [] if has_experts else [0],
                        "decoder_sparse_step": 1,
                    }
                ),
            )
            return qwen3_moe.Qwen3MoeDecoderLayer(args, layer_idx=0)
        case "qwen3_5" | "qwen3_5_moe":
            # Experts (and a shared expert) when the config has them. Linear attention or full
            # attention follows from the layer's index in the whole model: its weights say. mlx-lm makes layer i linear unless (i + 1) is a
            # multiple of full_attention_interval
            args = qwen3_5.TextModelArgs.from_dict(config)
            is_linear = f"layers.{index}.linear_attn.A_log" in weights
            return qwen3_5.DecoderLayer(
                args, layer_idx=0 if is_linear else args.full_attention_interval - 1
            )
        case other:
            raise ValueError(f"this stage can't run {other} layers")


# One layer's cache of one sequence, or of a batch: keys and values, or a linear attention
# layer's recurrent state (which ArraysCache keeps for a batch as for one sequence)
StageCache = KVCache | BatchKVCache | ArraysCache


def _is_linear(layer: nn.Module) -> bool:
    return bool(getattr(layer, "is_linear", False))


@final
class StageLayers:
    """The layers a node loaded, built as mlx-lm's own blocks."""

    def __init__(self, config: dict[str, object], weights: dict[str, mx.array]):
        weights = dict(weights)
        # mlx-lm's blocks make their RoPE frequencies from the config
        weights.pop("rope_freqs", None)
        layer_count = 1 + max(int(name.split(".")[1]) for name in weights)
        blocks = _Blocks(
            [_block(config, weights, index) for index in range(layer_count)]
        )
        # The node says how it quantized each module (client.stage_quantization)
        quantization = cast(dict[str, object], config.get("quantization") or {})

        def quantized(path: str, _: nn.Module) -> bool | dict[str, int]:
            if f"{path}.scales" not in weights:
                return False
            own = quantization.get(path)
            return cast(dict[str, int], own) if isinstance(own, dict) else True

        if quantization:
            nn.quantize(
                blocks,
                group_size=cast(int, quantization.get("group_size", 64)),
                bits=cast(int, quantization.get("bits", 4)),
                class_predicate=quantized,
            )
        blocks.load_weights(list(weights.items()), strict=True)
        # As mlx-lm's load_model leaves a model: a module in training mode may compute
        # otherwise (Qwen3.5's linear attention runs its ops instead of its Metal kernel)
        blocks.eval()
        mx.eval(blocks.parameters())
        self.layers: Final = blocks.layers
        heads = cast(int, config["num_attention_heads"])
        kv_heads = cast(int, config.get("num_key_value_heads", heads))
        head_dim = cast(
            int, config.get("head_dim") or cast(int, config["hidden_size"]) // heads
        )
        # Keys and values, per attention layer, kept in the activations' dtype
        dtype = weights["layers.0.input_layernorm.weight"].dtype
        linear = [_is_linear(layer) for layer in self.layers]
        self.kv_bytes_per_token: Final = (
            linear.count(False) * 2 * kv_heads * head_dim * dtype.size
        )
        # A linear attention layer's recurrent state: the last (kernel - 1) inputs of its
        # convolution, in the activations' dtype, and its float32 state per value head
        conv_dim = 2 * cast(int, config.get("linear_num_key_heads", 0)) * cast(
            int, config.get("linear_key_head_dim", 0)
        ) + cast(int, config.get("linear_num_value_heads", 0)) * cast(
            int, config.get("linear_value_head_dim", 0)
        )
        conv_bytes = (
            (cast(int, config.get("linear_conv_kernel_dim", 1)) - 1)
            * conv_dim
            * dtype.size
        )
        state_bytes = (
            cast(int, config.get("linear_num_value_heads", 0))
            * cast(int, config.get("linear_value_head_dim", 0))
            * cast(int, config.get("linear_key_head_dim", 0))
            * mx.float32.size
        )
        self.state_bytes: Final = linear.count(True) * (conv_bytes + state_bytes)

    def make_caches(self) -> list[StageCache]:
        """A sequence's empty caches, one per layer, as mlx-lm's models make them"""
        return [
            ArraysCache(size=2) if _is_linear(layer) else KVCache()
            for layer in self.layers
        ]

    def forward(self, hidden: mx.array, caches: list[StageCache]) -> mx.array:
        # As mlx-lm's models make them, from the first cache of each kind: "causal", None, or
        # for a batch, a mask of its padding
        attention = next((c for c in caches if not isinstance(c, ArraysCache)), None)
        recurrent = next((c for c in caches if isinstance(c, ArraysCache)), None)
        attention_mask = (
            cast(mx.array | None, create_attention_mask(hidden, attention))
            if attention is not None
            else None
        )
        recurrent_mask = (
            create_ssm_mask(hidden, recurrent) if recurrent is not None else None
        )
        for layer, cache in zip(self.layers, caches, strict=True):
            mask = recurrent_mask if _is_linear(layer) else attention_mask
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


def _batch(
    caches: dict[int, list[StageCache]], handle: int
) -> list[BatchKVCache | ArraysCache]:
    batch = caches[handle]
    assert all(isinstance(cache, (BatchKVCache, ArraysCache)) for cache in batch)
    return cast(list[BatchKVCache | ArraysCache], batch)


def _recurrent_state(layer_caches: list[StageCache]) -> list[mx.array]:
    return [
        array
        for cache in layer_caches
        if isinstance(cache, ArraysCache)
        for array in cache.cache
        if array is not None
    ]


def _merge(layer_caches: list[StageCache]) -> BatchKVCache | ArraysCache:
    """One layer's caches of several sequences, as one batch"""
    if all(isinstance(cache, ArraysCache) for cache in layer_caches):
        return ArraysCache.merge(cast(list[ArraysCache], layer_caches))
    return BatchKVCache.merge(cast(list[KVCache], layer_caches))


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
                        state_bytes=stage.state_bytes if stage else 0,
                    ),
                )
            case "new":
                if stage is not None:
                    caches[frame.handle] = stage.make_caches()
            case "copy":
                caches[frame.handle] = deepcopy(caches[frame.source])
            case "free":
                # Possibly a cache from before the node loaded other layers, which dropped
                # them all
                caches.pop(frame.handle, None)
            case "truncate":
                # Only a stage without recurrent state is sent truncate: the node goes
                # back to a copy instead (RemoteSnapshotCache)
                for cache in caches[frame.handle]:
                    assert isinstance(cache, KVCache)
                    cache.offset = min(cache.offset, frame.offset)
            case "merge":
                sequences = [caches[source] for source in frame.sources]
                caches[frame.handle] = [
                    _merge([sequence[layer] for sequence in sequences])
                    for layer in range(len(sequences[0]))
                ]
            case "extend":
                for cache, other in zip(
                    _batch(caches, frame.handle),
                    _batch(caches, frame.source),
                    strict=True,
                ):
                    cache.extend(other)  # pyright: ignore[reportArgumentType]
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
                    if isinstance(cache, ArraysCache):
                        # Its prepare takes each row's length, which right padding
                        # would need: exo pads only on the left, where the state needs
                        # nothing
                        assert not any(frame.right_padding)
                        continue
                    cache.prepare(
                        left_padding=frame.left_padding or None,
                        right_padding=frame.right_padding or None,
                    )
            case "finalize":
                for cache in _batch(caches, frame.handle):
                    cache.finalize()
            case "trim":
                for cache in caches[frame.handle]:
                    assert not isinstance(cache, ArraysCache)
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
                # With the recurrent state, which nothing else this step reads: copies of
                # the cache (snapshots) hold evaluated arrays, not a graph that grows
                mx.eval(output, *_recurrent_state(caches[frame.handle]))
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
