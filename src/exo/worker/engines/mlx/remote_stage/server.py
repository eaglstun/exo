"""A remote stage in Python: the reference the Swift stage on an iPad follows.

    python -m exo.worker.engines.mlx.remote_stage.server MODEL_PATH START END [--port 5560]

Serves one node at a time. Each cache handle is a list of mlx-lm KVCaches, one per layer.
"""

import argparse
import socket
from copy import deepcopy
from pathlib import Path
from typing import Final, cast, final

import mlx.core as mx
import mlx.nn as nn
from loguru import logger
from mlx.utils import tree_flatten
from mlx_lm.models.cache import KVCache
from mlx_lm.utils import load_model

from exo.worker.engines.mlx.remote_stage.protocol import (
    PROTOCOL_VERSION,
    Frame,
    array_to_bytes,
    bytes_to_array,
    receive_frame,
    send_frame,
)
from exo.worker.engines.mlx.vision import get_inner_model

DEFAULT_PORT: Final = 5560


@final
class StageLayers:
    def __init__(self, model_path: Path, start_layer: int, end_layer: int):
        model, config = load_model(model_path, lazy=True)
        layers: list[nn.Module] = list(get_inner_model(model).layers)  # pyright: ignore[reportAny]
        self.layers: Final = layers[start_layer:end_layer]
        mx.eval([layer.parameters() for layer in self.layers])
        parameters = dict(
            cast(list[tuple[str, mx.array]], tree_flatten(self.layers[0].parameters()))
        )
        # The activations' dtype: what the keys and values are kept in
        dtype = parameters["input_layernorm.weight"].dtype
        heads = int(cast(int, config["num_attention_heads"]))
        kv_heads = int(cast(int, config.get("num_key_value_heads", heads)))
        head_dim = int(
            cast(
                int,
                config.get("head_dim")
                or int(cast(int, config["hidden_size"])) // heads,
            )
        )
        self.kv_bytes_per_token: Final = (
            len(self.layers) * 2 * kv_heads * head_dim * dtype.size
        )

    def forward(self, hidden: mx.array, caches: list[KVCache]) -> mx.array:
        mask = "causal" if hidden.shape[1] > 1 else None
        for layer, cache in zip(self.layers, caches, strict=True):
            hidden = layer(hidden, mask, cache)
        return hidden


def serve_connection(connection: socket.socket, stage: StageLayers) -> None:
    caches: dict[int, list[KVCache]] = {}
    while True:
        try:
            frame, payload = receive_frame(connection)
        except ConnectionError:
            logger.info(f"node disconnected, dropping {len(caches)} caches")
            return
        match frame.operation:
            case "hello":
                send_frame(
                    connection,
                    Frame(
                        operation="welcome",
                        protocol_version=PROTOCOL_VERSION,
                        layer_count=len(stage.layers),
                        kv_bytes_per_token=stage.kv_bytes_per_token,
                    ),
                )
            case "new":
                caches[frame.handle] = [KVCache() for _ in stage.layers]
            case "copy":
                caches[frame.handle] = deepcopy(caches[frame.source])
            case "free":
                del caches[frame.handle]
            case "truncate":
                for cache in caches[frame.handle]:
                    cache.offset = min(cache.offset, frame.offset)
            case "stats":
                send_frame(
                    connection, Frame(operation="counted", live_caches=len(caches))
                )
            case "forward":
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
            case "welcome" | "hidden" | "counted" | "error":
                send_frame(
                    connection,
                    Frame(
                        operation="error",
                        message=f"a node doesn't send {frame.operation}",
                    ),
                )
                return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("start_layer", type=int)
    parser.add_argument("end_layer", type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    arguments = parser.parse_args()
    model_path = cast(Path, arguments.model_path)
    host = cast(str, arguments.host)
    port = cast(int, arguments.port)

    stage = StageLayers(
        model_path, cast(int, arguments.start_layer), cast(int, arguments.end_layer)
    )
    listener = socket.create_server((host, port))
    logger.info(f"stage ready: {len(stage.layers)} layers on {host}:{port}")
    while True:
        connection, address = cast(tuple[socket.socket, object], listener.accept())
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        logger.info(f"node connected from {address}")
        with connection:
            serve_connection(connection, stage)


if __name__ == "__main__":
    main()
