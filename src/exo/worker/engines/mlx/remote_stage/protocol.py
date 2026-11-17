"""The frames between an exo node and a remote stage: a device that runs a range of a model's
layers for that node, and keeps those layers' KV caches.

Every frame is a little-endian u32 header length, a JSON header, then `payload_bytes` of raw
array data. A stage starts with no layers: the node sends them in `load` frames, a layer or a
few at a time, as safetensors bytes with the model's config, so neither side ever holds a
second copy of them all. `welcome` says what the stage's device is and how much memory it has,
which a proxy node reports to the cluster as its own. The node names each cache by a handle it picks itself, so only
`hello`, `load`, `forward` and `stats` wait for an answer: every other operation is sent and
the stage applies it in order.
"""

import socket
import struct
from typing import Final, Literal

import mlx.core as mx
import numpy as np
from pydantic import Field

from exo.utils.pydantic_ext import FrozenModel

PROTOCOL_VERSION: Final = 4

Operation = Literal[
    # node -> stage
    "hello",
    "load",
    "new",
    "forward",
    "truncate",
    "copy",
    "free",
    "stats",
    # Batches, as mlx-lm's BatchKVCache does them (see client.RemoteBatchKVCache)
    "merge",
    "extend",
    "filter",
    "extract",
    "prepare",
    "finalize",
    "trim",
    # stage -> node
    "welcome",
    "loaded",
    "hidden",
    "counted",
    "error",
]


class Frame(FrozenModel):
    operation: Operation
    # The cache this operation is for; for `copy`, `merge` and `extract`, the new cache
    handle: int = 0
    # copy and extract: the cache to copy from. extend: the batch to append
    source: int = 0
    # merge: the single-sequence caches to merge into a new batch, in order
    sources: list[int] = Field(default_factory=list)
    # filter: the rows to keep. extract: [the row to copy out]
    rows: list[int] = Field(default_factory=list)
    # prepare: padding to add to each row
    left_padding: list[int] = Field(default_factory=list)
    right_padding: list[int] = Field(default_factory=list)
    # truncate: the cache's new length in tokens. trim: how many tokens to drop from a batch's
    # end. load: the index, among the stage's layers,
    # of the first layer in this frame; 0 starts a new set of layers
    offset: int = 0
    # forward and hidden: the array that follows the header
    shape: list[int] = Field(default_factory=list)
    dtype: str = ""
    payload_bytes: int = 0
    # hello and welcome
    protocol_version: int = 0
    # welcome: the stage's device (its model identifier, e.g. iPad17,1). welcome and
    # counted: all its memory, and how much of that the stage itself can still allocate
    device_model: str = ""
    memory_total: int = 0
    memory_available: int = 0
    # welcome: the most memory the device's GPU should use (Metal's recommended working set),
    # or 0 if it didn't say
    gpu_working_set: int = 0
    # load: the model's config.json; its payload is these layers' weights (named layers.N..,
    # N counted from the stage's first layer) and, in the first frame, `rope_freqs`, as
    # safetensors
    config_json: str = ""
    # load: how many layers the stage will run. loaded: how many it has so far
    layer_count: int = 0
    # loaded, once it has them all: the KV cache's size per token
    kv_bytes_per_token: int = 0
    # counted: how many caches the stage holds
    live_caches: int = 0
    # error
    message: str = ""


_HEADER_LENGTH: Final = struct.Struct("<I")

_DTYPES: Final[dict[str, tuple[mx.Dtype, type[np.generic]]]] = {
    "float16": (mx.float16, np.float16),
    # numpy has no bfloat16: its bits travel as uint16
    "bfloat16": (mx.bfloat16, np.uint16),
    "float32": (mx.float32, np.float32),
}


def dtype_name(dtype: mx.Dtype) -> str:
    for name, (mlx_dtype, _) in _DTYPES.items():
        if mlx_dtype == dtype:
            return name
    raise ValueError(f"a remote stage can't carry {dtype}")


def array_to_bytes(array: mx.array) -> bytes:
    if array.dtype == mx.bfloat16:
        array = array.view(mx.uint16)
    return np.array(array, copy=False).tobytes()


def bytes_to_array(data: bytes, shape: list[int], dtype: str) -> mx.array:
    mlx_dtype, numpy_dtype = _DTYPES[dtype]
    array = mx.reshape(mx.array(np.frombuffer(data, dtype=numpy_dtype)), shape)
    return array.view(mlx_dtype) if mlx_dtype == mx.bfloat16 else array


def send_frame(connection: socket.socket, frame: Frame, payload: bytes = b"") -> None:
    assert frame.payload_bytes == len(payload)
    header = frame.model_dump_json().encode("utf-8")
    # One write per frame, so frames from different threads never interleave
    connection.sendall(_HEADER_LENGTH.pack(len(header)) + header + payload)


def _receive_exactly(connection: socket.socket, length: int) -> bytes:
    buffer = bytearray(length)
    view = memoryview(buffer)
    received = 0
    while received < length:
        count = connection.recv_into(view[received:], length - received)
        if count == 0:
            raise ConnectionError("the remote stage's connection closed")
        received += count
    return bytes(buffer)


def receive_frame(connection: socket.socket) -> tuple[Frame, bytes]:
    header_length: int = _HEADER_LENGTH.unpack(  # pyright: ignore[reportAny]
        _receive_exactly(connection, _HEADER_LENGTH.size)
    )[0]
    frame = Frame.model_validate_json(_receive_exactly(connection, header_length))
    payload = _receive_exactly(connection, frame.payload_bytes)
    return frame, payload
