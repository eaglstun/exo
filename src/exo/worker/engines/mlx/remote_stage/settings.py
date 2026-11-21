"""How a node is told about its remote stage: environment variables only, read here without
importing MLX (the info gatherer reads them in exo's main process)."""

import os
from typing import Final

# host:port of a stage this node hands part of its layers to
REMOTE_STAGE_ENV: Final = "EXO_REMOTE_STAGE"
# How many layers it takes (default: half of this node's)
REMOTE_STAGE_LAYERS_ENV: Final = "EXO_REMOTE_STAGE_LAYERS"
# 1: this node is a proxy for the stage's device. It reports that device's memory as its own,
# so the cluster places layers by what the device can hold, and every layer placed on this
# node runs there; this node keeps only the embedding, the head, the tokenizer and sampling.
# Start it beside a full node, with its own --api-port, --zenoh-port and EXO_HOME.
REMOTE_STAGE_PROXY_ENV: Final = "EXO_REMOTE_STAGE_PROXY"


def configured_stage_address() -> tuple[str, int] | None:
    """EXO_REMOTE_STAGE as (host, port), if set"""
    address = os.environ.get(REMOTE_STAGE_ENV)
    if not address:
        return None
    host, _, port = address.rpartition(":")
    return host, int(port)


def is_stage_proxy() -> bool:
    """Whether this node stands in for its remote stage's device (EXO_REMOTE_STAGE_PROXY=1)"""
    return (
        os.environ.get(REMOTE_STAGE_PROXY_ENV) == "1"
        and configured_stage_address() is not None
    )
