"""What a node tells a stage about the layers it sends: how each module is quantized, and RoPE
the stage can reproduce (or a plain refusal)."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from exo.worker.engines.mlx.remote_stage.client import (
    RemoteStageError,
    rope_frequencies,
    stage_quantization,
)


class _Attention(nn.Module):
    def __init__(self, rope: nn.Module) -> None:
        super().__init__()
        self.rope = rope


class _Layer(nn.Module):
    def __init__(self, rope: nn.Module | None = None) -> None:
        super().__init__()
        # A module without parameters is an empty dict, so `rope or ...` won't do
        self.self_attn = _Attention(
            nn.RoPE(64, base=1_000_000.0) if rope is None else rope
        )
        self.up_proj = nn.Linear(64, 64, bias=False)
        self.gate: nn.Module = nn.Linear(64, 64, bias=False)


def test_each_quantized_module_is_named_as_the_stage_names_it() -> None:
    layers = [_Layer(), _Layer()]
    for layer in layers:
        nn.quantize(layer, group_size=64, bits=4)
        # A router, as mixture-of-experts checkpoints keep it: at 8 bits
        layer.gate = nn.QuantizedLinear(64, 64, bias=False, group_size=64, bits=8)

    assert stage_quantization(layers) == {
        "layers.0.up_proj": {"group_size": 64, "bits": 4},
        "layers.0.gate": {"group_size": 64, "bits": 8},
        "layers.1.up_proj": {"group_size": 64, "bits": 4},
        "layers.1.gate": {"group_size": 64, "bits": 8},
    }


def test_other_than_affine_quantization_is_refused() -> None:
    layer = _Layer()
    nn.quantize(layer, group_size=32, bits=4, mode="mxfp4")

    with pytest.raises(RemoteStageError, match="mxfp4"):
        stage_quantization([layer])


def test_plain_rope_is_its_base_to_the_power_of_each_dimension() -> None:
    frequencies = rope_frequencies([_Layer()])

    expected = mx.power(
        mx.array(1_000_000.0), mx.arange(0, 64, 2, dtype=mx.float32) / 64
    )
    assert mx.allclose(frequencies, expected).item()


def test_partial_rope_has_frequencies_for_the_dimensions_it_rotates() -> None:
    # Qwen3.5 rotates a quarter of each head's dimensions
    frequencies = rope_frequencies([_Layer(nn.RoPE(64, base=10_000_000.0))])

    assert frequencies.shape == (32,)


@pytest.mark.parametrize(
    "rope",
    [
        nn.RoPE(64, base=10_000.0, scale=0.25),
        nn.RoPE(64, traditional=True, base=10_000.0),
    ],
    ids=["scaled positions", "traditional"],
)
def test_rope_a_stage_cant_reproduce_is_refused(rope: nn.Module) -> None:
    with pytest.raises(RemoteStageError, match="RoPE"):
        rope_frequencies([_Layer(rope)])
