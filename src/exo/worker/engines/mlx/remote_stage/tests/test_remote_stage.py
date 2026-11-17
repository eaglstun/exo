"""A model whose later layers run on a remote stage generates exactly what the whole model does,
through exo's own generator and prefix cache: a miss, a partial hit (trim), an exact hit, a
sampled request and an abandoned one. For each model family a stage runs; a model that isn't
downloaded is skipped.
"""

import gc
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.base import create_attention_mask
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import load_model

from exo.download.download_utils import resolve_existing_model
from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.mlx.cache import KVPrefixCache, encode_prompt, make_kv_cache
from exo.worker.engines.mlx.generator.batch_generate import ExoBatchGenerator
from exo.worker.engines.mlx.generator.generate import mlx_generate
from exo.worker.engines.mlx.models.registry import get_model_classes
from exo.worker.engines.mlx.remote_stage.client import (
    RemoteStage,
    RemoteStageLink,
    RemoteStageLostError,
    attach_remote_stage,
    has_cache,
    is_recurrent,
    remote_cache,
    rope_frequencies,
)
from exo.worker.engines.mlx.sharding.pipeline import get_inner_model, get_layers
from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    load_tokenizer_for_model_id,
)


@dataclass(frozen=True)
class StagedModel:
    model_id: ModelId
    # The layers the stage runs: [start_layer, end_layer)
    start_layer: int
    end_layer: int


MODELS = [
    StagedModel(ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"), 8, 16),
    # Qwen3: norms of each head's queries and keys
    StagedModel(ModelId("mlx-community/Qwen3-0.6B-4bit"), 14, 28),
    # A mixture of experts: 128 experts, 8 per token
    StagedModel(ModelId("mlx-community/Qwen3-30B-A3B-4bit"), 40, 48),
    # Qwen3.5: linear attention layers with a recurrent state, every fourth layer attention
    StagedModel(ModelId("mlx-community/Qwen3.5-2B-MLX-8bit"), 12, 24),
    # Qwen3.5 with 256 experts, 8 per token, and a shared expert
    StagedModel(ModelId("mlx-community/Qwen3.5-35B-A3B-4bit"), 34, 40),
    # Nemotron-H: Mamba-2, attention without RoPE and MLP blocks, only the first two with a
    # cache; the layers left before the stage have no attention
    StagedModel(ModelId("mlx-community/NVIDIA-Nemotron-Nano-9B-v2-4bits"), 12, 24),
    # Nemotron-H with 128 experts, 6 per token by a sigmoid router with a correction bias, and
    # a shared expert
    StagedModel(
        ModelId("mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-4Bit"), 40, 48
    ),
]


def _load_model(model_path: Path) -> tuple[nn.Module, dict[str, object]]:
    """As exo loads it: with its own model classes where it has them (Nemotron-H's computes
    its Mamba-2 mixer and router in float32)"""
    return cast(
        tuple[nn.Module, dict[str, object]],
        load_model(model_path, get_model_classes=get_model_classes),
    )


# How close, in logprob, two greedy candidates are when another GPU may pick either
TIE_LOGPROB_GAP = 0.02

# Over 1000 tokens, the least prefix mlx_generate reuses from the prefix cache
LONG_CONTEXT = " ".join(
    f"Box {number} holds {number * 7 % 13} {['red', 'green', 'blue', 'amber'][number % 4]} marbles."
    for number in range(1, 141)
)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])  # pyright: ignore[reportAny]


@pytest.fixture(
    scope="module",
    params=MODELS,
    ids=[model.model_id.split("/")[1] for model in MODELS],
)
def staged_model(request: pytest.FixtureRequest) -> StagedModel:
    return cast(StagedModel, request.param)


@pytest.fixture(scope="module")
def model_path(staged_model: StagedModel) -> Path:
    path = resolve_existing_model(staged_model.model_id)
    if path is None:
        pytest.skip(f"{staged_model.model_id} isn't downloaded")
    return path


@pytest.fixture(scope="module")
def link() -> Iterator[RemoteStageLink]:
    # EXO_REMOTE_STAGE=host:port tests a stage that's already running, e.g. on an iPad;
    # otherwise the Python reference stage runs here
    given = os.environ.get("EXO_REMOTE_STAGE")
    if given:
        host, _, port_text = given.rpartition(":")
        connected = RemoteStageLink(host, int(port_text))
        yield connected
        connected.close()
        return

    port = _free_port()
    stage = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "exo.worker.engines.mlx.remote_stage.server",
            "--port",
            str(port),
        ]
    )
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                connected = RemoteStageLink("127.0.0.1", port)
                break
            except RemoteStageLostError:
                if time.monotonic() > deadline or stage.poll() is not None:
                    raise
                time.sleep(0.2)
        yield connected
        connected.close()
    finally:
        stage.terminate()
        stage.wait(timeout=10)


def _task(
    model_id: ModelId,
    question: str,
    *,
    context: str = "",
    temperature: float = 0.0,
    seed: int | None = None,
    max_tokens: int = 24,
) -> TextGenerationTaskParams:
    content = f"{context}\n\n{question}" if context else question
    return TextGenerationTaskParams(
        model=model_id,
        input=[InputMessage(role="user", content=InputMessageContent(content))],
        max_output_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
        # Read where a stage on another device parts from the whole model
        logprobs=True,
        top_logprobs=5,
    )


def _record_remaining(prefix_cache: KVPrefixCache) -> list[int]:
    """From now on, how many tokens of each prompt `prefix_cache` leaves to prefill"""
    remaining: list[int] = []
    get = prefix_cache.get_kv_cache

    def recording(
        *args: object, **kwargs: object
    ) -> tuple[KVCacheType, mx.array, int | None, bool]:
        found = get(*args, **kwargs)  # pyright: ignore[reportArgumentType]
        remaining.append(len(found[1]))
        return found

    prefix_cache.get_kv_cache = recording
    return remaining


def _generate(
    model: Model,
    tokenizer: TokenizerWrapper,
    task: TextGenerationTaskParams,
    prefix_cache: KVPrefixCache,
    stop_after: int | None = None,
) -> list[GenerationResponse]:
    responses: list[GenerationResponse] = []
    generation = mlx_generate(
        model=model,
        tokenizer=tokenizer,
        task=task,
        prompt=apply_chat_template(tokenizer, task),
        kv_prefix_cache=prefix_cache,
        group=None,
    )
    for response in generation:
        responses.append(response)
        if response.finish_reason is not None or len(responses) == stop_after:
            break
    # What a cancelled request does to the generator
    generation.close()
    return responses


def _compare(
    name: str,
    tokenizer: TokenizerWrapper,
    expected: list[GenerationResponse],
    actual: list[GenerationResponse],
    task: TextGenerationTaskParams,
    *,
    other_device: bool,
) -> str:
    """Tokens must match. A stage on another GPU rounds differently, so there they may part
    where the whole model itself was undecided, and only there."""
    expected_tokens = [response.token for response in expected]
    actual_tokens = [response.token for response in actual]
    if actual_tokens == expected_tokens:
        return f"identical ({len(actual_tokens)} tokens)"
    first = next(
        (
            index
            for index, (want, got) in enumerate(
                zip(expected_tokens, actual_tokens, strict=False)
            )
            if want != got
        ),
        None,
    )
    assert first is not None, f"{name}: same tokens, but it stopped at another length"
    assert other_device, f"{name}: other tokens from token {first}"
    candidates = {
        item.token: item.logprob for item in expected[first].top_logprobs or []
    }
    chosen = tokenizer.decode([actual_tokens[first]])
    assert chosen in candidates, (
        f"{name}: at token {first} the stage chose {chosen!r}, not in the whole model's "
        f"top {len(candidates)}: {candidates}"
    )
    gap = max(candidates.values()) - candidates[chosen]
    if not task.temperature:
        assert gap <= TIE_LOGPROB_GAP, (
            f"{name}: at greedy token {first} the stage chose {chosen!r}, "
            f"{gap:.4f} below the whole model's choice"
        )
    return f"identical for {first} tokens, then a near-tie (logprob gap {gap:.4f})"


@pytest.mark.slow
def test_remote_stage_generates_what_the_whole_model_does(
    staged_model: StagedModel, model_path: Path, link: RemoteStageLink
) -> None:
    model_id = staged_model.model_id
    tokenizer = load_tokenizer_for_model_id(model_id, model_path)
    whole, _ = _load_model(model_path)
    staged, _ = _load_model(model_path)
    config = cast(
        dict[str, object], json.loads((model_path / "config.json").read_text())
    )
    attach_remote_stage(
        staged,  # pyright: ignore[reportArgumentType]
        link,
        staged_model.start_layer,
        staged_model.end_layer,
        config,
    )
    whole_prefix_cache = KVPrefixCache(None)
    staged_prefix_cache = KVPrefixCache(None)
    # The Python reference stage runs on this GPU, so its tokens must match exactly
    other_device = bool(os.environ.get("EXO_REMOTE_STAGE"))

    # (name, task, stop after n tokens, whether the prefix cache must have been used)
    scenarios: list[tuple[str, TextGenerationTaskParams, int | None, bool]] = [
        (
            "short, greedy",
            _task(model_id, "What is the capital of France?"),
            None,
            False,
        ),
        (
            "long, miss",
            _task(model_id, "How many marbles are in box 12?", context=LONG_CONTEXT),
            None,
            False,
        ),
        (
            "long, partial hit",
            _task(model_id, "Which boxes hold amber marbles?", context=LONG_CONTEXT),
            None,
            True,
        ),
        (
            "long, exact hit",
            _task(model_id, "Which boxes hold amber marbles?", context=LONG_CONTEXT),
            None,
            True,
        ),
        (
            "long, sampled",
            _task(
                model_id,
                "Describe box 40.",
                context=LONG_CONTEXT,
                temperature=0.7,
                seed=3,
                max_tokens=40,
            ),
            None,
            True,
        ),
        (
            "long, abandoned",
            _task(model_id, "List boxes 1 to 30.", context=LONG_CONTEXT),
            3,
            True,
        ),
        (
            "long, after abandoned",
            _task(model_id, "How many marbles are in box 12?", context=LONG_CONTEXT),
            None,
            True,
        ),
    ]

    # How many of each prompt's tokens each prefix cache leaves to prefill: the same, whether
    # or not the stage runs some of the layers
    whole_remaining = _record_remaining(whole_prefix_cache)
    staged_remaining = _record_remaining(staged_prefix_cache)
    # A model with recurrent state goes back only to snapshots, taken at the end of each
    # prompt: a prompt that only shares a prefix reuses nothing
    reuses_prefixes = link.state_bytes == 0

    for name, task, stop_after, expect_hit in scenarios:
        prompt_length = len(
            encode_prompt(tokenizer, apply_chat_template(tokenizer, task))
        )
        forwards_before = len(link.forward_lengths)
        expected = _generate(whole, tokenizer, task, whole_prefix_cache, stop_after)  # pyright: ignore[reportArgumentType]
        actual = _generate(staged, tokenizer, task, staged_prefix_cache, stop_after)  # pyright: ignore[reportArgumentType]
        prefilled = sum(
            length for length in link.forward_lengths[forwards_before:] if length > 1
        )
        outcome = _compare(
            name, tokenizer, expected, actual, task, other_device=other_device
        )
        print(f"{name}: prompt {prompt_length}, stage prefilled {prefilled}: {outcome}")
        assert staged_remaining[-1] == whole_remaining[-1], (
            f"{name}: {staged_remaining[-1]} tokens left to prefill with the stage, "
            f"{whole_remaining[-1]} without"
        )
        if expect_hit and (reuses_prefixes or name.endswith("exact hit")):
            assert prefilled < prompt_length // 2, (
                f"{name}: the prefix cache wasn't used"
            )

    # Every cache the node dropped is gone from the stage too
    del whole_prefix_cache, staged_prefix_cache
    gc.collect()
    mx.clear_cache()
    assert link.live_caches() == 0


@pytest.mark.slow
def test_remote_stage_computes_the_layers_bit_for_bit(
    staged_model: StagedModel, model_path: Path, link: RemoteStageLink
) -> None:
    """The stage's output for a prompt is the model's own layers' output, bit for bit, when
    it runs on this GPU: matching tokens can hide a stage that computes otherwise (e.g. one
    whose layers are left in training mode, which runs Qwen3.5's linear attention as ops)"""
    tokenizer = load_tokenizer_for_model_id(staged_model.model_id, model_path)
    model, _ = _load_model(model_path)
    config = cast(
        dict[str, object], json.loads((model_path / "config.json").read_text())
    )
    inner = get_inner_model(model)
    layers = cast(list[nn.Module], list(get_layers(inner)))
    staged = layers[staged_model.start_layer : staged_model.end_layer]
    link.load(staged, config, rope_frequencies(staged))
    prompt = encode_prompt(
        tokenizer,
        apply_chat_template(
            tokenizer,
            _task(
                staged_model.model_id,
                "Which boxes hold amber marbles?",
                context=LONG_CONTEXT[:600],
            ),
        ),
    )
    # One per layer that has a cache (Nemotron-H's MLP and expert layers have none)
    caches = iter(make_kv_cache(model))  # pyright: ignore[reportArgumentType]
    owner = cast(object, inner)
    embed = cast(
        nn.Module,
        getattr(owner, "embed_tokens", None) or getattr(owner, "embeddings"),  # noqa: B009
    )
    hidden = embed(prompt[None])
    attention_mask = cast(mx.array | None, create_attention_mask(hidden, None))
    stage_input = hidden
    for index, layer in enumerate(layers):
        if index == staged_model.start_layer:
            stage_input = hidden
        if index == staged_model.end_layer:
            break
        if not has_cache(layer):
            hidden = layer(hidden)
            continue
        mask = None if is_recurrent(layer) else attention_mask
        hidden = layer(hidden, mask, next(caches))
    mx.eval(hidden)
    stage_output = RemoteStage(link, 0, len(staged))(
        stage_input, None, remote_cache(link)
    )
    if os.environ.get("EXO_REMOTE_STAGE"):
        difference = mx.abs(
            stage_output.astype(mx.float32) - hidden.astype(mx.float32)
        ).max()
        print(f"largest difference on the other device: {difference.item()}")
        return
    assert mx.array_equal(stage_output, hidden).item()


def _generate_batched(
    model: Model,
    tokenizer: TokenizerWrapper,
    schedule: list[tuple[int, TextGenerationTaskParams]],
) -> list[list[int]]:
    """Each task's tokens, generated together by exo's batch generator; a task joins the
    batch at its step"""
    generator = ExoBatchGenerator(
        model=model,
        tokenizer=tokenizer,
        group=None,
        kv_prefix_cache=KVPrefixCache(None),
    )
    tokens: dict[int, list[int]] = {}
    order: list[int] = []
    step = 0
    pending = sorted(schedule, key=lambda item: item[0])
    while pending or generator.has_work:
        while pending and pending[0][0] <= step:
            _, task = pending.pop(0)
            uid = generator.submit(task, apply_chat_template(tokenizer, task))
            order.append(uid)
            tokens[uid] = []
        for uid, response in generator.step():
            tokens[uid].append(response.token)
        step += 1
    generator.close()
    return [tokens[uid] for uid in order]


@pytest.mark.slow
def test_remote_stage_generates_what_the_whole_model_does_in_a_batch(
    staged_model: StagedModel, model_path: Path, link: RemoteStageLink
) -> None:
    """Requests of different lengths join a running batch and leave it at different times:
    the stage merges, extends, filters and extracts its batch caches as mlx-lm does"""
    model_id = staged_model.model_id
    tokenizer = load_tokenizer_for_model_id(model_id, model_path)
    whole, _ = _load_model(model_path)
    staged, _ = _load_model(model_path)
    config = cast(
        dict[str, object], json.loads((model_path / "config.json").read_text())
    )
    attach_remote_stage(
        staged,  # pyright: ignore[reportArgumentType]
        link,
        staged_model.start_layer,
        staged_model.end_layer,
        config,
    )
    schedule = [
        (0, _task(model_id, "What is the capital of France?", max_tokens=12)),
        (0, _task(model_id, "Name three primary colors.", max_tokens=30)),
        (3, _task(model_id, "How many marbles are in box 12?", context=LONG_CONTEXT)),
        (6, _task(model_id, "Count from one to ten.", max_tokens=20)),
        (6, _task(model_id, "Which boxes hold amber marbles?", context=LONG_CONTEXT)),
    ]

    expected = _generate_batched(whole, tokenizer, schedule)  # pyright: ignore[reportArgumentType]
    actual = _generate_batched(staged, tokenizer, schedule)  # pyright: ignore[reportArgumentType]

    other_device = bool(os.environ.get("EXO_REMOTE_STAGE"))
    for index, (want, got) in enumerate(zip(expected, actual, strict=True)):
        print(f"request {index}: {len(got)} tokens, identical: {want == got}")
        if not other_device:
            assert got == want, f"request {index}: {tokenizer.decode(got)!r}"
    if other_device:
        # Another GPU may part from the whole model at a near-tie; most requests don't
        assert sum(want == got for want, got in zip(expected, actual, strict=True)) >= 3

    gc.collect()
    mx.clear_cache()
    assert link.live_caches() == 0
