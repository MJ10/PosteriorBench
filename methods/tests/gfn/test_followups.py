"""Tests of the GFlowNet settings added beyond the paper (gfn/FOLLOWUPS.md): corpus-hit replay
and the Linex loss. Each is off by default, and at its default the method is the paper's."""

from __future__ import annotations

import dataclasses
import math
import types
from typing import Any

import numpy as np
import pytest
import torch
from posteriorbench import ledger as pb_ledger

from posteriorbench_methods.gfn import cost, reward
from posteriorbench_methods.gfn import settings as gfn_settings
from posteriorbench_methods.gfn import subtb, training

Settings = gfn_settings.Settings


# Settings


def test_added_settings_default_off() -> None:
    settings = Settings()
    assert settings.corpus_budget_fraction == 0 and settings.corpus_probability == 0
    assert settings.loss_function == "square"
    assert set(gfn_settings.ADDED_SETTINGS) <= {f.name for f in dataclasses.fields(Settings)}


@pytest.mark.parametrize(
    "changes",
    [
        {"corpus_budget_fraction": 0.1},  # without a probability
        {"corpus_probability": 0.2},  # without hits to draw
        {"corpus_budget_fraction": 1.0, "corpus_probability": 0.2},
        {"corpus_budget_fraction": 0.1, "corpus_probability": 1.5},
        {"loss_function": "cube"},
        {"loss_shape": 0.0},
    ],
)
def test_invalid_added_settings(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        Settings(**changes)


def test_training_run_settings_leave_out_defaults() -> None:
    context = types.SimpleNamespace(task=types.SimpleNamespace(task_id="t/b"), replicate=0)
    paper = training.training_run_settings(context, Settings())  # type: ignore[arg-type]
    assert not set(gfn_settings.ADDED_SETTINGS) & set(paper)
    changed = training.training_run_settings(
        context,  # type: ignore[arg-type]
        Settings(corpus_budget_fraction=0.1, corpus_probability=0.2, loss_function="linex"),
    )
    assert changed["corpus_probability"] == 0.2 and changed["loss_function"] == "linex"
    assert "loss_shape" not in changed


# Group plans


def _seeds(count: int) -> list[int]:
    return [1000 + index for index in range(count)]


def test_plan_groups_unchanged_without_corpus() -> None:
    """With the corpus off, or with no corpus hits, the plans are the paper's."""
    paper = training.plan_groups(_seeds(200), buffer_is_empty=False, settings=Settings())
    for settings, available in [
        (Settings(), True),
        (Settings(corpus_budget_fraction=0.1, corpus_probability=0.5), False),
    ]:
        plans = training.plan_groups(
            _seeds(200), buffer_is_empty=False, settings=settings, corpus_available=available
        )
        assert [(p.temperature, p.replay, p.corpus) for p in plans] == [
            (p.temperature, p.replay, p.corpus) for p in paper
        ]


def test_plan_groups_draw_corpus_groups() -> None:
    settings = Settings(corpus_budget_fraction=0.1, corpus_probability=0.3)
    plans = training.plan_groups(
        _seeds(4000), buffer_is_empty=False, settings=settings, corpus_available=True
    )
    corpus = [p for p in plans if p.corpus]
    replayed = [p for p in plans if p.replay]
    fresh = [p for p in plans if p.temperature is not None]
    assert all(p.temperature is None and not p.replay for p in corpus)
    assert len(corpus) + len(replayed) + len(fresh) == len(plans)
    assert abs(len(corpus) / len(plans) - 0.3) < 0.03
    assert abs(len(replayed) / len(plans) - 0.7 * 0.25) < 0.03
    # Corpus groups need no buffer: an empty buffer still allows them.
    first = training.plan_groups(
        _seeds(50), buffer_is_empty=True, settings=settings, corpus_available=True
    )
    assert any(p.corpus for p in first)


# The loss


def _response_scores(seed: int, length: int, stops: int = 2) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(length, 6, generator=generator, dtype=torch.float64)
    logq = torch.log_softmax(logits, 1)
    logp = torch.log_softmax(
        logits + torch.randn(length, 6, generator=generator, dtype=torch.float64), 1
    )
    hits = [bool(i % 3 == 0) for i in range(length)]  # terminated: N + 1 = length prefixes
    return dict(
        token_logq=logq[:, 0],
        stop_logq=logq[:, 1 : 1 + stops],
        token_logp=logp[:, 0],
        stop_logp=logp[:, 1 : 1 + stops],
        prefix_hits=hits,
        epsilon=1e-3,
        terminated=True,
    )


def _pairwise(scores: dict[str, Any], g: Any) -> float:
    """The loss by its definition: g of every pairwise residual, plus the squared stop-token
    terms."""
    zero = torch.zeros(1, dtype=torch.float64)
    length = len(scores["token_logq"])
    content = length - 1
    q_prefix = torch.cat((zero, scores["token_logq"][:content].cumsum(0)))
    p_prefix = torch.cat((zero, scores["token_logp"][:content].cumsum(0)))
    factor = torch.tensor(
        [0.0 if h else math.log(scores["epsilon"]) for h in scores["prefix_hits"]],
        dtype=torch.float64,
    )
    rewards = p_prefix[:, None] + scores["stop_logp"] + factor[:, None]
    flow = torch.logsumexp(rewards, 1) - torch.logsumexp(scores["stop_logq"], 1)
    a = flow - q_prefix
    total = 0.0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            total += float(g(a[i] - a[j]))
    stop_flow = rewards - scores["stop_logq"]
    total += float((stop_flow[:, 0] - stop_flow[:, 1]).square().sum())
    return total


def test_square_loss_matches_definition() -> None:
    scores = _response_scores(0, 9)
    total, count = subtb.subtb_terms(**scores)
    assert math.isclose(float(total), _pairwise(scores, lambda d: d * d), rel_tol=1e-9)
    assert count == subtb.term_count(9, terminated=True, stop_tokens=2)


@pytest.mark.parametrize("shape", [0.1, 0.5, 1.0])
def test_linex_loss_matches_definition(shape: float) -> None:
    scores = _response_scores(1, 12)
    total, count = subtb.subtb_terms(**scores, loss_function="linex", loss_shape=shape)

    def linex(d: torch.Tensor) -> float:
        return 2 * (math.exp(-shape * float(d)) + shape * float(d) - 1) / shape**2

    assert math.isclose(float(total), _pairwise(scores, linex), rel_tol=1e-9)
    assert count == subtb.term_count(12, terminated=True, stop_tokens=2)


def test_linex_penalty_properties() -> None:
    d = torch.tensor([-2.0, -0.01, 0.0, 0.01, 2.0], dtype=torch.float64)
    square = subtb.penalty(d, loss_function="square", shape=1.0)
    linex = subtb.penalty(d, loss_function="linex", shape=0.5)
    assert float(linex[2]) == 0
    assert torch.allclose(linex[1:4], square[1:4], rtol=1e-2, atol=1e-8)  # d^2 near 0
    assert float(linex[0]) > float(square[0]) > float(linex[4])  # heavier for d < 0
    small = subtb.penalty(d, loss_function="linex", shape=1e-4)
    assert torch.allclose(small, square, rtol=1e-3)  # the square as the shape goes to 0


def test_linex_gradient_finite() -> None:
    scores = _response_scores(2, 30)
    token_logq = scores["token_logq"].clone().requires_grad_(True)
    total, _ = subtb.subtb_terms(
        **(scores | {"token_logq": token_logq}), loss_function="linex", loss_shape=0.25
    )
    total.backward()
    assert token_logq.grad is not None and torch.isfinite(token_logq.grad).all()


# Corpus hits


def _response(token_ids: list[int], hit: bool = True, logp: float = -3.0) -> reward.Response:
    return reward.Response(
        token_ids=token_ids,
        text=" ".join(map(str, token_ids)),
        hit=hit,
        logp=logp,
        token_logp=[logp / len(token_ids)] * len(token_ids),
        stop_logp=[[-1.0, -2.0]] * len(token_ids),
        prefix_hits=[hit] * len(token_ids),
    )


class _FakeAccess:
    def __init__(self, ledger: pb_ledger.Ledger, *, available: int, probability: float) -> None:
        self.ledger, self.available, self.probability = ledger, available, probability
        self.taken: list[int] = []

    def cost_of(self, k: int) -> float:
        return k / self.probability

    def take(self, positions: list[int]) -> list[Any]:
        self.taken = list(positions)
        self.ledger.charge(
            pb_ledger.Category.CORPUS_HITS,
            inferences=len(positions) / self.probability,
            note="corpus hits",
        )
        return [types.SimpleNamespace(token_ids=(position, 7)) for position in positions]


def _training(settings: Settings, *, available: int, probability: float) -> Any:
    ledger = pb_ledger.Ledger()
    events: list[tuple[str, dict[str, Any]]] = []
    access = _FakeAccess(ledger, available=available, probability=probability)
    context = types.SimpleNamespace(
        options=types.SimpleNamespace(data=None),
        bench=types.SimpleNamespace(corpus=lambda task, spend: access),
        task=types.SimpleNamespace(task_id="t/b"),
        original=types.SimpleNamespace(decode=lambda ids: " ".join(map(str, ids))),
        tracker=types.SimpleNamespace(event=lambda name, **values: events.append((name, values))),
        stream_seed=lambda *parts: 1234,
    )
    run = training.Training(
        context=context,  # type: ignore[arg-type]
        settings=settings,
        policy=types.SimpleNamespace(stop_token_ids=(7, 8)),  # type: ignore[arg-type]
        p=None,  # type: ignore[arg-type]
        is_hit=lambda text: True,
        directory=None,  # type: ignore[arg-type]
        optimizer=None,  # type: ignore[arg-type]
        buffer=None,  # type: ignore[arg-type]
        ledger=ledger,
        checkpoints=[],
    )
    return run, access, events


@pytest.fixture
def fake_score(monkeypatch: pytest.MonkeyPatch) -> None:
    def score(p: Any, stops: Any, responses: Any, *, decode: Any, is_hit: Any) -> list[Any]:
        return [_response(list(ids)) for ids in responses]

    monkeypatch.setattr(training.reward, "score", score)


def test_buy_corpus_hits_charges_and_counts(fake_score: None) -> None:
    settings = Settings(
        training_budgets=(2**20,),
        budgets=(2**20,),
        update_caps=(),
        corpus_budget_fraction=0.1,
        corpus_probability=0.2,
    )
    run, access, events = _training(settings, available=100_000, probability=0.01)
    run.buy_corpus_hits()
    expected = int(0.1 * 2**20 // 101)  # 1 / p̂(S) + 1 for each hit
    assert len(run.corpus_pool) == len(access.taken) == expected
    assert len(set(access.taken)) == expected
    totals = run.ledger.totals().to_dict()
    assert math.isclose(run.ledger.inferences, expected * 101)
    assert run.ledger.budget <= 0.1 * 2**20
    assert events[-1][0] == "corpus_hits" and events[-1][1]["hits"] == expected
    assert totals  # the breakdown is recorded


def test_buy_corpus_hits_caps_and_skips(fake_score: None) -> None:
    settings = Settings(corpus_budget_fraction=0.5, corpus_probability=0.2)
    run, access, _ = _training(settings, available=12, probability=0.5)
    run.buy_corpus_hits()
    assert len(run.corpus_pool) == 12  # no more than the corpus holds
    rare = Settings(corpus_budget_fraction=0.01, corpus_probability=0.2)
    run, access, events = _training(rare, available=500, probability=1e-5)
    run.buy_corpus_hits()  # one hit would cost more than the fraction: none is bought
    assert run.corpus_pool == [] and access.taken == [] and run.ledger.inferences == 0
    assert events[-1][1]["hits"] == 0


def test_corpus_group_trains_on_pool_and_pays_as_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(corpus_budget_fraction=0.1, corpus_probability=0.2)
    run, _, _ = _training(settings, available=10, probability=0.5)
    run.corpus_pool = [_response([index, 7]) for index in range(5)]
    seen: list[list[reward.Response]] = []

    def add_gradient(responses: list[reward.Response]) -> tuple[float, list[float]]:
        seen.append(list(responses))
        return 0.5, [-1.0] * len(responses)

    monkeypatch.setattr(run, "add_gradient", add_gradient)
    plan = training.GroupPlan(0, 99, None, np.random.default_rng(99), corpus=True)
    loss, responses, logq = run.train_group(plan, None)
    assert loss == 0.5 and len(responses) == settings.responses_per_group == len(logq)
    assert all(response in run.corpus_pool for response in responses)
    count = settings.responses_per_group
    assert run.ledger.inferences == count and run.ledger.budget == 2 * count
    assert run.ledger.inferences == sum(s.inferences for s in cost.replayed_responses(count))


def test_training_rows_mark_corpus() -> None:
    plan = training.GroupPlan(3, 5, None, np.random.default_rng(5), corpus=True)
    rows = training._training_rows(  # pyright: ignore[reportPrivateUsage]
        plan, [_response([1, 7])], [-1.0], first=0, mark_corpus=True
    )
    assert rows[0]["corpus"] is True and rows[0]["replayed"] is False
    paper = training._training_rows(  # pyright: ignore[reportPrivateUsage]
        plan, [_response([1, 7])], [-1.0], first=0
    )
    assert "corpus" not in paper[0]
