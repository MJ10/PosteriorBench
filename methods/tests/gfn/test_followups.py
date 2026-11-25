"""Tests of the GFlowNet settings added beyond the paper (gfn/FOLLOWUPS.md): corpus-hit replay,
the Linex loss, local search and the VarGrad objective. Each is off by default, and at its default the method is the paper's."""

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
from posteriorbench_methods.gfn import subtb, teacher_forcing, training, vargrad

Settings = gfn_settings.Settings


# Settings


def test_added_settings_default_off() -> None:
    settings = Settings()
    assert settings.corpus_budget_fraction == 0 and settings.corpus_probability == 0
    assert settings.loss_function == "square"
    assert settings.local_search_probability == 0
    assert settings.objective == "subtb"
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
        {"local_search_probability": 1.5},
        {"local_search_probability": -0.1},
        {"objective": "db"},
        {"objective": "vargrad", "responses_per_group": 1, "generation_batch_size": 1},
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
    assert "loss_shape" not in changed and "local_search_probability" not in changed
    local = training.training_run_settings(
        context,  # type: ignore[arg-type]
        Settings(local_search_probability=0.2),
    )
    assert local["local_search_probability"] == 0.2
    assert "objective" not in local
    vargrad_run = training.training_run_settings(
        context,  # type: ignore[arg-type]
        Settings(objective="vargrad"),
    )
    assert vargrad_run["objective"] == "vargrad"


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


def test_plan_groups_unchanged_without_local_search() -> None:
    """Without seeds, or with the probability at 0, local search leaves the plans as they were."""
    corpus = Settings(corpus_budget_fraction=0.1, corpus_probability=0.3)
    for settings in [Settings(), corpus]:
        before = training.plan_groups(
            _seeds(300), buffer_is_empty=False, settings=settings, corpus_available=True
        )
        for changed, seeds in [(settings, True), (_with_local(settings, 0.5), False)]:
            plans = training.plan_groups(
                _seeds(300),
                buffer_is_empty=False,
                settings=changed,
                corpus_available=True,
                seeds_available=seeds,
            )
            assert [(p.temperature, p.replay, p.corpus, p.local) for p in plans] == [
                (p.temperature, p.replay, p.corpus, p.local) for p in before
            ]


def _with_local(settings: Settings, probability: float) -> Settings:
    return dataclasses.replace(settings, local_search_probability=probability)


def test_plan_groups_draw_local_groups() -> None:
    settings = Settings(
        corpus_budget_fraction=0.1, corpus_probability=0.2, local_search_probability=0.5
    )
    plans = training.plan_groups(
        _seeds(4000),
        buffer_is_empty=False,
        settings=settings,
        corpus_available=True,
        seeds_available=True,
    )
    local = [p for p in plans if p.local]
    assert all(p.temperature is None and not p.replay and not p.corpus for p in local)
    assert abs(len(local) / len(plans) - 0.8 * 0.5) < 0.03
    assert abs(sum(p.corpus for p in plans) / len(plans) - 0.2) < 0.03


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


def test_train_update_draws_only_fresh_groups(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """An update with corpus groups draws fresh responses for the fresh groups alone, and hands
    each group its own source."""
    settings = Settings(corpus_budget_fraction=0.1, corpus_probability=0.5, groups_per_update=12)
    run, _, events = _training(settings, available=10, probability=0.5)
    run.corpus_pool = [_response([index, 7]) for index in range(5)]
    run.buffer = [_response([1, 7])]  # a non-empty buffer
    run.directory = tmp_path
    parameter = torch.nn.Parameter(torch.zeros(1))
    run.optimizer = torch.optim.AdamW([parameter])
    drawn: list[Any] = []

    def draw(groups: list[Any]) -> list[list[Any]]:
        drawn.extend(groups)
        return [[object()] for _ in groups]

    run.policy = types.SimpleNamespace(
        draw=draw, model=types.SimpleNamespace(parameters=lambda: [parameter])
    )
    run.context.stream_seed = lambda *parts: hash(parts) % 2**31
    seen: list[tuple[Any, Any]] = []

    def train_group(plan: Any, draws: Any) -> tuple[float, list[Any], list[float]]:
        seen.append((plan, draws))
        return 0.0, [_response([1, 7])], [-1.0]

    monkeypatch.setattr(run, "train_group", train_group)
    run.train_update()
    assert len(seen) == 12 and any(plan.corpus for plan, _ in seen)
    assert len(drawn) == sum(plan.temperature is not None for plan, _ in seen)
    for plan, draws in seen:
        assert (draws is None) == (plan.temperature is None)
    update = [values for name, values in events if name == "update"][0]
    assert update["corpus_groups"] == sum(plan.corpus for plan, _ in seen)


# Local search


class _FakeBuffer:
    def __init__(self, held: list[reward.Response]) -> None:
        self.held, self.added = list(held), []

    def responses(self) -> list[reward.Response]:
        return list(self.held)

    def add(self, response: reward.Response) -> None:
        self.added.append(response)


def test_local_search_seeds_are_corpus_then_buffer_hits() -> None:
    run, _, _ = _training(Settings(local_search_probability=0.5), available=1, probability=0.5)
    run.corpus_pool = [_response([1, 2, 7])]
    run.buffer = _FakeBuffer([_response([3, 7], hit=False), _response([4, 5, 7])])
    assert [r.token_ids for r in run.local_search_seeds()] == [[1, 2, 7], [4, 5, 7]]


def test_local_group_continues_a_hit_prefix_and_pays_as_fresh(
    monkeypatch: pytest.MonkeyPatch, fake_score: None
) -> None:
    settings = Settings(local_search_probability=0.5)
    run, _, _ = _training(settings, available=1, probability=0.5)
    start = _response([11, 12, 13, 14, 7])
    run.buffer = _FakeBuffer([start])
    asked: list[tuple[list[int], Any]] = []

    def draw_after(prefix: list[int], group: Any) -> list[Any]:
        asked.append((list(prefix), group))
        return [types.SimpleNamespace(token_ids=[99, 7]) for _ in range(group.count)]

    run.policy = types.SimpleNamespace(
        stop_token_ids=(7, 8),
        generation=types.SimpleNamespace(response_format=types.SimpleNamespace(max_tokens=10)),
        draw_after=draw_after,
        logq=lambda responses: [-2.0] * len(responses),
    )
    monkeypatch.setattr(run, "add_gradient", lambda responses: (0.25, [-2.0] * len(responses)))
    prefixes = set()
    for seed in range(40):
        plan = training.GroupPlan(0, seed, None, np.random.default_rng(seed), local=True)
        loss, responses, logq = run.train_group(plan, None)
        prefix, group = asked[-1]
        prefixes.add(tuple(prefix))
        assert loss == 0.25 and len(responses) == settings.responses_per_group == len(logq)
        assert group.count == settings.responses_per_group and group.temperature == 1.0
        assert prefix == [11, 12, 13, 14][: len(prefix)] and len(prefix) < 4  # a strict prefix
        assert all(r.token_ids == [*prefix, 99, 7] for r in responses)
    assert prefixes == {(), (11,), (11, 12), (11, 12, 13)}
    count = 40 * settings.responses_per_group
    fresh = cost.fresh_responses(count)
    assert run.ledger.inferences == sum(s.inferences for s in fresh)
    assert run.ledger.budget == sum(s.budget for s in fresh)
    assert len(run.buffer.added) == count


def test_draw_after_extends_the_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    from posteriorbench_methods.common import hf_runtime
    from posteriorbench_methods.gfn import policy as gfn_policy

    seen: list[Any] = []

    def sample_groups(models: Any, groups: Any, generation: Any) -> list[list[str]]:
        seen.append((models.q.prompt_ids, generation.response_format.max_tokens, groups))
        return [["draw"]]

    monkeypatch.setattr(gfn_policy.hf_runtime, "sample_groups", sample_groups)
    model = types.SimpleNamespace(eval=lambda: None)
    fmt = hf_runtime.ResponseFormat((7,), 10)
    policy = gfn_policy.Policy(
        model,
        hf_runtime.CausalLM(model, (1, 2)),
        hf_runtime.Generation(fmt, batch_size=500, renormalize=False),
        20,
    )
    group = hf_runtime.DrawGroup(3, 5)
    assert policy.draw_after([4, 5, 6], group) == ["draw"]
    assert seen == [((1, 2, 4, 5, 6), 7, [group])]
    with pytest.raises(ValueError):
        policy.draw_after(list(range(10)), group)


# VarGrad


def test_vargrad_square_is_group_variance() -> None:
    generator = torch.Generator().manual_seed(0)
    logq = -torch.rand(6, generator=generator, dtype=torch.float64) * 30
    log_rewards = -torch.rand(6, generator=generator, dtype=torch.float64) * 30
    total = vargrad.vargrad_terms(logq, log_rewards)
    expected = 6 * (log_rewards - logq).var(unbiased=False)
    assert math.isclose(float(total), float(expected), rel_tol=1e-12)


def test_vargrad_zero_when_policy_is_proportional_to_reward() -> None:
    log_rewards = torch.tensor([-40.0, -12.0, -3.0, -25.0], dtype=torch.float64)
    total = vargrad.vargrad_terms(log_rewards - 7.5, log_rewards)
    assert abs(float(total)) < 1e-20


@pytest.mark.parametrize("loss_function", ["square", "linex"])
def test_vargrad_gradient_holds_log_z_fixed(loss_function: str) -> None:
    logq = torch.tensor([-10.0, -4.0, -20.0, -6.0], dtype=torch.float64, requires_grad=True)
    log_rewards = torch.tensor([-12.0, -3.0, -30.0, -5.0], dtype=torch.float64)
    total = vargrad.vargrad_terms(logq, log_rewards, loss_function=loss_function, loss_shape=0.25)
    total.backward()
    log_z = float((log_rewards - logq.detach()).mean())
    d = log_z + logq.detach() - log_rewards
    if loss_function == "square":
        expected = 2 * d
    else:  # g'(d) of 2 (exp(-a d) + a d - 1) / a^2
        expected = 2 * (1 - torch.exp(-0.25 * d)) / 0.25
    assert logq.grad is not None and torch.allclose(logq.grad, expected, rtol=1e-12)
    # With the square, holding log Z fixed gives the gradient of the variance itself.
    if loss_function == "square":
        free = logq.detach().clone().requires_grad_(True)
        (4 * (log_rewards - free).var(unbiased=False)).backward()
        assert free.grad is not None and torch.allclose(free.grad, logq.grad, rtol=1e-12)


def test_vargrad_needs_two_responses() -> None:
    with pytest.raises(ValueError):
        vargrad.vargrad_terms(torch.zeros(1), torch.zeros(1))
    with pytest.raises(ValueError):
        vargrad.vargrad_terms(torch.zeros(3), torch.zeros(2))


def test_add_gradient_vargrad_spans_forward_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    """log q of response i is w[i] split over its tokens; the group of 5 spans two forward
    batches of 3, and the gradient is that of one VarGrad loss over the whole group."""
    settings = Settings(objective="vargrad", responses_per_group=5, groups_per_update=2)
    run, _, _ = _training(settings, available=0, probability=0.5)
    weights = torch.tensor([-9.0, -2.0, -15.0, -4.0, -6.0], requires_grad=True)
    batches: list[int] = []

    def token_scores(lm: Any, stops: Any, responses: Any) -> list[tuple[torch.Tensor, Any]]:
        batches.append(len(responses))
        return [
            (weights[ids[0]] * torch.full((len(ids),), 1 / len(ids)), None) for ids in responses
        ]

    monkeypatch.setattr(teacher_forcing, "token_scores", token_scores)
    run.policy = types.SimpleNamespace(
        stop_token_ids=(7, 8),
        forward_batch_size=3,
        lm=None,
        model=types.SimpleNamespace(train=lambda: None),
    )
    responses = [
        _response([0, 7], hit=True, logp=-8.0),
        _response([1, 5, 7], hit=False, logp=-1.0),
        _response([2, 5, 5], hit=True, logp=-14.0),
        _response([3, 7], hit=True, logp=-5.0),
        _response([4, 5, 8], hit=False, logp=-2.0),
    ]
    loss, logq = run.add_gradient(responses)
    assert batches == [3, 2]
    assert np.allclose(logq, weights.detach().numpy(), atol=1e-5)
    log_rewards = torch.tensor([r.log_reward(settings.epsilon) for r in responses])
    d = (log_rewards - weights.detach()).mean() + weights.detach() - log_rewards
    assert math.isclose(loss, float(d.square().mean()), rel_tol=1e-5)
    assert weights.grad is not None
    assert torch.allclose(weights.grad, 2 * d / 5 / 2, rtol=1e-5)
