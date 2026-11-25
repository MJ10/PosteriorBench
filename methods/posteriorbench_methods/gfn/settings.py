"""The settings of GFlowNet fine-tuning."""

from __future__ import annotations

import dataclasses

LOSS_FUNCTIONS = ("square", "linex")
# Settings beyond the paper's; their defaults reproduce the paper's runs.
ADDED_SETTINGS = (
    "corpus_budget_fraction",
    "corpus_probability",
    "loss_function",
    "loss_shape",
    "local_search_probability",
)


@dataclasses.dataclass(frozen=True)
class Settings:
    """The settings of GFlowNet and GFlowNet + composition. The defaults are the paper's.

    budgets: in passes of a model over one response; each is one of `training_budgets`.
    update_caps: a run at cap u submits one of the checkpoints saved in the first u updates; each
        cap is on the checkpoint schedule.
    training_budgets: training closes each with a checkpoint (`budget_rule`) and ends at the
        largest. `budgets` and `update_caps` only decide how far a run trains, so runs that
        differ in them share one training run.
    epsilon: the reward of a miss is epsilon p(y); that of a hit is p(y).
    warmup_updates: update u has learning rate learning_rate min((u - 1) / warmup_updates, 1), so
        the first update has learning rate 0.
    replay_probability, tempered_probability, temperature_low, temperature_high: how the groups
        of an update are drawn (`training.plan_groups`).
    buffer_similarity: two responses of n and m tokens are near-duplicates if their edit distance
        is below buffer_similarity (n + m); the buffer keeps the one with the higher reward.
    checkpoint_updates, checkpoint_interval: a checkpoint is saved after each update of
        `checkpoint_updates`, then after every multiple of `checkpoint_interval`.
    checkpoint_budget_counts: a checkpoint is also saved after the first update at which the
        budget count reaches each of these.
    generation_batch_size, forward_batch_size: the most responses in one generating and one
        teacher-forced pass; they change memory use and the rounding of sums, not the method.

    Additions beyond the paper, each off by default (FOLLOWUPS.md):
    corpus_budget_fraction, corpus_probability: before the first update, training buys
        floor(corpus_budget_fraction min(training_budgets) / (1 / p̂(S) + 1)) corpus hits, true
        posterior samples, at 1 / p̂(S) inferences each plus one for p of their prefixes, and a
        group of an update is drawn from them, uniformly, with probability corpus_probability
        (`training.plan_groups`). Both are 0, or both positive.
    loss_function: how a residual d of subtrajectory balance is penalized (`subtb.penalty`):
        "square", d^2, the paper's; "linex", the zero-avoiding Linex(loss_shape),
        2 (exp(-a d) + a d - 1) / a^2, which weighs a residual where the policy underweights a
        segment (d < 0) more than one where it overweights it.
    local_search_probability: a group is a local-search group with this probability, once a hit
        is known (a bought corpus hit or a hit in the buffer): it takes one such hit, keeps its
        first k content tokens for k uniform in [0, its content length), and draws the group's
        responses as that prefix followed by the policy's continuation at temperature 1, after
        Local Search GFlowNets (Kim et al., ICLR 2024). The responses are scored, charged and
        offered to the buffer as fresh responses, each charged as a whole generation."""

    budgets: tuple[int, ...] = (2**20, 2**21, 2**22)
    update_caps: tuple[int, ...] = (8, 16, 32, 64, 128, 256)
    training_budgets: tuple[int, ...] = (2**20, 2**21, 2**22)
    epsilon: float = 1e-8
    lora_rank: int = 64
    lora_alpha: int = 16
    learning_rate: float = 1e-4
    warmup_updates: int = 20
    groups_per_update: int = 25
    responses_per_group: int = 20
    replay_probability: float = 0.25
    tempered_probability: float = 0.666
    temperature_low: float = 0.5
    temperature_high: float = 2.0
    buffer_size: int = 50
    buffer_similarity: float = 0.25
    selection_draws: int = 128
    checkpoint_updates: tuple[int, ...] = (1, 2, 4, 8, 16, 20, 32, 64, 128)
    checkpoint_interval: int = 32
    checkpoint_budget_counts: tuple[int, ...] = (2**10, 2**12, 2**14, 2**16, 2**18, 2**20)
    generation_batch_size: int = 500
    forward_batch_size: int = 20
    corpus_budget_fraction: float = 0.0
    corpus_probability: float = 0.0
    loss_function: str = "square"
    loss_shape: float = 1.0
    local_search_probability: float = 0.0

    def __post_init__(self) -> None:
        if not self.budgets and not self.update_caps:
            raise ValueError("a run needs at least one budget or update cap")
        if not self.training_budgets or min(self.training_budgets) < 1:
            raise ValueError(f"training_budgets are positive, got {self.training_budgets}")
        unknown = sorted(set(self.budgets) - set(self.training_budgets))
        if unknown:
            raise ValueError(
                f"budgets {unknown} are not among the training budgets {self.training_budgets}"
            )
        off_schedule = [u for u in self.update_caps if not self.saves_checkpoint_after(u)]
        if off_schedule:
            raise ValueError(
                f"update caps {off_schedule} are not on the checkpoint schedule: "
                f"{self.checkpoint_updates}, then every {self.checkpoint_interval}th update"
            )
        if not 0 < self.epsilon < 1:
            raise ValueError(f"epsilon must lie in (0, 1), got {self.epsilon}")
        counts = {
            "lora_rank": self.lora_rank,
            "warmup_updates": self.warmup_updates,
            "groups_per_update": self.groups_per_update,
            "responses_per_group": self.responses_per_group,
            "buffer_size": self.buffer_size,
            "selection_draws": self.selection_draws,
            "checkpoint_interval": self.checkpoint_interval,
            "forward_batch_size": self.forward_batch_size,
        }
        too_small = [name for name, value in counts.items() if value < 1]
        if too_small:
            raise ValueError(f"{', '.join(too_small)} must be at least 1")
        if self.learning_rate <= 0 or self.lora_alpha <= 0 or self.buffer_similarity < 0:
            raise ValueError(
                "learning_rate and lora_alpha are positive and buffer_similarity is not negative"
            )
        if not (0 <= self.replay_probability <= 1 and 0 <= self.tempered_probability <= 1):
            raise ValueError("replay_probability and tempered_probability lie in [0, 1]")
        if not 0 < self.temperature_low <= self.temperature_high:
            raise ValueError("temperatures are positive, with temperature_low <= temperature_high")
        if not self.checkpoint_updates or min(self.checkpoint_updates) < 1:
            raise ValueError("checkpoint_updates holds at least one update, and none below 1")
        if list(self.checkpoint_budget_counts) != sorted(set(self.checkpoint_budget_counts)):
            raise ValueError("checkpoint_budget_counts must increase")
        if not (0 <= self.corpus_budget_fraction < 1 and 0 <= self.corpus_probability <= 1):
            raise ValueError(
                "corpus_budget_fraction lies in [0, 1) and corpus_probability in [0, 1]"
            )
        if (self.corpus_budget_fraction > 0) != (self.corpus_probability > 0):
            raise ValueError(
                "corpus_budget_fraction and corpus_probability are both 0 or both positive"
            )
        if self.loss_function not in LOSS_FUNCTIONS:
            raise ValueError(
                f"loss_function is one of {LOSS_FUNCTIONS}, got {self.loss_function!r}"
            )
        if self.loss_shape <= 0:
            raise ValueError("loss_shape is positive")
        if not 0 <= self.local_search_probability <= 1:
            raise ValueError("local_search_probability lies in [0, 1]")
        if self.generation_batch_size < self.responses_per_group:
            raise ValueError("generation_batch_size must hold one group of responses")

    def saves_checkpoint_after(self, update: int) -> bool:
        """Whether the schedule saves a checkpoint after `update` updates. The checkpoints at
        `checkpoint_budget_counts`, and those that close a budget, come on top."""
        if update in self.checkpoint_updates:
            return True
        return update > max(self.checkpoint_updates) and update % self.checkpoint_interval == 0
