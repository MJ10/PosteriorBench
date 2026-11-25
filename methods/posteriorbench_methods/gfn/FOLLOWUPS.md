# Follow-up work on GFlowNet fine-tuning, and what it means for this method

This method adapts Hu et al., "Amortizing intractable inference in large language models" (ICLR 2024, arXiv:2310.04363; [GFNOrg/gfn-lm-tuning](https://github.com/GFNOrg/gfn-lm-tuning)). The notes below cover later work on GFlowNet training of language models and samplers that might improve its scores. They also describe the two settings added from that work and their first results. Every citation was checked against arXiv in November 2026.

## Where GFlowNet loses on the benchmark

At B = 2^20 (about 4.3 × 10^5 inferences), the paper's GFlowNet has a mean capped forward KL of 13.5 and a reverse KL of 20.5 over the 21 tasks (`figures/data/main_comparison_runs.csv.gz`). It loses in two ways:

1. **It never finds a hit on the rarest tasks.** On three tasks with p̂(S) ≈ 10^-5 (`wc_1301/chatgpt_mention`, `wc_1802/contrastive_phrasing`, `wc_0842/word_repetition`) every replicate falls back to a forward KL of 50. Those three tasks alone add 7.1 to the mean forward KL.
2. **It collapses onto a few modes on the common tasks.** There it reaches a 90–99% hit rate but covers the posterior poorly: forward KL 10.9 vs SFT's 1.5 on `wc_0670/chinese_character`, 9.5 vs 1.8 on `wc_1405/chinese_character`, 6.7 vs 0.9 on `wc_0670/contrastive_phrasing`. SFT trains on true posterior samples. GFlowNet trains on its own draws and on a buffer of the 50 highest-reward responses, and on-policy SubTB has the gradient of a reverse, mode-seeking KL.

Forward KL is the score where GFlowNet is furthest behind, so the ideas below are judged mostly by whether they help mass coverage. Each idea is also costed against the construction budget (`docs/cost_accounting.md`). Generations and p-scoring cost inferences; a change to the loss is free.

## The papers

### Off-policy and asynchronous training

- **TBA.** Bartoldson, Venkatraman, Diffenderfer, Jain, Ben-Nun, Lee, Kim, Obando-Ceron, Bengio, Kailkhura. *Trajectory Balance with Asynchrony: Decoupling Exploration and Learning for Fast, Scalable LLM Post-Training.* NeurIPS 2025, arXiv:2503.18929.
  - **Idea.** Searcher processes fill a central buffer while a learner trains on it with off-policy TB. The TB uses a VarGrad log Z: a detached batch mean of log R − log π over the K ≈ 20 responses to a prompt. For each prompt, the buffer is sampled by recency with probability m and otherwise by reward. m ≈ 0.5 worked best, and staler data hurt little.
  - **Here.** Asynchrony is a wall-clock gain, and the benchmark charges inferences, not time. Two parts would carry over:
    - VarGrad in place of the stop-probability flows. The flows need no log Z here: one prompt, and every prefix is terminable.
    - Recency mixing in the buffer. The buffer is already a mix: 25% replay from a reward-prioritized top-50, 75% fresh. Not tried.

### Data for the buffer: expert or true posterior samples

- **Hu et al. 2024 itself.** The buffer is seeded with ground-truth rationales in story infilling (App. C) and with 50 demonstrations in integer arithmetic (§4.4). In those tasks the ground-truth rationales are samples of the target, so seeding them is the paper's own recipe.
- **Flow of Reasoning.** Yu, Jiang, Kang, Hao, Qin. *Flow of Reasoning: Training LLMs for Divergent Reasoning with Minimal Examples.* ICML 2025, arXiv:2406.05673. Trains TB with a VarGrad log Z on a mix of on-policy, tempered and buffer data plus offline ground-truth trajectories. Removing the buffer cost 4 to 19 points.
- **GFlowNet red-teaming.** Lee, Kim, Cherif, Dobre, Lee, Hwang, Kawaguchi, Gidel, Bengio, Malkin, Jain. *Learning diverse attacks on large language models for robust red-teaming and safety tuning.* ICLR 2025, arXiv:2405.18540.
  - Stage 1 is TB with groups drawn 50/50 from a tempered policy and the buffer.
  - Stage 2 refits the reference model by maximum likelihood on all high-reward samples. It took under 5% of the compute and improved diversity.
- **Replay and mode discovery.** Vemgal, Lau, Precup. *An Empirical Study of the Effectiveness of Using a Replay Buffer on Mode Discovery in GFlowNets.* ICML 2023 SPIGM workshop, arXiv:2307.07674. Reward-prioritized replay speeds up mode discovery.
- **Here.** The benchmark sells true posterior samples: a corpus hit costs 1/p̂(S) inferences. That is under 300 inferences per hit on 9 of the 21 tasks, but 10^4 to 10^5 on the rarest. Training SubTB on them is an off-policy update toward exactly the target. It pushes the policy up where q(y) is far below R(y)/Z, which is the deficit that forward KL measures. **Implemented** as `corpus_budget_fraction` and `corpus_probability` (below).

### Loss design

- **Beyond Squared Error.** Hu, Zhang, Li, Huang. *Beyond Squared Error: Exploring Loss Design for Enhanced Training of Generative Flow Networks.* ICLR 2025, arXiv:2410.02596.
  - **Idea.** A residual t = log(target) − log(sampler) is penalized by g(t), and on-policy the gradient is that of an f-divergence. Some rows of their table:

    | g(t) | divergence | behaviour |
    |---|---|---|
    | t²/2 | reverse KL | zero-forcing |
    | Linex(1), e^t − t − 1 | forward KL | zero-avoiding |
    | Linex(1/2) | α = 1/2 | neither |

  - **Results.** Zero-avoiding losses found more modes and gave more diverse samples; zero-forcing losses gave higher reward.
- **f-Trajectory Balance.** Fawkes, Hartford. *f-Trajectory Balance: A Loss Family for Tuning GFlowNets, Generative Models, and LLMs with Off- and On-Policy Data.* ICML 2026, arXiv:2605.15417.
  - It derives the same family, with α = 0 giving the forward KL, which is Linex(1). It shows that off-policy training keeps the minimizer, and gives a tempered form L(βΔ)/β.
  - On a hypergrid, forward-KL and Hellinger losses found modes faster than TB.
- **Here.** The loss is free, and forward KL is the score we lose on most. **Implemented** as `loss_function=linex` with `loss_shape` = a, as g(d) = 2(e^{−a d} + a d − 1)/a² on d = −t, the left side of a SubTB condition minus its right side.
  - The tempering matters. At initialization, a pair of a miss prefix and a later hit prefix has d ≈ log ε ≈ −18.4, and e^{18.4} ≈ 10^8 would swamp every other term. With a = 0.25 that pair weighs about 20 times what it does under the square.
- **Other loss variants.** These are known and not tried:
  - SubTB(λ) (Madan et al., ICML 2023, arXiv:2209.12782): the paper's runs weigh all pairs alike, λ = 1, as Hu et al. do.
  - VarGrad / log-partition variance (Richter et al., NeurIPS 2020, arXiv:2010.10436; Zhang et al., ICLR 2023, arXiv:2302.05446).
  - Divergence-based losses with control variates (da Silva, da Silva, Mesquita, NeurIPS 2024, arXiv:2410.09355): their forward-KL estimator is importance-weighted and had high variance on sequences.
  - Relative TB (Venkatraman et al., NeurIPS 2024, arXiv:2405.20971): this is TB with the base model as prior, which is already this method's reward.
  - FlowRL (Zhu et al., ICLR 2026, arXiv:2509.15207): it length-normalizes, which changes the target, so it is not suitable for exact posteriors.
- **RapTB.** Wang, Lu, Wang. *Rooted Absorbed Prefix Trajectory Balance with Submodular Replay.* arXiv:2603.00454, 2026.
  - It reports that SubTB with stop-probability flows, as in Hu et al., can drift: the stop log-probabilities sink and validity collapses.
  - Its fix is TB plus rooted-prefix residuals and a diversity-aware buffer.
  - This is relevant to the mode collapse above. It is not tried here because it is a larger change.

### Exploration

- **Local Search GFlowNets.** Kim, Yun, Bengio, Zhang, Bengio, Ahn, Park. ICLR 2024, arXiv:2310.02710.
  - Backtrack half a trajectory and rebuild it with the policy, keeping the result if its reward is higher. The default is 7 iterations of 4 candidates.
  - Flow of Reasoning applies it to LMs as suffix regeneration; removing it cost 25 points there.
  - **Here.** Each refinement pays a generation of the suffix and a p-scoring of the new response, so roughly 2 inferences per candidate, the same as a fresh response. It is a different way to spend the same budget, not a free gain. It could help the rare tasks, where a run otherwise never finds a hit, by starting from a known one. **Implemented** as `local_search_probability`: one restart from a random prefix of a known hit, without the accept/reject step, since SubTB trains on every response whatever its reward.
- **Temperature-conditional GFlowNets.** Kim, Ko, Yun, et al. *Learning to Scale Logits for Temperature-Conditional GFlowNets.* ICML 2024, arXiv:2310.02823. Conditions the policy on a reward temperature. Here the target temperature is fixed at 1 and the sampling temperature is already U[0.5, 2] for 2/3 of fresh groups.
- **Thompson-sampling GFlowNets.** Rector-Brooks et al. ICML 2023 workshop, arXiv:2306.17693. Uses an ensemble of policy heads: about 15% more compute and 60% more modes on bit sequences.
- **Adaptive teachers.** Kim et al. ICLR 2025, arXiv:2410.01432. Trains a second sampler that targets high-loss regions. Its generations are charged too, so it costs a second model's draws.
- **Prioritized replay.** Shen et al., ICML 2023, arXiv:2305.07170: 50% of a batch from the top 10% by reward. The present buffer is all top-50, so it is already more greedy than this.
- **Reward-temperature annealing.** Hu et al. anneal the reward temperature in some tasks. Here that would mean annealing epsilon, or a power of p. Not tried.

## What was implemented

Both additions are settings, off by default. At their defaults the training run's `settings.json` is the paper's, so existing training runs continue unchanged. Tests are in `methods/tests/gfn/test_followups.py`.

| Setting | Default | Meaning |
|---|---|---|
| `corpus_budget_fraction` | 0 | before the first update, buy n = ⌊f · min(training_budgets) / (1/p̂(S) + 1)⌋ corpus hits at random positions of the task's hit sequence |
| `corpus_probability` | 0 | a group is drawn uniformly from those hits with this probability; otherwise the group is replayed or fresh as before |
| `loss_function` | `square` | `linex`: penalize each pairwise SubTB residual d with 2(e^{−a d} + a d − 1)/a² in place of d² |
| `loss_shape` | 1.0 | a, for `linex` |
| `local_search_probability` | 0 | once a hit is known (a bought corpus hit or a hit in the buffer), a group is a local-search group with this probability: it keeps the first k content tokens of one such hit, k uniform in [0, its content length), and draws the group's 20 responses as that prefix followed by the policy's continuation at temperature 1 |
| `objective` | `subtb` | `vargrad`: the loss of a group is Σ_k g(d_k) / K with d_k = log Ẑ + log q(y_k) − log R(y_k) and log Ẑ = mean_k (log R(y_k) − log q(y_k)), held fixed (`vargrad.py`); g is the square or Linex. No prefix flows and no stop-token terms |

The costs follow the rules of `cost.py`, and everything is charged to the ledger:

- **Buying a corpus hit:** 1/p̂(S) inferences (`CORPUS_HITS`, charged by the corpus access), plus 1 for p of its prefixes (`P_SCORING`).
- **Training on a corpus hit:** 1 inference for the loss's forward pass (`TRAINING_FORWARD`) and 1 budget-only pass for the backward, the same as a replayed response.

At f = 0.1 and B = 2^20, a task with p̂(S) = 0.0088 buys 912 hits for about 10% of the budget. A task with p̂(S) = 8 × 10^-4 buys 82 hits. The rarest task buys one.

- **A local-search response:** charged as a whole fresh response, 2 inferences and 5 passes, although only its continuation is generated. It is offered to the buffer like a fresh response.

The Linex loss costs nothing. The residuals between two stop tokens' flows have no direction and stay squared.

- **The VarGrad objective** costs what SubTB costs: one teacher-forced forward and one backward per response, under the same charges. It needs only log p(y) of each response, which the p-scoring already gives, so nothing is added.

## Fairness

The added settings are held to the rules every method follows:

- **Everything is charged.** Every inference goes through the run's ledger under the categories of `docs/cost_accounting.md`. Corpus hits are bought through the benchmark's own corpus access at 1/p̂(S) each, as SFT buys them.
- **No evaluation data.** The corpus access leaves out the hits that equal a reference, as it does for every method. A setting reads nothing but the task table, which every method may see; p̂(S) is in it.
- **Settings are chosen fairly.** The runs below are on benchmark tasks and are exploration. Settings for a claim about the benchmark are to be chosen on the paper's two tuning pairs (`wc_0346/goblin_mention`, `wc_1852/chinese_character`), as the paper's were. The 21 tasks are then run once at those settings.
- **Costs are compared fairly.** Comparisons with other methods are at matched construction inferences. A corpus variant spends more inferences per budget pass than the baseline.

## Results

Six runs were made on Mila (one A100-80GB each) from commit `1824091`, at B = 2^20 (`--set budgets=1048576 --set update_caps=`), replicate 0, with the same seeds:

- **Baseline:** no added settings.
- **Corpus:** `--set corpus_budget_fraction=0.1 --set corpus_probability=0.25`. This bought 912 hits on `wc_0670` and 82 on `wc_1853`, about 1.04 × 10^5 inferences each.
- **Linex:** `--set loss_function=linex --set loss_shape=0.25`.

Each run took about 4.5–5 h of wall-clock time. Two runs on the `long` partition were preempted and resumed from their checkpoints.

### `gflownet` (the sampler)

The paper's value for the same task and replicate is in parentheses. Lower is better for both KLs.

| Task | Run | Capped forward KL | Reverse KL | Native hit rate | Construction inferences |
|---|---|---|---|---|---|
| `wc_0670/chinese_character` | baseline | 10.76 (10.46) | 3.05 (3.09) | 0.966 (0.967) | 431,814 (431,599) |
| | corpus | **2.84** | **2.04** | 0.775 | 503,595 |
| | linex | 3.97 | 3.52 | 0.371 | 431,006 |
| `wc_1853/goblin_mention` | baseline | 5.85 (5.91) | 3.03 (3.10) | 0.970 (0.945) | 432,065 (432,368) |
| | corpus | **3.70** | 2.72 | 0.942 | 503,444 |
| | linex | 4.25 | **2.70** | 0.902 | 431,912 |

For reference, at the same budget the paper's SFT has a forward KL of 1.40 and a reverse KL of 1.03 on `wc_0670`, and 2.21 and 2.65 on `wc_1853`. At twice the budget (B = 2^21, about 8.65 × 10^5 inferences), the paper's GFlowNet has a forward KL of 10.35 on `wc_0670` and 4.74 on `wc_1853`.

The baselines were also run on rorqual (H100) as a cross-cluster check:

| Task | Forward KL | Reverse KL | Hit rate |
|---|---|---|---|
| `wc_0670` | 10.65 | 3.05 | 0.960 |
| `wc_1853` | 5.89 | 3.02 | 0.961 |

So the run-to-run spread at a fixed seed is about 0.1 in forward KL.

### `gflownet+composition`

| Task | Baseline | Corpus | Linex | Paper `gfn+composition` | Paper `sft+composition` |
|---|---|---|---|---|---|
| `wc_0670` forward / reverse KL | 3.24 / 2.65 | **1.58 / 1.50** | 2.95 / 2.67 | 3.22 / 2.83 | 1.40 / 1.03 |
| `wc_1853` forward / reverse KL | 3.24 / 2.02 | **1.16 / 1.11** | 2.38 / 1.51 | 3.37 / 1.90 | 1.77 / 2.17 |

### Verdict

**Corpus replay helps a lot on both tasks, at about 17% more inferences.**

- **Scores.**
  - Forward KL falls from 10.76 to 2.84 on `wc_0670` and from 5.85 to 3.70 on `wc_1853`.
  - Reverse KL also improves.
  - On `wc_0670` the hit rate falls from 0.97 to 0.78, because the sampler now spreads mass beyond its few modes. On `wc_1853` it barely changes.
- **Cost.**
  - The pass budget is the same, but construction inferences rise from about 432k to about 504k. Bought hits cost one inference per pass, while fresh generations cost 2 inferences per 5 passes.
  - Extra compute does not explain the gain: the baseline at twice the budget barely moves (forward KL 10.35 and 4.74).
- **With composition.** Forward KL is 1.58 and 1.16, near or better than SFT+composition.
- **Where it cannot help.** On the rarest tasks, 10% of the budget buys about one hit. Those are the tasks where GFlowNet falls back to a forward KL of 50.

**Linex(0.25) is free and cuts forward KL, but it trades off hit rate on one task.**

- Forward KL falls to 3.97 and 4.25, and reverse KL improves on `wc_1853`.
- On `wc_0670`, reverse KL worsens from 3.05 to 3.52 and the hit rate drops from 0.97 to 0.37: the zero-avoiding loss also puts mass on misses.

**Caveats.**

- This is one replicate of two tasks, both common enough that GFlowNet finds hits. These are benchmark tasks, not tuning pairs.
- The values f = 0.1, p = 0.25 and a = 0.25 were chosen before any run and were not tuned. Choosing them on these results would be tuning on the test set.
- The gains, 2 to 8 in forward KL, are far above the 0.1 spread between the two baseline runs. They are still not a benchmark claim.

**Next steps:** see the end of the rare-task section below.

## Rare tasks

GFlowNet falls back on the three rarest tasks at every budget up to 2^21 in the paper. Their forward KL is 50 and their reverse KL 79–150. They are 3 of the 21 tasks, so on their own they add about 7 (3 × 50 / 21) to the method's mean forward KL. Six runs were made on rorqual (one H100 each) from commit `1824091`, at B = 2^20, replicate 0, with `corpus_probability=0.25` and `corpus_budget_fraction` f = 0.1 or 0.3.

| Task | p̂(S) | f | Hits bought | Native hit rate | Forward KL | Reverse KL | Composition forward / reverse KL | Construction inferences |
|---|---|---|---|---|---|---|---|---|
| `wc_0842/word_repetition` | 1.80e-5 | 0.1 | 1 | 0.000 | 50 (fallback) | 106.73 | 50 / 106.73 | 476,382 |
| | | 0.3 | 5 | **0.171** | **7.91** | **7.54** | 7.91 / 7.54 | 603,211 |
| `wc_1301/chatgpt_mention` | 1.02e-5 | 0.1 | 1 | 0.000 | 50 (fallback) | 150.34 | 50 / 150.34 | 500,225 |
| | | 0.3 | 3 | **0.531** | **7.94** | **4.22** | **6.33 / 3.62** | 611,644 |
| `wc_1802/contrastive_phrasing` | 1.47e-5 | 0.1 | 1 | 0.000 | 50 (fallback) | 79.05 | 50 / 79.05 | 482,330 |
| | | 0.3 | 4 | 0.002 | 50 (fallback) | 79.05 | 50 / 79.05 | 601,153 |

On `wc_0842` the composition step kept the plain sampler (one inference per response), so the two rows are equal.

For comparison, the best of the paper's other methods on the same tasks and replicate:

| Method (paper) | Construction inferences | `wc_0842` forward / reverse KL | `wc_1301` forward / reverse KL | `wc_1802` forward / reverse KL |
|---|---|---|---|---|
| `mixture-eb16+composition` K = 16384 | 370k–417k | 11.03 / 11.14 | 8.01 / 5.35 | 16.28 / 16.66 |
| `mixture-eb16+composition` K = 32768 | 734k–826k | 11.40 / 10.58 | 8.86 / 5.10 | 14.93 / 12.00 |
| `prompt-ebjc+composition` K = 32768 | 729k–821k | 11.77 / 10.06 | 9.37 / 5.08 | **12.58 / 9.80** |
| `sft+composition` B = 2^20 | 914k | 34.88 / 68.57 | 7.74 / 7.81 | 18.14 / 22.86 |
| `gfn+composition` B = 2^22 | 1.73M | 8.93 / 3.22 (mean of 3) | 21.70 / 52.46 (mean of 3) | 36.30 / 53.98 (mean of 3) |
| **corpus f = 0.3** (this run) | 601k–613k | **7.91** / 7.54 | **6.33 / 3.62** | 50 / 79.05 |

**Verdict.**

- **Ten percent is not enough.** At f = 0.1 the run buys one hit. The sampler then never produces a hit on its own, and the result is the fallback as before.
- **Thirty percent clears the fallback on two of the three tasks.** With 3–5 bought hits, `wc_0842` and `wc_1301` go from 50 to a forward KL of about 6–8. That is better than every method in the paper on those tasks at up to 914k inferences, and better than the paper's GFlowNet at four times the budget.
- **`wc_1802` still fails.** Four bought hits gave 2 hits in 1,024 evaluation draws. The bought hits alone do not teach the sampler this task with the plain loss. With Linex added, they do (next section).
- **Cost.** These runs spend about 600k construction inferences against about 432k for the baseline, and about 40% more than `mixture-eb16+composition` K = 16384. The fair comparisons are with methods at 600k–900k, as in the table.
- **What it means for the benchmark, roughly.** With composition, the rare-task mean would be (7.91 + 6.33 + 50) / 3 = 21.4 in forward KL and (7.54 + 3.62 + 79.05) / 3 = 30.1 in reverse KL, against 50 and 112 now. That is under the about 24 / 36 that GFlowNet would need to have the best mean in the paper. This is a projection from one replicate, not a measurement; see the caveats.

**Caveats.**

- **f = 0.3 was chosen after looking at rare-task results.** f = 0.1 was fixed in advance. f = 0.3 was added as a second arm on the same rare tasks and is the arm that worked. Reporting f = 0.3 as the method's setting would be tuning on the test set. A fair version needs a rule fixed before the runs. One candidate is to set f per task from p̂(S), which the task table gives every method, so that it buys a fixed number of hits. That rule would need to be set on the tuning pairs, which are common tasks, so it is an extrapolation.
- **One replicate.** With 1–5 bought hits, which hits are bought is luck, and replicates 1 and 2 may differ a lot.
- **Non-rare tasks at f = 0.3 are untested.** The common-task results above are at f = 0.1.

### Corpus replay with Linex, and a higher replay probability

Eight more runs were made on rorqual at B = 2^20, replicate 0, from commit `1824091` or later. The code paths the runs use did not change between those commits. Two arms were run on each rare task at f = 0.3: `corpus_probability=0.5`, and `corpus_probability=0.25` with `loss_function=linex` (a = 0.25). On the two common tasks, f = 0.1 was combined with Linex.

Rare tasks, forward / reverse KL. Each run bought the same hits as its f = 0.3 run above, since the seeds are the same:

| Task | Corpus f = 0.3, p = 0.25 | p = 0.5 | **p = 0.25 + Linex** | Linex native hit rate | Linex construction inferences |
|---|---|---|---|---|---|
| `wc_0842` sampler | 7.91 / 7.54 | 8.16 / 7.18 | **7.68 / 7.19** | 0.237 | 603,200 |
| `wc_0842` composition | 7.91 / 7.54 | 8.16 / 7.18 | **7.68 / 7.19** | | 603,868 |
| `wc_1301` sampler | 7.94 / 4.22 | 7.81 / 4.21 | **5.44 / 4.16** | 0.396 | 611,613 |
| `wc_1301` composition | 6.33 / 3.62 | 7.81 / 4.21 | 7.87 / 3.71 | | 612,461 |
| `wc_1802` sampler | 50 / 79.05 (fallback) | 50 / 79.05 (fallback) | **6.06 / 5.45** | 0.743 | 599,696 |
| `wc_1802` composition | 50 / 79.05 (fallback) | 50 / 79.05 (fallback) | **7.40 / 3.68** | | 600,961 |

Common tasks, forward / reverse KL:

| Task | Corpus f = 0.1 | **Corpus f = 0.1 + Linex** | Linex native hit rate |
|---|---|---|---|
| `wc_0670` sampler | 2.84 / 2.04 | 2.53 / 2.78 | 0.209 |
| `wc_0670` composition | 1.58 / 1.50 | 1.81 / 2.01 | |
| `wc_1853` sampler | 3.70 / 2.72 | 3.66 / 2.79 | 0.923 |
| `wc_1853` composition | 1.16 / 1.11 | **0.93 / 1.10** | |

**Verdict.**

- **Corpus replay with Linex clears the fallback on all three rare tasks.**
  - On `wc_1802`, the task that failed with the plain loss, it gives a forward KL of 6.06 and a hit rate of 0.74 from the same four bought hits.
  - The zero-avoiding loss seems to be what lets the sampler spread out from a handful of hits, rather than collapse onto them or lose them.
  - A higher replay probability (p = 0.5) does not do this.
- **The three rare tasks with composition** average (7.68 + 7.87 + 7.40) / 3 = 7.65 in forward KL and (7.19 + 3.71 + 3.68) / 3 = 4.86 in reverse KL. The paper's GFlowNet has 50 and 112 on them.
  - The sampler alone averages 6.39 / 5.60.
  - With composition, it beats every method in the paper at up to 914k inferences on each rare task and both KLs. The one exception is forward KL on `wc_1301`, where `sft+composition` B = 2^20 has 7.74 against 7.87.
- **On the common tasks, Linex on top of corpus replay is mixed.**
  - On `wc_0670` it lowers the sampler's forward KL a little, but worsens reverse KL and the composition scores. As in the Linex-only run, the hit rate falls (0.21).
  - On `wc_1853` it improves composition a little.
- **A rough benchmark projection.**
  - Take the paper's `gfn+composition` scores on the 18 other tasks (5.22 / 4.92 at 432k) and add these three: the 21-task mean would be about (18 × 5.22 + 3 × 7.65) / 21 = 5.57 in forward KL and 4.91 in reverse KL.
  - At B = 2^20, the best paper methods are `mixture-eb16+composition` K = 16384 at 7.87 / 9.39 and `sft+composition` B = 2^19 at 8.48 / 12.22.
  - At four times the budget, `sft+composition` B = 2^21 has 5.91 / 7.34.
  - This is a projection, not a measurement; the caveats below are serious.

**Caveats, which apply with more force now.**

- **Arms were added after seeing results, on the test tasks.** Each rare task has now had four arms: f = 0.1, f = 0.3, f = 0.3 with p = 0.5, and f = 0.3 with Linex. f = 0.3 and the corpus+Linex combination were both chosen after an earlier arm failed. The best of four arms on one replicate is biased downward. The combination itself was planned before batch 1 finished (FOLLOWUPS next steps), and a = 0.25 was fixed before any run. Still, these numbers are exploration, not a claim.
- **The 18 other tasks were not run with these settings.** On `wc_0670`, Linex hurts some scores. Nothing is known about f = 0.3 on common tasks, where it spends far more inferences.
- **One replicate.** The 3–5 bought hits are a lucky or unlucky draw. Replicates 1 and 2 buy different hits.
- **Cost.** The rare-task runs spend about 600k construction inferences against the baseline's 432k.

**What a fair claim needs.**

1. Fix a rule in advance. For example, set f per task from p̂(S) to buy a fixed number of hits, with a floor such as f = 0.1, and use corpus replay with Linex(0.25) everywhere.
2. Check the rule on the tuning pairs only.
3. Run all 21 tasks × 3 replicates once with it, and compare at matched construction inferences.

**Local search.** The `local_search_probability` setting was smoke-tested on `wc_1301` (f = 0.3, local 0.25, 8 updates). 580 of its 920 local-search responses were hits (63%), against 0 of 1,680 fresh responses. A few bought hits become many new and different hits to train on. No full run has been made.

**Next steps** (none launched; each needs approval):

- A rule for f from p̂(S), with corpus replay and Linex, fixed on the tuning pairs (`wc_0346`, `wc_1852`): a few runs, about 10–20 GPU-hours.
- A full 21 × 3 run at B = 2^20 with the fixed rule: about 63 runs × 5 GPU-hours ≈ 300 GPU-hours. This is the run that would support a claim.
- Local search on the rare tasks at f = 0.1, to test whether it removes the need for a large f: 3 runs, about 9 GPU-hours. It is less urgent now that corpus replay with Linex clears all three tasks at f = 0.3.

## VarGrad trajectory balance in place of SubTB

The `objective=vargrad` setting replaces SubTB with VarGrad trajectory balance (`vargrad.py`): one residual per response against the group's mean log R − log q, with no prefix flows. It costs the same. Seven runs were made on Mila (one A100-80GB each) from commit `0bad7cd`, at B = 2^20, replicate 0:

- **Tuning pairs:** the baseline (SubTB, no added settings) and `--set objective=vargrad`, on `wc_0346/goblin_mention` and `wc_1852/chinese_character`. These are the pairs settings may be chosen on.
- **Rare tasks:** VarGrad with the settings of the best SubTB arm above (`corpus_budget_fraction=0.3`, `corpus_probability=0.25`, Linex a = 0.25). Each run bought the same hits as its SubTB counterpart.

Tuning pairs, forward / reverse KL, with the native hit rate. Construction inferences were 431k–433k in every run:

| Task | SubTB sampler | VarGrad sampler | SubTB composition | VarGrad composition |
|---|---|---|---|---|
| `wc_0346` | **5.06 / 2.65**, 0.755 | 5.94 / 2.96, 0.948 | 5.86 / 2.96 | **3.19 / 2.36** |
| `wc_1852` | **5.62 / 2.91**, 0.743 | 7.50 / 2.77, 0.979 | **3.80 / 2.81** | 5.52 / 3.10 |

Rare tasks, forward / reverse KL, against SubTB with the same settings (previous section):

| Task | SubTB sampler | VarGrad sampler | SubTB composition | VarGrad composition | VarGrad construction inferences |
|---|---|---|---|---|---|
| `wc_0842` | **7.68 / 7.19** | 8.59 / 7.82, 0.232 | **7.68 / 7.19** | 8.72 / 8.14 | 603,094 |
| `wc_1301` | **5.44 / 4.16** | 23.30 / 20.80, 0.200 | **7.87 / 3.71** | 12.22 / 14.33 | 611,519 |
| `wc_1802` | **6.06 / 5.45** | 50 / 79.05 (fallback), 0.013 | **7.40 / 3.68** | 50 / 79.05 (fallback) | 601,152 |

**Verdict: VarGrad is simpler but not better here. SubTB stays the default.**

- **On the tuning pairs, VarGrad sharpens the sampler.**
  - Its hit rate rises to 0.95–0.98, but its forward KL is worse (5.94 vs 5.06, and 7.50 vs 5.62). It puts more mass on hits, but on fewer of them.
  - Composition splits: VarGrad is much better on `wc_0346` (3.19 vs 5.86) and worse on `wc_1852` (5.52 vs 3.80).
  - During training, VarGrad's selection hits (about 110–125 of 128) were steadier than SubTB's (about 85–95).
- **On the rare tasks, VarGrad is worse everywhere and falls back on `wc_1802`.**
  - `wc_0842` and `wc_1301` peaked at 31 and 58 selection hits (of 128) around update 224, then fell to 0 from about update 320. The submission rule picked the earlier checkpoints, so the runs did not fall back.
  - `wc_1802` never had a selection hit.
- **A likely reason, not tested.** With all-miss groups, every log R is log p + log ε, so log R − log q is nearly constant when q ≈ p. The group-mean baseline then gives almost no signal: the smoke run's loss at update 1 was about 5 × 10^-10. SubTB's per-prefix terms and learned prefix flows give signal in such groups. On the rare tasks, nearly all fresh groups are all-miss, and the signal comes from a few repeated bought hits. With a group-local log Ẑ, that signal seems to sharpen q onto those hits and then lose them.
- **Fairness.** Charges are identical to SubTB, and so are the construction inferences. The tuning-pair comparison is the one settings may be chosen on. On those pairs, VarGrad does not win on the sampler's KLs, so choosing it would not be justified.

**Caveats.** One replicate. The rare-task arm reuses settings chosen for SubTB, so a VarGrad-specific setting (for example a learned log Z, or a lower `loss_shape`) might do better. Such a setting would have to be chosen on the tuning pairs.
