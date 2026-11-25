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
  - **Here.** Each refinement pays a generation of the suffix and a p-scoring of the new response, so roughly 2 inferences per candidate, the same as a fresh response. It is a different way to spend the same budget, not a free gain. It could help the rare tasks by keeping the suffix of a near-hit.
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

The costs follow the rules of `cost.py`, and everything is charged to the ledger:

- **Buying a corpus hit:** 1/p̂(S) inferences (`CORPUS_HITS`, charged by the corpus access), plus 1 for p of its prefixes (`P_SCORING`).
- **Training on a corpus hit:** 1 inference for the loss's forward pass (`TRAINING_FORWARD`) and 1 budget-only pass for the backward, the same as a replayed response.

At f = 0.1 and B = 2^20, a task with p̂(S) = 0.0088 buys 912 hits for about 10% of the budget. A task with p̂(S) = 8 × 10^-4 buys 82 hits. The rarest task buys one.

The Linex loss costs nothing. The residuals between two stop tokens' flows have no direction and stay squared.

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

**Next steps** (none launched; each needs approval):

- A corpus+Linex run.
- A sweep of f and a on the tuning pairs.
- A full 21 × 3 run at B = 2^20, about 63 runs × 5 GPU-hours ≈ 300 GPU-hours per variant.
