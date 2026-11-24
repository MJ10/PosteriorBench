# PosteriorBench

[📄 Paper](https://arxiv.org/abs/2611.25735) · [🤗 Dataset](https://huggingface.co/datasets/Transluce/PosteriorBench)

This repository contains the code for the paper **"PosteriorBench: Evaluating Behavior Posterior Sampling in Language Models"**.

Rare behaviors of language models are difficult to study, because finding one through ordinary sampling can take a very large number of samples. *Behavior posterior sampling* aims to efficiently sample from the model's distribution over transcripts that show a given behavior. PosteriorBench is a benchmark that compares methods by their compute cost and by how close their samples are to the true posterior.

To evaluate your own sampler, start with the [quickstart](#quickstart). To run the methods from the paper, see [`methods/`](methods).

> [!NOTE]
> PosteriorBench includes sensitive content: offensive language and racial slurs, model responses that discuss or encourage suicide and self-harm, and false statements.

## Setup

```sh
uv sync
```

To run experiments on the Mila and DRAC clusters with [cluv](https://github.com/mila-iqia/cluv), see [CLUSTERS.md](CLUSTERS.md).

Evaluation runs Qwen3.5-0.8B, so we recommend a GPU. [docs/protocol.md](docs/protocol.md#runtime) lists the installed versions and describes installation with pip.

## Quickstart

```python
import posteriorbench as pb

bench = pb.Benchmark()                           # the bundled task table and evaluation file
task = bench.task("wc_1853/goblin_mention")      # prompt, behavior, rule, p̂(S)
model = pb.OriginalModel.from_pretrained()       # downloads Qwen3.5-0.8B on first use

candidate = pb.instruction_sampler(model, task)  # a simple sampler; put yours here
result = bench.evaluate(task, candidate, model, ledger=pb.Ledger(), seed=0)
print(result.native_hit_rate, result.capped_forward_kl, result.reverse_kl)
```

This evaluates a simple baseline: the model with an instruction that asks for the behavior. The three printed numbers are the sampler's hit rate, capped forward KL, and reverse KL. The hit rate is the fraction of its responses that show the behavior. Lower KL is better. On an H100 it takes about a minute and a half and prints roughly `0.40 14.6 18.3`.

[`examples/01_quickstart`](examples/01_quickstart) goes through the same steps in a notebook.

## Evaluating your own sampler

A sampler has two methods. `sample` draws responses to the task's prompt, and `logprobs` returns the log-probability of given responses under the sampler:

```python
import posteriorbench as pb


class MySampler(pb.Sampler):
    name = "my-sampler"             # recorded in the result

    def sample(self, n, *, seed):   # n responses, each a list of generated token ids
        ...

    def logprobs(self, responses):  # log q(y) of each response; -inf if q cannot produce it
        ...
```

The `ledger` records the number of model inferences that a method uses to build its sampler. `bench.corpus(task, ledger)` provides posterior samples for training. Each sample costs $1/\hat{p}(S)$ inferences, where $\hat{p}(S)$ is the probability of the behavior. This is the expected cost of finding one by rejection sampling.

[`examples/02_custom_sampler`](examples/02_custom_sampler) contains a complete example. [docs/protocol.md](docs/protocol.md) describes how the scores are computed, and [docs/cost_accounting.md](docs/cost_accounting.md) describes how cost is counted.

## Running the benchmark

The full benchmark runs each of the 21 tasks three times. These 63 evaluations take about an hour and a half on one H100. Continuing from the quickstart:

```python
import json

with open("my_method.jsonl", "w") as out:
    for task in bench.tasks():  # the 21 tasks
        for r in (0, 1, 2):  # three replicates
            ledger = pb.Ledger()
            sampler = pb.instruction_sampler(model, task)  # build yours here and charge `ledger`
            result = bench.evaluate(task, sampler, model, ledger=ledger, seed=r, method="my-method")
            out.write(json.dumps(result.to_dict()) + "\n")
```

```sh
uv run posteriorbench aggregate my_method.jsonl --expect-tasks benchmark --expect-replicates 0,1,2
```

`aggregate` averages over replicates within a task and then over tasks, and prints the mean and standard error of each score. If fewer than 10% of a sampler's draws show the behavior, the sampler fails the task and receives the scores of the [fallback](docs/protocol.md#the-fallback) for it.

Reported results should include the construction cost together with both KLs, state any cost that is not in the ledger, and use the same evaluation settings for every task.

The same steps work from the command line (`posteriorbench <command> --help` lists the options):

```bash
uv run posteriorbench list-tasks
uv run posteriorbench evaluate --task wc_1853/goblin_mention --sampler instruction --out instruction_r0.json
uv run posteriorbench aggregate instruction_r0.json --expect-replicates 0
```

## Data

The [dataset](https://huggingface.co/datasets/Transluce/PosteriorBench) contains 50 million labeled responses from Qwen3.5-0.8B for each of 20 prompts (146 GB). Evaluating a sampler does not require downloading it.

[docs/data.md](docs/data.md) describes the files.

## Citation

```bibtex
@article{xun2026posteriorbench,
  title={PosteriorBench: Evaluating Behavior Posterior Sampling in Language Models},
  author={Zhiyang Xun and Daniel D. Johnson and Sarah Schwettmann and Neil Chowdhury},
  year={2026},
  eprint={2611.25735},
  archivePrefix={arXiv}
}
```

## License

The code is released under the MIT license ([LICENSE](LICENSE)). The two data tables in [`posteriorbench/dataset`](posteriorbench/dataset) are released under the Open Data Commons Attribution License 1.0 ([posteriorbench/dataset/LICENSE](posteriorbench/dataset/LICENSE)), the license of WildChat-1M, whose prompts the dataset uses.
