# Tools

Scripts that build dataset files. They are part of this repository, not of the installed package;
run them from a clone, in its environment.

| Script | What it does |
|---|---|
| `add_behavior.py` | builds a dataset directory for a new behavior on one of the prompts (references, fallback response, hit file) |
| `build_hits_index.py` | builds the hit files of one prompt from its corpus, leaving out the references |

`uv run python tools/<script>.py --help` lists a script's options.
