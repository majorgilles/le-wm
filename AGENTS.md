# AGENTS.md

## Work in WSL

This working tree is `~/dev/le-wm` inside WSL. The checkout under
`C:\Users\giloz\dev\le-wm` is obsolete. Keep the repo and `$STABLEWM_HOME` on
the WSL filesystem: training reads a large HDF5 file with six data-loader
workers, and `/mnt/c` is much slower.

SSH pushes currently fail because `~/.ssh/id_ed25519` is not registered with
the GitHub account.

## Commands

```bash
uv sync
uv run nbdev-export
./tools/run_nbs.sh 00_module 01_jepa 02_utils
uv run ty check --exit-zero-on-warning
uv run python train.py data=pusht
uv run python eval.py --config-name=pusht.yaml policy=random
```

`ty` is advisory because PyTorch, Hydra, and OmegaConf expose partly untyped
APIs. Investigate warnings that identify a real missing API; do not add casts
only to silence third-party typing gaps.

## Source of truth

The notebooks are canonical and nbdev exports model code:

- `nbs/00_module.ipynb` → `lewm/module.py`
- `nbs/01_jepa.ipynb` → `lewm/jepa.py`
- `nbs/02_utils.ipynb` → `lewm/utils.py`
- `nbs/03_tutorial_pusht.ipynb` is the end-to-end tutorial and is not exported

Edit notebooks, then run `uv run nbdev-export`. Treat `lewm/*.py` and
`lewm/_modidx.py` as generated files. `lib_path` must stay `lewm`; setting it
to `.` makes nbdev walk `.venv` and create `__init__.py` files in installed
packages.

`train.py` and `eval.py` remain scripts because their `@hydra.main` entry
points require a real command line. Import model code as `lewm.*`; duplicate
root modules are intentionally absent.

Notebook outputs are committed so GitHub readers can see the checks and figures
without a local GPU. Preserve existing outputs; do not install an output-cleaning
hook. Notebook execution and refreshed outputs are user-run.

## Data and checkpoints

`$STABLEWM_HOME` defaults to `~/.stable_worldmodel`. PushT training and the
tutorial read `$STABLEWM_HOME/pusht_expert_train.h5` directly through
stable-worldmodel's format registry. `LOCAL_DATASET_DIR` can override the
training dataset directory.

The Hugging Face checkpoint predates Transformers 5. Its ViT keys must be
remapped before strict loading. Keep the working conversion in `README.md` and
the tutorial aligned.

## Dependency constraints

Dependencies use uv; commit both `pyproject.toml` and `uv.lock` when they
change. Keep these constraints unless their original problem has been
re-tested:

- `box2d-py` is disabled because the used environments do not need Box2D and
  Python 3.10 has no wheel.
- `datasets>=2.20.0` prevents an incompatible backsolve to 1.1.1.
- uv resolves only win32, Linux, and Darwin to avoid Pyodide backsolves.
- Linux and Windows use PyTorch's cu130 index; macOS uses PyPI CPU wheels.

Verified in WSL: PyTorch 2.14.0+cu130, Triton 3.8.0, and CUDA on an RTX 4070
SUPER.

## Verification

Notebook execution is user-run: do not execute notebooks or start GPU training
unless the user explicitly asks. After model or notebook changes:

1. Export with `uv run nbdev-export` only when the user permits command execution.
2. Ask the user to execute the affected notebook with `./tools/run_nbs.sh <name>`.
3. Run `uv run ty check --exit-zero-on-warning` only when permitted; inspect every warning.
4. Confirm a second export produces no diff when command execution is permitted.

The full tutorial needs the 46 GB PushT dataset and pretrained checkpoint. With
`RUN_TRAINING = False`, its expected planning result is expert rank `1/64` for
a reachable goal and about `4/64` for a goal beyond the horizon. Real training
is deliberately opt-in.
