# AGENTS.md

Notes for agents and humans working on this repo.

## Environment: WSL, not Windows

**This project lives in WSL at `~/dev/le-wm` and should be worked on there.**
A copy may still exist at `C:\Users\giloz\dev\le-wm`; that is the older Windows
checkout and is no longer the working tree.

Keep the repo and `$STABLEWM_HOME` on the WSL filesystem, under `~`. Do not move
either onto `/mnt/c`. Crossing the Windows filesystem boundary is slow, and
training reads HDF5 datasets per batch with 6 dataloader workers
(see `config/train/lewm.yaml`), so the penalty is real.

## Packaging: uv

Dependencies are managed with uv. Both `pyproject.toml` and `uv.lock` are committed.

    uv sync                           # create or refresh the environment
    uv run python train.py data=pusht

`package = false` is set because the scripts are flat files at the repo root,
not an installable package.

`train.py` and `eval.py` are Hydra entry points using `@hydra.main`, which needs
a real command line. They are run as scripts, never imported.

## Dependency pins, and why

Each of these is also commented in `pyproject.toml`. Do not remove one without
re-checking that the reason still holds.

- **`box2d-py` overridden off every platform.** It ships no wheel for Python
  3.10 on any platform, so it builds from source and needs the SWIG binary. It
  arrives through `gymnasium[all]` inside `stable-worldmodel[env]`, but the only
  environments used here are PushT, Reacher, Cube and TwoRoom, none of them
  Box2D. Drop the override if a Box2D environment is ever added.
- **`datasets>=2.20.0` named explicitly.** `stable-pretraining` requires it, but
  without a direct dependency the resolver backsolved to version 1.1.1, which
  crashes on modern pyarrow because `pa.PyExtensionType` was removed.
- **`environments` restricted to win32, linux and darwin.** Otherwise uv also
  solves for Pyodide and emscripten, which is what allowed `datasets` to fall
  back to 1.1.1.
- **torch and torchvision from the cu130 index.** The default PyPI wheels are
  CPU-only. The cu130 build matches the local driver, CUDA 13.4, on an
  RTX 4070 SUPER. macOS has no CUDA and stays on the PyPI wheels.

## Verified working in WSL

    torch 2.14.0+cu130
    RTX 4070 SUPER, compute capability (8, 9)
    torch.cuda.is_available() -> True, GPU matmul OK
    triton 3.8.0
    all project modules import

`triton` has a Linux wheel but none for Windows, so `torch.compile` is available
here and was not on the Windows checkout.

## History of this setup

What was asked, in order, and what came of it:

1. **"convert this to a uv project"** - the repo had no packaging files at all.
   Added `pyproject.toml` and `uv.lock`, with dependencies inferred from the
   imports in the five scripts. Replaced the README's manual venv and pip steps
   with `uv sync`, and prefixed the documented run commands with `uv run`.
2. **`uv sync` failed on Windows** - four resolution problems surfaced in
   sequence: the lancedb Windows wheel, the box2d-py SWIG build, the datasets
   backsolve, and a CPU-only torch. Each is recorded above.
3. **"convert all code to jupyter notebooks and use nbdev"** - raised, not done.
   See the open item below.
4. **"i need to use cuda"** - torch had resolved to a CPU-only build. Pointed
   torch and torchvision at the cu130 PyTorch index and verified with a real GPU
   matmul rather than just the availability flag.
5. **"i should use WSL?"** - assessed. CUDA already worked natively, so WSL was
   not needed for that. The actual wins were wheel availability, triton, and
   forked rather than spawned dataloader workers.
6. **"move this project to wsl under ~/dev"** - committed the uv work, cloned
   from the Windows checkout into `~/dev/le-wm` preserving history, restored the
   GitHub remote, removed the Windows-only workarounds, then relocked and
   verified.

## Open item

Converting the code to Jupyter notebooks with nbdev was requested but not done.
The blocker is that `train.py` and `eval.py` use `@hydra.main`, which does not
work inside a notebook cell. The suggested scope was to convert only `jepa.py`,
`module.py` and `utils.py`, which are pure definitions and a clean nbdev fit,
and to leave the two Hydra entry points as scripts. Not yet decided.

Note that SSH to GitHub does not currently work from WSL. The key at
`~/.ssh/id_ed25519` is not registered with the account, so `git push` will fail
until a key is added.
