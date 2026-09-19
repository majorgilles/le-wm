
# LeWorldModel
### Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

> [!IMPORTANT]
> **Unofficial learning fork.** This repository is a substantially LLM-assisted educational rewrite of the [original LeWorldModel repository](https://github.com/lucas-maes/le-wm), created by a beginner trying to understand the project. It is not original research, is not maintained or endorsed by the paper's authors, and may contain mistakes. Use the [original paper](https://arxiv.org/pdf/2603.19312v1) and repository as the authoritative sources.

**Original research by:** [Lucas Maes*](https://x.com/lucasmaes_), [Quentin Le Lidec*](https://quentinll.github.io/), [Damien Scieur](https://scholar.google.com/citations?user=hNscQzgAAAAJ&hl=fr), [Yann LeCun](https://yann.lecun.com/) and [Randall Balestriero](https://randallbalestriero.github.io/)

**Abstract:** Joint Embedding Predictive Architectures (JEPAs) offer a compelling framework for learning world models in compact latent spaces, yet existing methods remain fragile, relying on complex multi-term losses, exponential moving averages, pretrained encoders, or auxiliary supervision to avoid representation collapse. In this work, we introduce LeWorldModel (LeWM), the first JEPA that trains stably end-to-end from raw pixels using only two loss terms: a next-embedding prediction loss and a regularizer enforcing Gaussian-distributed latent embeddings. This reduces tunable loss hyperparameters from six to one compared to the only existing end-to-end alternative. With ~15M parameters trainable on a single GPU in a few hours, LeWM plans up to 48× faster than foundation-model-based world models while remaining competitive across diverse 2D and 3D control tasks. Beyond control, we show that LeWM's latent space encodes meaningful physical structure through probing of physical quantities. Surprise evaluation confirms that the model reliably detects physically implausible events.

<p align="center">
   <b>[ <a href="https://arxiv.org/pdf/2603.19312v1">Paper</a> | <a href="https://huggingface.co/collections/quentinll/lewm">Checkpoints &amp; Data</a> | <a href="https://le-wm.github.io/">Website</a> ]</b>
</p>

<br>

<p align="center">
  <img src="assets/lewm.gif" width="80%">
</p>

If you find this code useful, please reference it in your paper:
```
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

## Start here

This codebase builds on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for environments, planning, and evaluation, and [stable-pretraining](https://github.com/galilai-group/stable-pretraining) for training. The model code is taught in three executable notebooks and exported by nbdev into the `lewm/` package.

Install the environment, then start with the small notebooks; they need no dataset or checkpoint:

```bash
uv sync
uv run jupyter lab nbs/00_module.ipynb
```

Read them in order: `00_module` builds the neural-network pieces, `01_jepa` assembles the world model, and `02_utils` explains data preprocessing and checkpoints.

For the complete model test, first follow [Data](#data) and [Pretrained Checkpoints](#pretrained-checkpoints), then open:

```bash
uv run jupyter lab nbs/03_tutorial_pusht.ipynb
```

In Jupyter, choose **Run → Run All Cells**. Training is off by default, so the tutorial uses the pretrained model and takes a few minutes on a CUDA GPU.

### What the tutorial tests—and teaches

A successful run shows:

- **Checkpoint integrity:** `with remap: strict load OK`. Older ViT checkpoint names are translated to the installed Transformers version.
- **Useful representations:** nearby frames have cosine similarity around `+0.996`, while frames from different episodes are near `0`.
- **Why SIGReg exists:** collapsed embeddings score far worse than Gaussian or learned embeddings. SIGReg prevents the trivial “everything maps to the same vector” solution.
- **World-model limits:** rollout error grows as the model imagines farther into the future.
- **Planning:** for a reachable goal, the recorded expert actions rank `1/64`; when the goal is beyond the planning horizon, no candidate can solve it reliably.

To watch a fresh model escape collapse, set these values in section 5 and rerun from that cell:

```python
RUN_TRAINING = True
MINUTES = 5
```

On an RTX 4070 SUPER, the measured five-minute run reduced prediction loss about 5× and raised embedding standard deviation from about `0.001` (collapsed) to `1.0`.

For a repeatable command-line check:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 nbs/03_tutorial_pusht.ipynb
```

The smaller notebooks explain and test each exported module:

```bash
./tools/run_nbs.sh 00_module 01_jepa 02_utils
uv run ty check --exit-zero-on-warning
```

### Repository map

| Path | Purpose |
|:---|:---|
| `nbs/` | Canonical, editable notebooks |
| `lewm/` | Python package generated by `uv run nbdev-export` |
| `train.py`, `eval.py` | Hydra command-line entry points |
| `config/` | Training and evaluation settings |

Edit the notebooks rather than generated files under `lewm/`.

## Data

Datasets use the HDF5 format for fast loading. Download the data from [HuggingFace](https://huggingface.co/collections/quentinll/lewm) and decompress with:

```bash
tar --zstd -xvf archive.tar.zst
```

Place the extracted `.h5` files under `$STABLEWM_HOME` (defaults to `~/.stable_worldmodel/`). You can override this path:
```bash
export STABLEWM_HOME=/path/to/your/storage
```

The PushT commands expect `$STABLEWM_HOME/pusht_expert_train.h5`. The training config includes the `.h5` extension and reads it directly; no conversion step is needed.

## Training

`nbs/01_jepa.ipynb` explains the model and nbdev exports it to `lewm/jepa.py`. Training is configured via [Hydra](https://hydra.cc/) files under `config/train/`.

Before training, set your WandB `entity` and `project` in `config/train/lewm.yaml`:
```yaml
wandb:
  config:
    entity: your_entity
    project: your_project
```

To launch training:
```bash
uv run python train.py data=pusht
```

Checkpoints are saved to `$STABLEWM_HOME` upon completion.

For baseline scripts, see the stable-worldmodel [scripts](https://github.com/galilai-group/stable-worldmodel/tree/main/scripts/train) folder.

## Planning

Evaluation configs live under `config/eval/`. Set the `policy` field to the checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix:

```bash
# ✓ correct
uv run python eval.py --config-name=pusht.yaml policy=pusht/lewm

# ✗ incorrect
uv run python eval.py --config-name=pusht.yaml policy=pusht/lewm_object.ckpt
```

## Pretrained Checkpoints

Pretrained LeWM checkpoints for each environment are mirrored on the Hugging Face
Hub (model repos), alongside the datasets (dataset repos) in the same collection:

- [`quentinll/lewm-pusht`](https://huggingface.co/quentinll/lewm-pusht)
- [`quentinll/lewm-cube`](https://huggingface.co/quentinll/lewm-cube)
- [`quentinll/lewm-tworooms`](https://huggingface.co/quentinll/lewm-tworooms)
- [`quentinll/lewm-reacher`](https://huggingface.co/quentinll/lewm-reacher)

The full baseline checkpoint suite (PLDM, LeJEPA, IVL, IQL, GCBC, DINO-WM, DINO-WM-noprop)
is available on [Google Drive](https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e):

<div align="center">

| Method | two-room | pusht | cube | reacher |
|:---:|:---:|:---:|:---:|:---:|
| pldm | ✓ | ✓ | ✓ | ✓ |
| lejepa | ✓ | ✓ | ✓ | ✓ |
| ivl | ✓ | ✓ | ✓ | — |
| iql | ✓ | ✓ | ✓ | — |
| gcbc | ✓ | ✓ | ✓ | — |
| dinowm | ✓ | ✓ | — | — |
| dinowm_noprop | ✓ | ✓ | ✓ | ✓ |

</div>

## Loading a checkpoint

### From the Drive archive

Each tar archive contains two files per checkpoint:
- `<name>_object.ckpt` — a serialized Python object for convenient loading; this is what `eval.py` and the `stable_worldmodel` API use
- `<name>_weight.ckpt` — a weights-only checkpoint (`state_dict`) for cases where you want to load weights into your own model instance

Place the extracted files under `$STABLEWM_HOME/` and load via:

```python
import stable_worldmodel as swm

# Load the cost model (for MPC)
cost = swm.policy.AutoCostModel('pusht/lewm')
```

`AutoCostModel` accepts:
- `run_name` — checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix
- `cache_dir` — optional override for the checkpoint root (defaults to `$STABLEWM_HOME`)

The returned module is in `eval` mode with its PyTorch weights accessible via `.state_dict()`.

### From the Hugging Face mirror

The HF model repos ship the LeWM checkpoint as a `weights.pt` (state dict) plus a
`config.json` describing the model. Convert once to produce the `_object.ckpt`
that `eval.py` expects:

```bash
# download weights.pt + config.json
hf download quentinll/lewm-pusht --local-dir $STABLEWM_HOME/hf_pusht

# convert to object checkpoint under $STABLEWM_HOME/pusht/lewm_object.ckpt
uv run python - <<'PY'
import json, re, torch, stable_pretraining as spt
from pathlib import Path
from lewm.jepa import JEPA
from lewm.module import ARPredictor, Embedder, MLP
import stable_worldmodel as swm

src = Path(swm.data.utils.get_cache_dir(), "hf_pusht")
out = Path(swm.data.utils.get_cache_dir(), "pusht", "lewm_object.ckpt")
cfg = json.loads((src / "config.json").read_text())
strip_hydra = lambda d: {k: v for k, v in d.items() if k != "_target_"}

encoder = spt.backbone.utils.vit_hf(
    cfg["encoder"]["size"],
    patch_size=cfg["encoder"]["patch_size"],
    image_size=cfg["encoder"]["image_size"],
    pretrained=False, use_mask_token=False,
)
mlp = lambda k: MLP(input_dim=cfg[k]["input_dim"], output_dim=cfg[k]["output_dim"],
                    hidden_dim=cfg[k]["hidden_dim"], norm_fn=torch.nn.BatchNorm1d)
model = JEPA(
    encoder=encoder,
    predictor=ARPredictor(**strip_hydra(cfg["predictor"])),
    action_encoder=Embedder(**strip_hydra(cfg["action_encoder"])),
    projector=mlp("projector"),
    pred_proj=mlp("pred_proj"),
)

# Transformers 5 renamed ViT internals used by this Transformers 4 checkpoint.
renames = [
    (r"^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.query\.", r"encoder.layers.\1.attention.q_proj."),
    (r"^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.key\.", r"encoder.layers.\1.attention.k_proj."),
    (r"^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.value\.", r"encoder.layers.\1.attention.v_proj."),
    (r"^encoder\.encoder\.layer\.(\d+)\.attention\.output\.dense\.", r"encoder.layers.\1.attention.o_proj."),
    (r"^encoder\.encoder\.layer\.(\d+)\.intermediate\.dense\.", r"encoder.layers.\1.mlp.fc1."),
    (r"^encoder\.encoder\.layer\.(\d+)\.output\.dense\.", r"encoder.layers.\1.mlp.fc2."),
    (r"^encoder\.encoder\.layer\.(\d+)\.", r"encoder.layers.\1."),
]
sd = torch.load(src / "weights.pt", map_location="cpu", weights_only=False)
for old in list(sd):
    for pattern, replacement in renames:
        new = re.sub(pattern, replacement, old)
        if new != old:
            sd[new] = sd.pop(old)
            break
model.load_state_dict(sd, strict=True)
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(model, out)
PY
```

After conversion, load via `swm.policy.AutoCostModel('pusht/lewm')` as usual.

## Contact & Contributions
Feel free to open [issues](https://github.com/lucas-maes/le-wm/issues)! For questions or collaborations, please contact `lucas.maes@mila.quebec`
