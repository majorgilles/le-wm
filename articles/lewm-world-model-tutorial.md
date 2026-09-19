# Building a World Model That Predicts Without Reconstructing Pixels

> A beginner-friendly walkthrough of LeWorldModel, JEPA-style prediction, representation collapse, and planning in latent space.

World models try to answer a useful question: **if an agent takes an action, what happens next?**

The obvious approach is to predict the next image. But exact image prediction is expensive. A model must spend capacity reproducing textures, shadows, and other pixel-level details even when they do not matter for choosing an action.

[LeWorldModel (LeWM)](https://arxiv.org/pdf/2603.19312v1) takes a different route inspired by the Joint Embedding Predictive Architecture, or JEPA, advocated by Yann LeCun. Instead of reconstructing future pixels, it predicts a compact representation of the future.

I worked through the model in an executable tutorial and published the notebooks and code in my [LeWM GitHub repository](https://github.com/majorgilles/le-wm). This article explains what the experiment taught me.

## The central idea

A standard video predictor looks roughly like this:

```text
current pixels + action → predicted future pixels
```

LeWM instead learns this pipeline:

```text
current pixels ──encoder──▶ current embedding ──┐
                                                  ├──predictor──▶ future embedding
current action ──encoder──▶ action embedding ────┘

future pixels ──same encoder──▶ target future embedding
```

The predictor is trained to make its future embedding match the target embedding. It never needs to reconstruct the image.

In the PushT tutorial, each `224 × 224 × 3` image contains 150,528 pixel values. The encoder compresses it to only 192 values—a reduction of roughly **784×**.

This makes planning much cheaper: the model can imagine many candidate futures in a compact latent space rather than generating full images for every possibility.

## The dangerous shortcut: representation collapse

There is a catch.

The encoder is trained at the same time as the predictor. It could make prediction trivially easy by mapping every image to the same embedding:

```text
image A → [0, 0, 0, ...]
image B → [0, 0, 0, ...]
image C → [0, 0, 0, ...]
```

The predictor would then be perfectly accurate while learning nothing about the world. This failure mode is called **representation collapse**.

LeWM prevents it with the Sketch Isotropic Gaussian Regularizer, or `SIGReg`. The regularizer encourages the embedding distribution to resemble a standard Gaussian. A constant representation cannot satisfy that constraint.

The tutorial makes this visible by evaluating the regularizer on three kinds of embeddings:

| Embeddings | SIGReg loss |
|:--|--:|
| Perfect Gaussian samples | 1.08 |
| Pretrained model on real frames | 5.63 |
| Collapsed embeddings | 121.88 |

Lower is better. The collapsed representation is penalized heavily, while useful, varied embeddings are much closer to the Gaussian target.

The important lesson is not simply that “Gaussian embeddings are good.” The regularizer removes the easiest useless solution while still allowing prediction loss to teach the encoder which visual distinctions matter.

## What the model learns from pixels

The PushT task contains a movable T-shaped block, a target area, and a controllable pusher. The model receives pixels and actions—not the simulator's hidden state.

The dataset used by the tutorial contains:

- 18,685 expert episodes
- 2,336,736 image frames
- actions describing pusher movement
- simulator state used only for analysis and evaluation

After loading the pretrained model, I compared embeddings from related and unrelated frames:

| Comparison | Cosine similarity |
|:--|--:|
| Consecutive frames from one episode | +0.9956 |
| Frames from different episodes | −0.0304 |

Nearby moments map to nearby points, while unrelated situations are almost orthogonal. The embedding is not merely avoiding collapse; it has learned temporal structure from visual experience.

## Actions make it a world model

Encoding images alone produces a visual representation. A world model must also understand intervention: **what changes when I do this?**

LeWM embeds the action and feeds it into an autoregressive predictor. Given several observed embeddings and their actions, the predictor estimates the next embedding. That prediction can then be fed back into the model to imagine another step.

This creates a latent rollout:

```text
observed state
    ↓ action 1
predicted state 1
    ↓ action 2
predicted state 2
    ↓ action 3
predicted state 3
```

The model becomes less certain as it rolls farther beyond observed data. In the tutorial, latent prediction error grew from approximately `0.0031` at the first imagined step to `0.0582` after twelve imagined steps.

That is not just an implementation detail. It teaches an important planning principle: **the farther a model imagines into the future, the more its errors compound**.

## Planning by imagining candidate actions

Once the model can imagine outcomes, planning becomes surprisingly direct:

1. Sample many possible action sequences.
2. Roll each sequence forward through the world model.
3. Compare the final predicted embedding with the goal embedding.
4. Choose the action sequence with the lowest distance.

The model does not need labels saying which plan is good. It ranks plans by where it predicts they will lead.

The tutorial evaluates 64 candidates: one sequence of actions taken from an expert demonstration and 63 random alternatives. For a reachable goal, the expert sequence ranks **1st out of 64**.

Then the experiment deliberately moves the goal twice as far away while leaving the planning horizon unchanged. The expert sequence falls to roughly **4th out of 64**, and all plans have high cost.

This is a valuable sanity check. A planner cannot solve a goal beyond its horizon, no matter how accurate the world model is. If the task requires sixteen steps but the planner can only choose eight, a poor ranking does not automatically mean the model failed—the experimental setup may be impossible.

## Watching a model escape collapse

The tutorial also includes an optional five-minute training experiment. Before training, a freshly initialized model produces embeddings with a standard deviation of about `0.0011`: almost every frame maps to the same point.

A measured run on an RTX 4070 SUPER produced the following changes:

| Measurement | Before | After 1,501 steps |
|:--|--:|--:|
| Prediction loss | 0.2023 | 0.0384 |
| SIGReg loss | 10.52 | 1.75 |
| Embedding standard deviation | 0.0013 | 1.0369 |

Prediction improves while embedding variance increases by roughly 800×. This is the paper's core argument made concrete: SIGReg pulls the representation out of collapse without preventing the model from learning to predict.

Training is disabled by default because it is the expensive part of the tutorial. To opt in:

```python
RUN_TRAINING = True
MINUTES = 5
```

## Running the tutorial

The repository uses `uv` for dependencies and nbdev to export notebook code into the `lewm` Python package.

```bash
git clone https://github.com/majorgilles/le-wm.git
cd le-wm
uv sync
uv run jupyter lab nbs/03_tutorial_pusht.ipynb
```

The full PushT tutorial requires the dataset at:

```text
$STABLEWM_HOME/pusht_expert_train.h5
```

and the pretrained Hugging Face checkpoint at:

```text
$STABLEWM_HOME/hf_pusht/
```

Training remains off during the default run. The smaller notebooks can be read first without the full dataset:

1. `nbs/00_module.ipynb` — transformer blocks, action embeddings, and SIGReg
2. `nbs/01_jepa.ipynb` — encoding, prediction, rollout, and plan scoring
3. `nbs/02_utils.ipynb` — preprocessing, normalization, and checkpoints
4. `nbs/03_tutorial_pusht.ipynb` — the complete experiment on real data

## What I took away

The most useful lessons were broader than one architecture:

1. **A useful prediction target does not need to be human-readable.** Predicting compact embeddings can be enough for control.
2. **A low training loss can hide a useless solution.** Collapse demonstrates why objectives need to be examined, not merely optimized.
3. **Actions turn representation learning into world modeling.** The model learns not only what a scene looks like, but how it changes under intervention.
4. **Planning quality depends on horizon.** A model should not be blamed for failing an impossible planning setup.
5. **Small experiments expose the real mechanism.** Comparing Gaussian, learned, and collapsed embeddings made SIGReg easier to understand than the equation alone.

The complete implementation, executable notebooks, checkpoint-loading notes, and PushT walkthrough are available in my [GitHub repository](https://github.com/majorgilles/le-wm).

If you are learning about JEPAs or world models, I recommend running the notebooks in order and treating every printed number as a question: what would this value look like if the model had learned nothing?
