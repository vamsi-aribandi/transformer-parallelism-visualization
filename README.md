# Transformer Parallelism Visualizations

Animated videos explaining how transformers are parallelized in the **forward
pass**, in the sharding notation of the
[JAX scaling book](https://jax-ml.github.io/scaling-book/): `A[B_X, T, D]`,
partial sums `C[I, K]{U_X}`, `AllGather_X`, `ReduceScatter_{Y,D}`, `AllToAll_Z`.

Six single-strategy videos over a 2-layer transformer (simple MHA + MLP; the
MLP becomes a routed MoE for expert parallelism):

| video | strategy | forward-pass collectives |
|---|---|---|
| `dp` | Data parallelism — `In[B_X, T, D]`, replicated weights | none |
| `fsdp` | FSDP / ZeRO-3 — weights `W[D_X, F]` | AllGather per weight, just-in-time |
| `tp` | Tensor parallelism (Megatron) — `W_in[D, F_Y]`, `W_out[F_Y, D]` | AllGather in, ReduceScatter out, per block |
| `cp` | Context parallelism — `In[B, T_X, D]` | AllGather K, V in attention; MLP silent |
| `pp` | Pipeline parallelism — stages own layers, 4 microbatches | P2P activation sends (Gantt + bubble shown) |
| `ep` | Expert parallelism (MoE) — `W[E_Z, D, F]` | AllToAll dispatch + combine |
| `zero1` | ZeRO-1 — replicated weights, sharded optimizer | none forward; backward ReduceScatter → sharded step → AllGather |

## Recipes from frontier models

Three multi-axis configs play on the mesh-grid canvas (`scenes/five_d.py`):
`dense4d` (Llama 3 style FSDP × TP × CP × PP, 16 devices), `moe3d`
(DeepSeek-V3 / Kimi K2 style EP × PP × ZeRO-1, no TP, 8 devices) and `5d`
(Nemotron 3 / Qwen3-VL style, everything, 32 devices). `tests/test_recipes.py`
pins the first two.

## The 5D combination

`scenes/five_d.py` runs all five at once on 32 devices —
`Mesh({'X': 2, 'Y': 2, 'C': 2, 'Z': 2, 'stage': 2})`: FSDP over `X`, tensor
parallelism over `Y`, context parallelism over `C`, two experts on `Z`, two
pipeline stages. Activations are `In[B_{XZ}, T_C, D_Y]`, weights
`W_qkv[D_X, H_Y]`, experts `W_in[E_Z, D_X, F_Y]`. The step list is derived by
the same engine (no new choreography rules): 25 forward and 49 backward
collectives, each attributed to exactly one mesh axis. The canvas is one 4×4
device grid per stage (rows = X outer / C inner, columns = Z outer / Y inner),
so every axis is a fixed geometric relation between boxes; flights carry an
axis-colored halo and a badge names the parallelism responsible.
`tests/test_five_d.py` pins the per-block sequences. Render with
`make lq-5d` / `make hq-5d`.

## Forward + backward (training-step) videos

Each strategy also has a `*_train` scene showing the full training step:
activations saved to a per-station stash during the forward pass (the memory
cost), then rose gradient tensors flowing right→left — `dX = dY·Wᵀ` and
`dW = Xᵀ·dY` per matmul, so backward is visibly 2× the forward compute — with
each strategy's backward collectives (DP's gradient AllReduce, FSDP's weight
re-gather + gradient ReduceScatter, TP's mirrored AllGather/ReduceScatter,
CP's dK/dV ReduceScatter, EP's gradient AllToAlls, PP's reverse pipeline with
double-width Gantt cells). Render at 480p with `make lq-train-all`.

## How it works

The collectives are **derived, not hand-animated**: `core/engine.py` implements
the scaling book's four sharded-matmul cases symbolically, and
`core/model.py` walks the transformer emitting a renderer-independent step list
(`core/steps.py`). Scenes (`scenes/base.py`) just play the steps back. Adding a
multi-axis combo (FSDP × TP × CP × PP × EP, up to 5D) is a new `StrategyConfig`
in `configs.py` — the engine emits the right collectives for the composed mesh.
`tests/test_engine.py` pins every strategy's collective sequence to the book.

## Setup (macOS)

```bash
uv sync                        # manim CE ≥ 0.19 (video encoding bundled, no ffmpeg needed)
brew install cairo pango pkgconf   # native deps for pycairo/manimpango
# LaTeX for MathTex — TinyTeX is user-local, no sudo:
curl -sL https://yihui.org/tinytex/install-bin-unix.sh | sh
~/Library/TinyTeX/bin/universal-darwin/tlmgr install \
    standalone preview doublestroke setspace rsfs relsize ragged2e \
    fundus-calligra microtype wasysym physics dvisvgm jknapltx wasy cm-super \
    babel-english gnu-freefont mathastext cbfonts-fd
```

The Makefile prepends `~/Library/TinyTeX/bin/universal-darwin` to `PATH`.

## Rendering

```bash
make test                 # engine tests (collective sequences vs the book)
make gallery              # static component sheet -> media/images/gallery/
make lq S=dp C=DPScene    # fast 480p iteration render
make frames S=dp C=DPScene   # dump 1-frame-per-3s PNGs to qa/dp for review
make hq-all               # final 1080p60 renders -> renders/*.mp4
make web                  # record all figures -> web/dist (data.json, data.js, index.html)
make web-one T=5d_train   # re-record one timeline, merged into the existing data.json
make web-qa S=5d M=train T=60   # headless-Chrome still of one figure -> qa/web/
make site                 # sync the figure assets into ../vamsi-aribandi.github.io
```
