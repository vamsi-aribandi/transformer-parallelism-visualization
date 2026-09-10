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
```
