# STATE

Current state and design decisions. Companion: [LOG.md](LOG.md) (linear work log), [README.md](README.md) (setup/usage).

## What this is

Manim CE videos explaining transformer parallelism in the **forward pass**, one video per strategy, in the [JAX scaling book](https://jax-ml.github.io/scaling-book/)'s sharding notation: `A[B_X, T, D]` (subscript = mesh axis sharding that dim), partial sums `C[I,K]{U_X}`, `Mesh({'X': 4})`, `AllGather_X`, `ReduceScatter_{Y,D}`, `AllToAll_Z`.

**Deliverables to date**: 6 videos in `renders/` (1080p60): `dp`, `fsdp`, `tp`, `cp`, `pp`, `ep`. Model shown: 2-layer transformer, simple MHA (fused head dim `H`) + MLP; the MLP is a routed MoE for EP.

## Architecture (the load-bearing decision)

Collectives are **derived, not hand-animated**, so future multi-axis combos (FSDP×TP×CP×PP×EP, up to 5D) are configs, not new scenes:

- `src/tpviz/core/engine.py` — `plan_matmul()`: the scaling book's four sharded-matmul cases, symbolically. Case 2 alone yields both FSDP's jit weight-gather and TP's activation-gather; case 3 yields TP's `{U_Y}` partial + ReduceScatter; case 4 yields FSDP's second weight-gather.
- `src/tpviz/core/model.py` — `forward_steps(cfg)`: walks the 2-layer transformer, emits a renderer-independent step list (`core/steps.py`). Non-matmul-shaped pieces are emitted explicitly: CP's K/V AllGather, MoE routing + AllToAlls, the GPipe pipeline schedule (`tick = microbatch + stage`).
- `src/tpviz/configs.py` — the 6 `StrategyConfig`s. Axis conventions: X = data/FSDP/context, Y = tensor, Z = expert, `stage` = pipeline.
- `tests/test_engine.py` — pins each strategy's exact collective sequence to the book. **Any semantic change must keep these green.**
- Scenes (`src/tpviz/scenes/`) replay the steps: `base.py` is the config-driven player; `dp/fsdp/tp/cp` are thin subclasses; `pp.py` (tick-grouped playback + Gantt) and `ep.py` (token-square AllToAll) carry custom choreography.

## Visual language

- Weights = blue hatched rects; activations = amber fake-3D slab decks (B slabs of T×D; sharding: B→slab subset, T→horizontal band, D→vertical slice, always with a dashed ghost outline of the full extent); K/V = teal; partial sums = translucent + dashed + `{U_X}` in the label.
- All LaTeX built in `src/tpviz/notation.py` (single choke point; brace escaping, subscripts).
- Persistent bottom **equation strip** shows every op in book notation; comm counter top-right; per-layer collective summary card at the end.
- 2D `Scene` only (fake-3D decks) — never `ThreeDScene`.

## Environment facts (non-obvious)

- LaTeX = **TinyTeX**, user-local at `~/Library/TinyTeX/bin/universal-darwin` (chosen over BasicTeX to avoid sudo). Makefile prepends it to PATH.
- No system ffmpeg needed (Manim ≥0.19 encodes via PyAV). `scripts/extract_frames.py` uses PyAV for the QA loop: `make lq` → `make frames` → inspect `qa/<scene>/*.png`.
- Manim deep-copies mobjects: anything they hold must be picklable (LTensor uses a plain dict, not MappingProxyType).
- Never cache scene-unit geometry at construction (boxes get arranged/shifted after); compute anchors live. Don't delete `media/Tex` while a render runs.

## Known gaps / next work

- **v2 in progress**: (1) per-device shard offsets — every device currently draws slice 0; (2) horizontal model canvas — model depth becomes the x-axis (stations `In → L1·Attn → L1·MLP → L2·Attn → L2·MLP → Out`), devices become stacked lanes, collectives fly vertically; PP shares the canvas (stages own station ranges); sets up backprop (right→left) later.
- Multi-axis combos need: multi-lane/grid mesh layout, PP composition in the scheduler, per-axis coloring.
- Ignored by design: residual stream, LayerNorm, GQA/attention tricks, MoE capacity/load-balancing (called out in the EP video).
