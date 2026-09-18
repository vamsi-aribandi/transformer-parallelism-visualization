# STATE

Current state and design decisions. Companion: [LOG.md](LOG.md) (linear work log), [README.md](README.md) (setup/usage).

## What this is

Manim CE videos and an interactive article explaining transformer parallelism, one strategy at a time, in the [JAX scaling book](https://jax-ml.github.io/scaling-book/)'s sharding notation: `A[B_X, T, D]` (subscript = mesh axis sharding that dim), partial sums `C[I,K]{U_X}`, `Mesh({'X': 4})`, `AllGather_X`, `ReduceScatter_{Y,D}`, `AllToAll_Z`.

**Deliverables to date**:
- 6 forward-pass videos in `renders/` (1080p60): `dp`, `fsdp`, `tp`, `cp`, `pp`, `ep`.
- 6 **forward+backward** (training-step) videos, 480p only until the look is signed off (`make lq-train-all`; scenes `*_train.py`). They add: saved activations parking in a per-station stash row (the memory cost), rose gradient tensors flowing right→left, weight-gradient "shadow" rects behind each weight, a matmul counter showing backward = 2× forward, and PP's double-width backward Gantt cells.
- **The interactive article** (`web/dist/index.html`, `make web`): all six strategies as prose sections with embedded `<tpviz-figure>` web components — discrete stepper (click = teleport, play = the only animation, speed slider), side program with sticky sections/live highlight/forward-citation links, hover tooltips with concrete shapes and memory, a fixed detail band where the B×T×D diagram hands off to wire-level bidirectional-ring collective animations, Gruvbox light/dark themes matching vamsi-aribandi.github.io. Data recorded from the manim scenes (RecorderMixin), equations as selectable HTML (closed-grammar tex→HTML). PP figures are driven by `mark_step` hooks in the scenes.

Model shown: 2-layer transformer, simple MHA (fused head dim `H`) + MLP; the MLP is a routed MoE for EP.

## Architecture (the load-bearing decision)

Collectives are **derived, not hand-animated**, so future multi-axis combos (FSDP×TP×CP×PP×EP, up to 5D) are configs, not new scenes:

- `src/tpviz/core/engine.py` — `plan_matmul()`: the scaling book's four sharded-matmul cases, symbolically. Case 2 alone yields both FSDP's jit weight-gather and TP's activation-gather; case 3 yields TP's `{U_Y}` partial + ReduceScatter; case 4 yields FSDP's second weight-gather.
- `src/tpviz/core/model.py` — `forward_steps(cfg)`: walks the 2-layer transformer, emits a renderer-independent step list (`core/steps.py`). Non-matmul-shaped pieces are emitted explicitly: CP's K/V AllGather, MoE routing + AllToAlls, the GPipe pipeline schedule (`tick = microbatch + stage`).
- `src/tpviz/configs.py` — the 6 `StrategyConfig`s. Axis conventions: X = data/FSDP/context, Y = tensor, Z = expert, `stage` = pipeline.
- `src/tpviz/core/backward.py` — `train_steps(cfg)`: forward (each matmul's consumed input emitted as a `SaveActivationStep`) + backward **derived with the same engine**: per forward matmul, `dX = dY·Wᵀ` (single contract) and `dW = Xᵀ·dY` (contracts over B *and* T — `plan_matmul` accepts multi-dim contraction). The four cases then yield: DP gradient AllReduce, FSDP weight re-gather + grad ReduceScatter, TP's mirrored AG/RS, CP's dK/dV ReduceScatter + weight-grad AllReduce over context, EP's grad AllToAlls. PP gets a GPipe backward schedule with 2-tick-wide cells.
- `tests/test_engine.py` + `tests/test_backward.py` — pin each strategy's exact forward AND backward collective sequences, the 2× matmul ratio, and save-before-backward ordering. **Any semantic change must keep these green.**
- Scenes (`src/tpviz/scenes/`) replay the steps: `base.py` is the config-driven player; `dp/fsdp/tp/cp` are thin subclasses; `pp.py` (tick-grouped playback + Gantt) and `ep.py` (token-square AllToAll) carry custom choreography.

## Visual language

- **Model canvas (v2)**: model depth is the x-axis — stations `In → L1·Attn → L1·MLP → L2·Attn → L2·MLP → Out`; devices are horizontal lanes stacked vertically (`src/tpviz/mobjects/canvas.py`). The activation deck physically travels left→right; collectives fly **vertically** between lanes at the current station's x. Weight fixtures sit on each lane's upper track; decks travel the lower track. Weight notation appears once per station in a header row; a single shared "tracker" label under the canvas follows the current activation. PP shares the canvas (stage lanes own their layer's stations, others dimmed; microbatches hop diagonally at the boundary; Gantt inset below). Backprop later = same canvas, right→left.
- Weights = blue hatched rects; activations = amber fake-3D slab decks (B slabs of T×D; sharding: B→owned slab within a ghost stack, T→horizontal band, D→vertical slice; the solid part sits at THIS device's offset inside a dashed ghost outline, so lanes visibly hold different slices); K/V = teal; partial sums = translucent + dashed + `{U_X}` in the label.
- All LaTeX built in `src/tpviz/notation.py` (single choke point; brace escaping, subscripts).
- Persistent bottom **equation strip** shows every op in book notation; comm counter top-right; per-layer collective summary card at the end.
- 2D `Scene` only (fake-3D decks) — never `ThreeDScene`.
- Geometry debug harness: `CanvasDebug` / `CanvasDebugPP` scenes in `scenes/gallery.py` (anchor dots on a static canvas).

## Environment facts (non-obvious)

- LaTeX = **TinyTeX**, user-local at `~/Library/TinyTeX/bin/universal-darwin` (chosen over BasicTeX to avoid sudo). Makefile prepends it to PATH.
- No system ffmpeg needed (Manim ≥0.19 encodes via PyAV). `scripts/extract_frames.py` uses PyAV for the QA loop: `make lq` → `make frames` → inspect `qa/<scene>/*.png`.
- Manim deep-copies mobjects: anything they hold must be picklable (LTensor uses a plain dict, not MappingProxyType).
- Never cache scene-unit geometry at construction (boxes get arranged/shifted after); compute anchors live. Don't delete `media/Tex` while a render runs.
- **Render with `--disable_caching`** (Makefile does): Manim's animation-hashing on the canvas's many submobjects made a 1-minute render take 13 minutes.

## Known gaps / next work

- Multi-axis combos need: lanes for multi-axis meshes (lane per device with axis-coord grouping/coloring), PP composition in the scheduler, per-axis flight styling. `axis_coord()` in tensor_mobject.py already unravels multi-axis device indices.
- The training videos (`*_train`) predate several article-era fixes (dW anchoring, gelu-derivative notation land automatically on re-render; chips/counters are video-only by design) — re-render with `make lq-train-all` when needed.
- Ignored by design: residual stream, LayerNorm, GQA/attention tricks, MoE capacity/load-balancing (called out in the EP video).
