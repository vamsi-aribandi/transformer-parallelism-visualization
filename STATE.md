# STATE

Current state and design decisions. Companion: [LOG.md](LOG.md) (linear work log), [README.md](README.md) (setup/usage).

## What this is

Manim CE videos and an interactive article explaining transformer parallelism, one strategy at a time, in the [JAX scaling book](https://jax-ml.github.io/scaling-book/)'s sharding notation: `A[B_X, T, D]` (subscript = mesh axis sharding that dim), partial sums `C[I,K]{U_X}`, `Mesh({'X': 4})`, `AllGather_X`, `ReduceScatter_{Y,D}`, `AllToAll_Z`.

**Deliverables to date**:
- 6 forward-pass videos in `renders/` (1080p60): `dp`, `fsdp`, `tp`, `cp`, `pp`, `ep`.
- 6 **forward+backward** (training-step) videos, 480p only until the look is signed off (`make lq-train-all`; scenes `*_train.py`). They add: saved activations parking in a per-station stash row (the memory cost), rose gradient tensors flowing right→left, weight-gradient "shadow" rects behind each weight, a matmul counter showing backward = 2× forward, and PP's double-width backward Gantt cells.
- **The interactive article** (`web/dist/index.html`, `make web`): all six strategies as prose sections with embedded `<tpviz-figure>` web components — discrete stepper (click = teleport, play = the only animation, speed slider), side program with sticky sections/live highlight/forward-citation links, hover tooltips with concrete shapes and memory, a fixed detail band where the B×T×D diagram hands off to wire-level bidirectional-ring collective animations, Gruvbox light/dark themes matching vamsi-aribandi.github.io. Data recorded from the manim scenes (RecorderMixin), equations as selectable HTML (closed-grammar tex→HTML). PP figures are driven by `mark_step` hooks in the scenes.

- **ZeRO-1** (`ZERO1` config, `zero1_axis`): weights replicated, gradients ReduceScatter over the data axis, then `OptimizerStep` (each device updates its shard) + AllGather of the updated weights — `backward.update_steps()` appends these after the backward. Renders on the lane canvas like the other singles.
- **Recipe figures** (`scenes/five_d.py` subclasses on `MeshGrid`, which lays out whichever of X/C (rows) and Z/Y (columns) a mesh has): `DENSE4D` = Llama 3 style FSDP × TP × CP × PP (16 devices), `MOE4D` = DeepSeek / Kimi style EP × CP × PP × ZeRO-1 with no TP (16 devices), and the 5D figure below (Nemotron 3 / Qwen3-VL style). The article is structured as "The axes of transformer sharding" (the singles + ZeRO-1) and "Examples from frontier models" (the three recipes).
- **The 5D figure** (`scenes/five_d.py`, `mobjects/meshgrid.py`, config `FIVE_D`): all five strategies on 32 devices, `Mesh({'X': 2, 'Y': 2, 'C': 2, 'Z': 2, 'stage': 2})`, forward + backward, in the article as the closing section. Every collective is attributed to its mesh axis three ways: an axis-colored halo on the flying copies + an on-canvas badge ("AllGather over X · FSDP"), axis chips on every program line + axis-colored step-bar segments, and a per-axis tally row (hover isolates, click jumps). Activation saving is not drawn there.

Model shown: 2-layer transformer, simple MHA (fused head dim `H`) + MLP; the MLP is a routed MoE for EP.

## Architecture (the load-bearing decision)

Collectives are **derived, not hand-animated**, so future multi-axis combos (FSDP×TP×CP×PP×EP, up to 5D) are configs, not new scenes:

- `src/tpviz/core/engine.py` — `plan_matmul()`: the scaling book's four sharded-matmul cases, symbolically, **applied per mesh-axis letter**. Sharding subscripts are letter strings (`"X"`, or `"XZ"` = one dim split over two axes at once; canonical letter order in `mesh.AXIS_ORDER`). Case 2 alone yields both FSDP's jit weight-gather and TP's activation-gather — and when the two operands shard the contracting dim on *different* axes (FSDP × TP) both get gathered, weight first; case 3 yields `{U_Y}` partials (several letters at once render as one badge, `{U_{XZC}}`); case 4 gathers the weight when one letter would shard two different output dims, but a batched dim co-sharded on the same axis on both operands (`E_Z`) is legal. Partial sums resolve with a ReduceScatter along `scatter_axis` (the weight's FSDP axis, from `backward._wt_scatter`) and AllReduces along every other partial axis.
- `src/tpviz/core/model.py` — `forward_steps(cfg)`: walks the 2-layer transformer, emits a renderer-independent step list (`core/steps.py`). Non-matmul-shaped pieces are emitted explicitly: CP's K/V AllGather, MoE routing + AllToAlls, the GPipe pipeline schedule (`tick = microbatch + stage`).
- `src/tpviz/configs.py` — the 7 single `StrategyConfig`s (incl. `ZERO1`) + the recipes `DENSE4D`, `MOE4D`, `FIVE_D`. Axis conventions: X = data/FSDP (the CP single reuses it for context), Y = tensor, C = context (combos), Z = expert, `stage` = pipeline. `axis_roles` names the parallelism per axis (the attribution the figures show).
- MoE and pipeline compose through the engine too: `model.routed_tokens()` gives the dispatched form `X[E_Z, S_{XC}, D_Y]` (the AllToAll re-sorts tokens along Z only, so S keeps the other token axes and D its TP sharding); expert matmuls go through `plan_matmul` (so they pick up FSDP gathers and TP gather/scatter on their own); `PipelineConfig(composed=True)` runs each stage's layer inline and `model.stage_send()` inserts the P2P hop between layers (forward and backward).
- `src/tpviz/core/backward.py` — `train_steps(cfg)`: forward (each matmul's consumed input emitted as a `SaveActivationStep`, plus the pre-gelu `Z` — the MLP is `Z = X·W_in`, `Tmp = gelu(Z)`, so the backward's `gelu′(Z)` is evaluated at a tensor that was actually saved; under CP the K/V **shards** are saved, not the gathered full-sequence forms, and the backward re-gathers them) + backward **derived with the same engine**: per forward matmul, `dX = dY·Wᵀ` (single contract) and `dW = Xᵀ·dY` (contracts over B *and* T — `plan_matmul` accepts multi-dim contraction). The four cases then yield: DP gradient AllReduce, FSDP weight re-gather + grad ReduceScatter, TP's mirrored AG/RS, CP's dK/dV ReduceScatter + weight-grad AllReduce over context, EP's grad AllToAlls. PP gets a GPipe backward schedule with 2-tick-wide cells.
- `tests/test_engine.py` + `tests/test_backward.py` + `tests/test_five_d.py` — pin each strategy's exact forward AND backward collective sequences (per block and per axis for 5D), the 2× matmul ratio, and save-before-backward ordering. **Any semantic change must keep these green.**
- Scenes (`src/tpviz/scenes/`) replay the steps: `base.py` is the config-driven player; `dp/fsdp/tp/cp` are thin subclasses; `pp.py` (tick-grouped playback + Gantt) and `ep.py` (token-square AllToAll) carry custom choreography.

## Visual language

- **Model canvas (v2)**: model depth is the x-axis — stations `In → L1·Attn → L1·MLP → L2·Attn → L2·MLP → Out`; devices are horizontal lanes stacked vertically (`src/tpviz/mobjects/canvas.py`). The activation deck physically travels left→right; collectives fly **vertically** between lanes at the current station's x. Weight fixtures sit on each lane's upper track; decks travel the lower track. Weight notation appears once per station in a header row; a single shared "tracker" label under the canvas follows the current activation. PP shares the canvas (stage lanes own their layer's stations, others dimmed; microbatches hop diagonally at the boundary; Gantt inset below). Backprop later = same canvas, right→left.
- Weights = blue hatched rects; activations = amber fake-3D slab decks (B slabs of T×D; sharding: B→owned slab within a ghost stack, T→horizontal band, D→vertical slice; the solid part sits at THIS device's offset inside a dashed ghost outline, so lanes visibly hold different slices); K/V = teal; partial sums = translucent + dashed + `{U_X}` in the label.
- All LaTeX built in `src/tpviz/notation.py` (single choke point; brace escaping, subscripts).
- **All article prose lives in `scripts/build_prose.py`** (`make prose` writes both `web/article.html` and the site's post). Edit the text there, not in the outputs. Voice: scaling-book concise.
- Persistent bottom **equation strip** shows every op in book notation; comm counter top-right; per-layer collective summary card at the end.
- **Mesh grid canvas (5D)**: one 4×4 device grid per pipeline stage, side by side (stage 0 = layer 1 left, stage 1 = layer 2 right); rows = X outer / C inner, columns = Z outer / Y inner, so FSDP partners are two rows apart, context partners adjacent rows, expert partners two columns apart, tensor partners adjacent columns, pipeline partners the same cell in the other grid. Each box is a miniature lane (weights on top, the activation deck below; the deck drifts right through the forward, left through the backward). Only the active station's weights are shown per stage (swapped on phase change; a stage's weights dim rather than vanish when the activation leaves for the other stage). Axis colors live in `style.AXIS_HUES` → web tokens `--cv-ax{X,Y,C,Z,P}`.
- 2D `Scene` only (fake-3D decks) — never `ThreeDScene`.
- Geometry debug harness: `CanvasDebug` / `CanvasDebugPP` scenes in `scenes/gallery.py` (anchor dots on a static canvas).

## Environment facts (non-obvious)

- LaTeX = **TinyTeX**, user-local at `~/Library/TinyTeX/bin/universal-darwin` (chosen over BasicTeX to avoid sudo). Makefile prepends it to PATH.
- No system ffmpeg needed (Manim ≥0.19 encodes via PyAV). `scripts/extract_frames.py` uses PyAV for the QA loop: `make lq` → `make frames` → inspect `qa/<scene>/*.png`.
- Manim deep-copies mobjects: anything they hold must be picklable (LTensor uses a plain dict, not MappingProxyType).
- Never cache scene-unit geometry at construction (boxes get arranged/shifted after); compute anchors live. Don't delete `media/Tex` while a render runs.
- **Render with `--disable_caching`** (Makefile does): Manim's animation-hashing on the canvas's many submobjects made a 1-minute render take 13 minutes.
- **Never point-morph a deck into a group containing `Text`** (`ReplacementTransform`/`Transform`): Manim aligns the two mobject trees by deep-copying submobjects until the families match, and a Text's glyphs against a deck's dashes explode combinatorially — at 16 devices one such step took 280 s and 8 GB. Use `FadeTransform` for heterogeneous morphs, and keep chips out of morph targets. Ghost dash counts follow the rect perimeter (`_ghost`) so small decks stay cheap to copy; the recorder caches each unit's family (`serializers._family`).
- `make web-one T=<timeline>` re-records one timeline and merges it into the existing `dist/data.json` (`export.merge_documents` re-keys the positional glyph ids and tooltip indices). The 5D recording takes ~10 min.

## Known gaps / next work

- The 5D figure shows one microbatch (stages take turns) and no activation saving; FSDP shards over X only (in practice over the whole X×Z data group, which would fold the dW AllReduce over Z into the ReduceScatter). The three dW resolutions (RS_X, AR_Z, AR_C) are kept separate for attribution; real systems fuse them into one ReduceScatter over the (X, Z, C) group.
- Intermediate combos (FSDP×TP, TP×CP, ...) are now one `StrategyConfig` each and would play on `MeshGrid` as long as the row/column axis pairs exist; `MeshGrid` assumes the axes X, C (rows) and Z, Y (columns) are present.
- No 5D video has been rendered yet (`make hq-5d`); the figure is web-only so far.
- The training videos (`*_train`) predate several article-era fixes (dW anchoring, gelu-derivative notation land automatically on re-render; chips/counters are video-only by design) — re-render with `make lq-train-all` when needed.
- Ignored by design: residual stream, LayerNorm, GQA/attention tricks, MoE capacity/load-balancing (called out in the EP video).
