# LOG

Linear, detailed log of work. Newest entries at the bottom. See [STATE.md](STATE.md) for the current-state summary.

## 2026-09-10 — Session 1: scaffold → six single-strategy videos

**Decisions made with the user up front**
- Library: Manim CE (LaTeX-native MathTex for the scaling-book notation; MP4 output) over an interactive web app or Motion Canvas.
- Scope: the 5 single strategies first (combos up to 5D later, architecture must anticipate them); DP and FSDP both get videos (6 total); FSDP is the data-axis variant in future combos.

**Environment setup**
- `uv init` + `uv add manim` — pycairo failed to build until `brew install cairo pango pkgconf`. Manim CE 0.21.0.
- LaTeX: `brew install --cask basictex` failed (installer needs interactive sudo). Pivoted to **TinyTeX** (`curl … tinytex/install-bin-unix.sh | sh`) — user-local, no sudo; `tlmgr install` of Manim's package list (the `ms` package no longer exists upstream; harmless). `dvisvgm` 3.6 included.
- Verified end-to-end with a MathTex smoke scene (`AllGather_X In[B_X, D] → In[B, D] C[I,K]{U_X}`) rendered to PNG and inspected.
- No ffmpeg anywhere: Manim ≥0.19 encodes via PyAV; wrote `scripts/extract_frames.py` (PyAV) to dump 1-frame-per-N-seconds PNGs for visual QA, since Claude can read images but not watch video.

**Semantic core (built before any animation)**
- `notation.py`: all LaTeX strings (tensor tex with sharding subscripts and `{U_X}`, collective lines, mesh tex).
- `core/tensors.py` `LTensor` (dims, dim→axis sharding, partial axes, kind). Initially used `MappingProxyType` for immutability — reverted to plain dict because **Manim deep-copies mobjects and mappingproxy can't be pickled** (TypeError mid-render).
- `core/steps.py`: kw-only dataclass step IR (MatMul, AllGather(jit), ReduceScatter, AllReduce, AllToAll(direction), SplitQKV, AttentionCore, Gelu, Route, StageCompute(tick), P2PSend(tick), Annotate).
- `core/engine.py` `plan_matmul()`: the book's four cases; case 4 gathers the weight (preserves activation sharding — FSDP's W_out path). Rejects contract dims sharded over two different axes.
- `core/model.py` `forward_steps(cfg)`: per layer QKV matmul → split → (CP: gather K,V) → attn core → out-proj (scatter_dim="D") → MLP or MoE (route, AllToAll dispatch, expert matmuls, AllToAll combine). GPipe scheduler for PP (`tick = m + s`). Activation naming: In → X (between blocks) → Out (final).
- `configs.py`: DP/FSDP/TP/CP (X axis), TP (Y), EP (Z, 4 experts), PP (2 stages × 4 microbatches).
- `tests/test_engine.py`: 8 tests pinning collective sequences (DP none; FSDP 4 jit AllGathers/layer in order W_qkv,W_o,W_in,W_out; TP AG+RS per block with `{U_Y}` partials; CP AG(K),AG(V) only, MLP silent; EP dispatch/combine AllToAll per layer; PP schedule shape + overlap at tick 1; notation strings; engine rejection case). All green before scene work started.

**Visual kit**
- `style.py` palette/z-indices; `mobjects/tensor_mobject.py` (`ActivationDeck` fake-3D slab deck, `WeightRect` hatched, ghost outlines for sharded extents), `device_grid.py` (DeviceBox with weights/acts slots, DeviceGrid row), `equation_strip.py`, `labels.py`.
- `scenes/gallery.py` static QA sheet. Fixes found via gallery PNG: DashedVMobject was dropping partial-sum fills (draw fill + dashed outline separately); ghost outlines on back slabs were noise (front slab only).

**Scene player + the six scenes**
- `scenes/base.py` `ForwardPassScene`: title card → device row fade-in → initial tensor placement (full deck splits into shards) → step playback by handler dispatch → per-layer phase badge, captions, comm counter → layer 2 at 0.45× → summary card with per-op collective counts.
- Bug: DeviceBox slot frames were computed at construction, before the grid arranged boxes — every tensor landed at screen center. Fix: compute slot geometry live from the box's current position.
- Bug: `self.mobjects` contains non-VMobjects → summary fade filtered to VMobject.
- Collective animation polish after TP frames: only the tensor body flies (labels stay), exchange copies shrunk/faded so ReduceScatter isn't a wall of decks.
- CP: ring-attention callout inset (4-device ring, arrows); "MLP is token-independent — no communication" beat.
- EP: custom token choreography — decks explode into 8 router-colored squares per device (`expert = k % 4`, balanced), AllToAll dispatch = crossflight that visibly sorts by color, expert compute, combine flies them home. Expert tag under each device title.
- PP: custom scene — tick-grouped playback so stage overlap is genuinely simultaneous on screen; microbatch queue with hue frames; live Gantt inset built from the scheduler's tick fields; bubble beat (dashed idle cells); P2P counter.
- QA loop for everything: `-ql` render → frame dump → read PNGs → fix → repeat.

**Ship**
- `make hq-all`: all six 1080p60 MP4s in `renders/` (~45 min in background; killing `media/Tex` during a background render kills the render — learned the hard way).
- Two commits: scaffold+DP+FSDP; TP/CP/PP/EP + polish + README.

## 2026-09-10 — Session 2 (v2): docs, shard offsets, horizontal canvas

- User feedback: (1) sharded tensors all highlight slice 0 on every device — must show device i's slice; (2) model depth must be laid out horizontally (input flows left→right through stations; devices become stacked lanes; shared canvas for PP; sets up backprop later).
- Added STATE.md + this LOG.md.

**Per-device shard offsets**
- `TensorVis` gained `device:`; `_shard_fraction` now returns the device's offset (via `axis_coord`, which unravels flat device indices over multi-axis meshes for later). Solid shard anchored at its offset inside the ghost. B-sharded decks draw a ghost slab stack with the owned slab solid at position i. Gallery gained a device-progression row (same tensor, devices 0..3). Verified: FSDP weight bands march down/right per device; TP ReduceScatter slices land at different offsets per lane.

**Horizontal model canvas**
- New `mobjects/canvas.py`: `ModelCanvas` = lanes (devices, y) × stations (model depth, x): `In` dock, `L{i}·Attn`, `L{i}·MLP/MoE` blocks (2.9 wide, anchor waypoints entry/w1/core/w2/exit as width-fractions), `Out` dock. Header band shows station titles + weight notation once per station (swapped live during FSDP jit gathers). All geometry resolved live from an invisible frame rect. Two tracks per lane: weight fixtures above, traveling activation below. Debug scenes `CanvasDebug`/`CanvasDebugPP` render anchor dots for geometry sign-off.
- `scenes/base.py` rewritten: station highlight + decks slide right on phase change; matmul products prebuilt at the next waypoint so the ReplacementTransform IS the rightward drift; single shared tracker label (clamped to frame) replaces per-lane tensor labels; collectives unchanged — vertical flights fall out of stacked lanes.
- `scenes/pp.py`: 2 stage lanes over the same stations; non-owned stations dimmed; microbatch queue/done stacks in the In/Out docks; tick-grouped simultaneous traversals; diagonal P2P hop across the stage boundary; Gantt below lanes. Bug found via frames: `key_of is self._attn_key` compared fresh bound methods (always False) so Gantt never filled — replaced with a plain flag.
- `scenes/ep.py`: token grids at the MoE station's dispatch column; vertical crossfly sort-by-hue; expert compute at core via base handlers; combine column explodes and flies tokens home. `E{i}` chips on the weight track.
- Fixes from frame QA: station-highlight got BRIGHTER during summary dim (set_opacity raised its 0.06 fill → FadeOut instead); tracker clipped at right edge (x clamp); header-title dict `.get` evaluated its fallback eagerly for dock stations (KeyError on phase "").
- **Perf**: renders went 13 min → ~1 min with `--disable_caching` — Manim's animation-hash on the canvas's many submobjects was the bottleneck, not rendering. Makefile targets updated.
- Deleted `mobjects/device_grid.py` (gallery no longer uses DeviceBox). Engine tests untouched and green throughout.
- Re-rendered all six at 1080p60 into `renders/`; pushed to github.com/vamsi-aribandi/transformer-parallelism-visualization.

## 2026-09-10 — Session 2 (v3): forward+backward training videos

- New goal from user: separate per-strategy videos showing forward AND backward — saved activations, 2× backward compute, gradient tensors. 480p only until sign-off.
- **Semantics** (`core/backward.py`, all engine-derived):
  - `plan_matmul` generalized to multi-dim contraction (`dW = Xᵀ·dY` contracts B and T); partial axes can now stack and resolve sequentially. `LTensor` gained `transposed()`/`grad()` and a `grad` kind; notation renders `dX`, `dW_in` as `\mathrm{d}X`, ….
  - `annotate_forward` injects a `SaveActivationStep` for each forward matmul's (post-gather) input operand + Q/K/V — "save what the matmul consumed" — and records per-layer operands for the backward walk.
  - Backward per layer (reverse order): MLP `dTmp = dOut·W_outᵀ`, `dW_out = Tmpᵀ·dOut`, gelu′, `dX = dTmp·W_inᵀ`, `dW_in`; attention `dA = dX·W_oᵀ`, `dW_o`, attention-core backward (CP: dK/dV partial over the context axis → ReduceScatter onto T), merge dQKV, `dX = dQKV·W_qkvᵀ`, `dW_qkv`. MoE: grad AllToAll dispatch/expert backward/combine. All collectives fall out of the engine's four cases; 8 new tests pin them (DP: AllReduce ×8; FSDP: re-gather AG ×8 + grad RS ×8 landing sharded like the weights; TP: mirrored AG/RS, dW local; CP: dK/dV RS + dW AllReduce over X; EP: 4 grad AllToAlls + attention dW AllReduce over Z; PP: reverse schedule, first backward at last stage, 2-tick spacing). Tests passed on first run — the engine really did derive the backward pass.
- **Visuals** (`scenes/train_base.py` + `*_train.py`):
  - Stash row along each lane's bottom edge: saved activations park as dimmed minis (first save shows the memory-cost caption); dW matmuls pulse the stash mini they consume.
  - Gradients are rose; weight-gradients appear as rose shadow rects behind their weight (partial → dashed until AllReduce/ReduceScatter resolves them, per-lane slice offsets visible after RS).
  - Backward on_phase slides everything to station EXIT anchors (flow runs right→left); `GradInitStep` transforms Out into dOut at the Out dock ("backward starts from the loss"); finale lands dIn at the In dock.
  - matmul counter beside the collectives counter; train summary card: fwd vs bwd matmuls + per-pass collectives + "backward ≈ 2× forward".
  - EP refactored into `MoETokenMixin` (fwd + bwd token crossflys, mirrored columns for backward); PP train: forward drain then rose gradient microbatches flowing back with UP-LEFT P2P hops and double-width rose Gantt cells (`gantt_fill_bwd`), smaller cells (0.42) to fit the 15-tick timeline.
  - Makefile: `lq-train-all` (480p, `--disable_caching`); hq-all deliberately unchanged.

## 2026-09-11 — Session 3: duality notes everywhere + web player begins

- Re-rendered all six train scenes with the clarity fixes; verified per-strategy: DP attention-backward cites `from forward: Attn(Q,K,V)→A`; FSDP re-gathers cite the identical forward gather; CP dK/dV ReduceScatters cite the forward K/V AllGathers; EP grad AllToAlls cite the opposite-direction forward AllToAll; PP backward P2P sends cite the forward handoff (`dual of forward's Send: In stage 0→1`).
- Started v4: interactive web player (see STATE.md / plan). Architecture: RECORD the real manim scenes (RecorderMixin overrides play/wait, samples animations at 5 alphas via manim's own compile/interpolate machinery, scene-membership diffs give spawn/despawn) → quantized keyframe JSON + deduped LaTeX glyph atlas → dependency-free JS player. First recorder run on DPScene: 34.4s timeline, 234 semantic objects, 15 steps, zero unserialized mobjects.

**Interactive web player (v4, TP first)**
- Architecture shipped as planned: `RecorderMixin` replays each scene's construct() with play() overridden — manim's own `compile_animations`/`interpolate` sampled at 5 alphas per play (easing, arcs, pulses captured as piecewise-linear tracks), scene-membership diffs give spawn/despawn, `add_mobjects_from_animations` mirrored for Transform bookkeeping. Bugs found: CPython id() reuse silently swallowed re-registered mobjects (the equation strip vanished — fixed by pinning every recorded mobject); manim `Text.get_color()` is always black (real color lives on family-member fills); a sed-loop self-match left all three `_tag_collective_notes` calls in backward_mlp.
- LaTeX pre-rendered via manim's tex pipeline into a deduped glyph atlas (paths keyed by `d`-data, not dvisvgm's unstable ids); equations render client-side as `<use>` lists with `currentColor`.
- `web/`: dependency-free JS player (binary-search keyframes, rAF), program panel (collapsible phase sections, live highlight, click-to-seek, duality notes inline), tensor hover tooltips (concrete dims B=8 T=128 D=1024 F=4096, memory, shard sentence), transport (fwd/bwd-tinted slider, phase ticks, scrub preview, keyboard), strategy tabs + mode toggle, legend. Self-contained `web/dist/index.html` (0.57 MB with tp_fwd + tp_train).
- In-frame chrome (title header, badges, captions, corner counters) stripped via `web_role` tags; HTML chips/tabs replace them. QA loop: headless Chrome screenshots at `#s=tp&m=train&t=…&still=1` (isolated --user-data-dir per shot; Chrome lingers after writing — run under `timeout`).
- `make web`, `make web-all`, `make web-qa`; tests/test_web_export.py (validation, step↔time monotonicity, summary==semantic counts, tooltips present, zero unserialized mobjects).

**v5: article rework (user feedback on the app format)**
- Threw away the full-app shell + continuous-time player. New format: a distill-style scrollable ARTICLE (`web/article.html` + `web/tpviz.js` + rewritten `style.css`) with `<tpviz-figure strategy="…">` custom elements embeddable between prose — no framework, no bundler.
- Data model v2 (`schema.py` rewritten): discrete STEP STATES — per step an exact sparse end-state diff (teleport target), spawn/despawn lists, and a few "beats" (play-mode tweens that snap to the exact state afterwards). Title card and summary card dropped (prose carries them); the recorded equation strip replaced by an HTML readout under the canvas (equation + duality note + caption).
- Interactions per user: discrete segmented stepper (one segment per operation, amber fwd / rose bwd, click = teleport, hover/current scale up), play mode = the only animation, prev/next buttons + arrow keys per figure, backward pass collapsed behind a labeled toggle with matmul meta, program list as a collapsible details block, hover tooltips carried over.
- Light + dark page themes (CSS vars, prefers-color-scheme + toggle persisted); the canvas stays a deliberate dark panel in both. Serif prose (Source Serif 4), Inter UI, JetBrains Mono code.
- Article ships TP (fwd + collapsible bwd) with intro + how-to-read + TP commentary; other strategies join as sections. `#step=N&bwd=1&theme=dark` deep links double as the headless-Chrome QA hooks. data 0.32 MB, page 0.36 MB.

**v6: article polish round (user feedback)**
- Counters removed from the figure header; mode is now a Forward | + Backward toggle. Forward mode hides SaveActivation steps AND the stash-mini objects (`saveObjs` exported per timeline); play mode silently absorbs skipped steps then animates the visible one.
- Equations are now selectable HTML text (`texthtml.py`: a closed-grammar tex→HTML converter — we generate all the LaTeX, so no KaTeX needed). The glyph atlas remains only for in-canvas equations. Bug: bare `{...}` groups (mesh notation `{:}`) initially passed through literally.
- Program moved beside the canvas (grid: canvas+stepper left, program right; stacks on narrow); the bottom readout is gone — the program auto-centers and expands the current entry with its caption, and backward entries show their forward citation plus a "jump to that forward step" link (`fwdStep` computed by matching note_tex to forward equations).
- Full Gruvbox theming conforming to vamsi-aribandi.github.io (light #fbf1c7/#af3a03, dark #282828/#fe8019): article chrome AND the canvas now both theme (canvas via --cv-* vars; tensor fills are neutral gruvbox tones shared across themes, strokes bright-in-dark/faded-in-light; hexes mapped to tokens at export via HEX2TOK).
- Bugs from headless QA: `stop()` before the stepper exists (guard), id-reuse pin from v5 still good, `Text.get_color()` family-fill fix carried.

**v7: figure interaction polish**
- "initial state" entry at the top of the program (click → base state); program auto-scroll now rect-based centering (works upward too — offsetTop was measured against the wrong ancestor and over-scrolled).
- Forward mode truly hides saved activations: stash minis are now marked at the SOURCE (`web_save` attr → `sv` flag → `saveObjs`) instead of inferred from spawn-boundary attribution, which missed half of them (spawn windows straddle step boundaries).
- Program subtext trimmed to just the forward citation (+ jump link) on backward steps; section headings are sticky within the list.
- Collective animations restored to the real recorded choreography: beats now carry spawn-time states (flight copies fly from their SOURCE lane instead of fading in at the destination) and interior path samples for arced crossflies (AllGather/ReduceScatter/AllToAll arcs replay faithfully in play mode).
