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
- (entries below appended as v2 work lands)
