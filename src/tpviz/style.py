"""Single source of truth for colors, sizes, and z-ordering.

Visual language:
  - weights   -> blue flat rects with hatch texture
  - activations -> amber fake-3D slab decks (B slabs of T x D)
  - K/V tensors -> teal
  - partial (unreduced) tensors -> desaturated + dashed + {U_axis} badge
  - device identity -> border hue of the DeviceBox and of shards that "belong" to it
"""

BACKGROUND = "#0e1116"

WEIGHT_FILL = "#3b82f6"
WEIGHT_STROKE = "#93c5fd"
ACT_FILL = "#f59e0b"
ACT_STROKE = "#fcd34d"
KV_FILL = "#14b8a6"
KV_STROKE = "#5eead4"

TENSOR_FILL = {"weight": WEIGHT_FILL, "activation": ACT_FILL, "kv": KV_FILL}
TENSOR_STROKE = {"weight": WEIGHT_STROKE, "activation": ACT_STROKE, "kv": KV_STROKE}

# Fill opacity for solid vs partial (unreduced) tensors.
FILL_OPACITY = 0.85
PARTIAL_FILL_OPACITY = 0.30

DEVICE_HUES = ["#e879f9", "#60a5fa", "#4ade80", "#facc15", "#fb7185", "#38bdf8", "#a3e635", "#f472b6"]
DEVICE_BOX_FILL = "#1a2029"
DEVICE_BOX_STROKE = "#3a4454"

EXPERT_HUES = ["#f472b6", "#22d3ee", "#a3e635", "#fb923c"]
MICROBATCH_HUES = ["#f472b6", "#22d3ee", "#a3e635", "#fb923c"]

TEXT_COLOR = "#e5e7eb"
MUTED_TEXT = "#9ca3af"
ACCENT = "#f59e0b"
COMM_COLOR = "#f87171"  # collective/communication highlights
GOOD_COLOR = "#4ade80"

# z-index layers
Z_DEVICE = 0
Z_TENSOR = 2
Z_GRID_LINES = 3
Z_FLYING = 6
Z_LABEL = 8
Z_STRIP = 10
