"""Single source for the article prose: writes the Hugo post body (site) and the
standalone article's <article> body. Notation is typeset through notation.py +
tex_to_html, so it matches the figures' program panel exactly.

    uv run python scripts/build_prose.py
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

from tpviz import notation
from tpviz.core.tensors import LTensor
from tpviz.web.texthtml import tex_to_html

ROOT = Path(__file__).resolve().parents[1]
ARTICLE = ROOT / "web" / "article.html"
POST = ROOT.parent / "vamsi-aribandi.github.io" / "content" / "posts" / "parallel-transformers.md"
BOOK = "https://jax-ml.github.io/scaling-book/"


def m(tex: str) -> str:
    return f'<span class="meq">{tex_to_html(tex)}</span>'


def t(name, dims, sharding=None, *, kind="activation", partial=""):
    return m(LTensor(name, tuple(dims), sharding or {}, partial=frozenset(partial), kind=kind).tex())


def op(name, axis, dim=None):
    return m(notation.op_tex(name, axis, dim))


def mesh(**axes):
    return m(notation.mesh_tex(axes))


def a(text, url):
    return f'<a href="{url}">{text}</a>'


X, Y, Z, C = m("X"), m("Y"), m("Z"), m("C")
BTD = ("B", "T", "D")

DEK = (f"How a transformer's forward and backward passes are sharded across devices: each axis "
       f"of parallelism on its own, then the combinations frontier models train with, in the "
       f"notation of the {a('JAX scaling book', BOOK)}.")

# (level, heading, id, paragraphs, figure strategy or None, caption)
SECTIONS = [
    (2, "How to read the notation", "notation", [
        f"""Everything here uses the scaling book's {a("sharding notation", BOOK + "sharding/")}.
        Devices form a <em>mesh</em>: {mesh(X=4)} is four devices along one axis named {X}.
        A subscript names the mesh axis a dimension is split over: {t("In", BTD, {"B": "X"})}
        means the batch is sharded four ways over {X}, so each device holds a
        {m(r"[B/4,\, T,\, D]")} slice. A dimension without a subscript is not split, and a
        tensor with no subscripts is replicated. A trailing {m(r"\{U_X\}")} marks an
        <strong>unreduced partial sum</strong>: each device holds a full-shaped tensor that is
        only its contribution, and the contributions still have to be summed over {X}. That sum
        is what AllReduce and ReduceScatter do.""",
        f"""Sizes assumed by the tooltips: {m("B=8")} sequences of {m("T=128")} tokens,
        {m("D=1024")} model width, {m("F=4096")} MLP width, {m("H=1024")} the attention heads
        concatenated, {m("E=4")} experts with {m("S=256")} routed tokens each, all in bf16.""",
    ], None, None),
    (2, "How to read the figures", "figures", [
        """Model depth runs left to right: the input enters at <span class="meq"><span class="mu">In</span></span>,
        passes each layer's attention and MLP stations, and leaves at
        <span class="meq"><span class="mu">Out</span></span>. Devices are horizontal lanes; when
        they communicate, tensors fly vertically between lanes. Weights are
        <span class="sw wt"></span> blue, activations <span class="sw act"></span> amber, keys and
        values <span class="sw kv"></span> teal, gradients <span class="sw grad"></span> rose. A
        dashed outline is the full logical tensor; the solid patch inside it is the slice this
        device holds, drawn at its offset. Hover a tensor for its shape, sharding and memory.""",
        """Each segment of the step bar is one operation. Click to jump to the state after it, or
        press play. While the current operation is a collective, the panel under the canvas shows
        how it runs on the wire, hop by hop. The program beside the canvas lists every operation
        in book notation and follows along. Figures open forward-only; <em>+ Backward</em> adds the
        rest of the training step, where saved activations park under the weights that need them
        and every backward operation cites the forward operation it differentiates.""",
    ], None, None),
    (2, "The axes of transformer sharding", "axes", [
        """Each way of splitting a transformer is a choice of dimension to shard: the batch, the
        sequence, the weights' features, the experts, or the layers. Here each runs alone on a
        four-device mesh, so its communication is visible in isolation.""",
    ], None, None),
    (3, "Data parallelism", "dp", [
        f"""Replicate the weights, shard the batch: {t("In", BTD, {"B": "X"})}. The forward pass
        communicates nothing. The backward does: each device computes weight gradients from its
        quarter of the batch, so every {t("dW", ("D", "F"), partial="X", kind="grad")} is a partial
        sum, and each one is AllReduced over {X} before the optimizer step.""",
    ], "dp", f"Data parallelism over {mesh(X=4)}: a silent forward pass, one AllReduce per weight gradient."),
    (3, "ZeRO-1: shard the optimizer, keep the weights", "zero1", [
        f"""Data parallelism also replicates the optimizer state, which for Adam is two more
        model-sized tensors. <strong>ZeRO-1</strong> keeps the weights replicated and shards the
        optimizer state over {X}. Only the gradient step changes. Each
        {t("dW", ("D", "F"), partial="X", kind="grad")} is <strong>ReduceScattered</strong> rather
        than AllReduced, so a device holds {t("dW", ("D", "F"), {"D": "X"}, kind="grad")}; it
        updates its slice {t("W", ("D", "F"), {"D": "X"}, kind="weight")}; and an
        <strong>AllGather</strong> of the updated slices makes the weights whole again. An
        AllReduce is a ReduceScatter followed by an AllGather, so ZeRO-1 moves the same bytes as
        data parallelism, with the optimizer step in between. This is the data axis DeepSeek-V3
        and Kimi K2 train with.""",
    ], "zero1", f"ZeRO-1 over {mesh(X=4)}: ReduceScatter, a sharded optimizer step, then an AllGather of the updated weights."),
    (3, "Fully-sharded data parallelism (ZeRO-3)", "fsdp", [
        f"""ZeRO-1 still replicates every weight. FSDP shards them too, along the same axis:
        {t("W_in", ("D", "F"), {"D": "X"}, kind="weight")},
        {t("W_out", ("F", "D"), {"D": "X"}, kind="weight")}. Each weight is
        <strong>AllGathered just before its matmul</strong> and discarded right after; the station
        header shows the gathered shape and back. The backward gathers each weight again, and
        each gradient <strong>ReduceScatters</strong> back to the layout the weights live in.
        Memory of a shard, communication of a gather, in both passes.""",
    ], "fsdp", f"FSDP over {mesh(X=4)}: just-in-time weight gathers forward, re-gathers and gradient ReduceScatters backward."),
    (3, "Tensor parallelism", "tp", [
        f"""Tensor parallelism shards the feature dimensions: the heads in attention
        ({t("W_qkv", ("D", "H"), {"H": "Y"}, kind="weight")},
        {t("W_o", ("H", "D"), {"H": "Y"}, kind="weight")}) and the hidden width in the MLP
        ({t("W_in", ("D", "F"), {"F": "Y"}, kind="weight")},
        {t("W_out", ("F", "D"), {"F": "Y"}, kind="weight")}). Activations travel sharded on
        {m("D")}. Each block <strong>AllGathers on the way in and ReduceScatters on the way
        out</strong>: the first matmul needs all of {m("D")}, so the activations are gathered;
        the second contracts a sharded dimension, so each device is left with a partial sum
        {m(r"\{U_Y\}")}, which the ReduceScatter resolves while re-sharding for the next block.""",
    ], "tp", f"Tensor parallelism over {mesh(Y=4)}: AllGather into every block, ReduceScatter out of it, mirrored in the backward."),
    (3, "Context parallelism", "cp", [
        f"""Context parallelism shards the sequence: {t("In", BTD, {"T": "X"})}. The MLP acts per
        token and never communicates. Attention needs every key and value for every query, so
        {m("K")} and {m("V")} are <strong>AllGathered</strong> over the sequence shards while
        {m("Q")} stays local; the gathered K and V are used and dropped, and the backward gathers
        them again. In the backward every device contributes to every token's {m(r"\mathrm{d}K")}
        and {m(r"\mathrm{d}V")}, so those are partial sums that <strong>ReduceScatter</strong>
        back over the sequence, and weight gradients AllReduce over {X} as in data parallelism.""",
    ], "cp", f"Context parallelism over {mesh(X=4)}: K/V gathers in attention, a silent MLP, sequence-scattered dK/dV in the backward."),
    (3, "Expert parallelism", "ep", [
        f"""In a mixture-of-experts layer the MLP becomes {m("E")} experts, one per device,
        {t("W_in", ("E", "D", "F"), {"E": "Z"}, kind="weight")}, and a router sends each token to
        one of them. Tokens are drawn as squares colored by their expert. The <strong>AllToAll
        dispatch</strong> sends every device the tokens for its expert; the experts run an
        ordinary MLP; a second <strong>AllToAll</strong> sends every token home. The backward
        mirrors this: token gradients AllToAll out and back, expert weight gradients stay local
        because each expert owns its weights, and attention weight gradients AllReduce over the
        batch axis as in data parallelism.""",
    ], "ep", f"Expert parallelism over {mesh(Z=4)}: tokens sorted by expert in the AllToAll, expert MLPs, mirrored gradient AllToAlls."),
    (3, "Pipeline parallelism", "pp", [
        """Pipeline parallelism shards the layers: stage 0 owns layer 1, stage 1 owns layer 2, and
        activations cross the boundary with one point-to-point send, the cheapest communication
        here. The cost is idleness. With one batch each stage would wait on the other, so the
        batch is split into four microbatches that follow each other through the pipe. The
        schedule under the lanes fills as you step: the staircase is the pipe filling, the empty
        cells are the bubble. In the backward each cell is twice as wide, because backward is
        about twice the compute, so the bubble grows during training.""",
    ], "pp", "Pipeline parallelism, 2 stages × 4 microbatches: activations hop forward, gradients hop back, and the schedule shows who idles."),
    (2, "Examples from frontier models", "recipes", [
        f"""Training runs combine these axes. From recent reports
        ({a("DeepSeek-V3", "https://arxiv.org/abs/2412.19437")},
        {a("Kimi K2", "https://arxiv.org/abs/2507.20534")},
        {a("GLM-4.5V", "https://arxiv.org/abs/2507.01006")},
        {a("Nemotron-4", "https://arxiv.org/abs/2406.11704")},
        {a("Nemotron-H", "https://arxiv.org/abs/2504.03624")},
        {a("Nemotron 3", "https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf")},
        {a("Qwen3-VL", "https://arxiv.org/abs/2511.21631")},
        {a("Llama 3", "https://arxiv.org/abs/2407.21783")}) the recipes fall into three families.
        The figures use two devices per axis so the grids stay readable; the reported degrees are
        in the text.""",
        f"""The canvas changes here. Each pipeline stage is a grid of devices, arranged so that
        every mesh axis is a fixed direction: an FSDP partner is two rows away, a tensor-parallel
        partner is the next column, and so on. Every collective flies along exactly one axis,
        wrapped in that axis's color, with a badge under the grid and a tag in the program naming
        the parallelism responsible. The tally under the step bar counts collectives per axis.""",
    ], None, None),
    (3, "Dense models: Llama 3", "dense4d", [
        f"""<strong>FSDP × TP × CP × PP.</strong> Llama 3 405B trained with TP 8, PP 16 and FSDP 64
        at 8K tokens, then TP 8, <strong>CP 16</strong>, PP 16 and FSDP 8 at 128K: context
        parallelism is turned on only when sequences get long. Axes are ordered by bandwidth,
        TP within a server and data parallelism across the cluster. Nemotron-4 340B
        (TP 8 × PP 12 × DP) and Nemotron-H (TP 8 × DP 768, no PP) are the same family without
        the context axis. Here {mesh(X=2, Y=2, C=2, stage=2)} and
        {t("In", BTD, {"B": "X", "T": "C", "D": "Y"})}. Every matmul opens with FSDP's weight
        gather and TP's activation gather; CP speaks only in attention; the pipeline is one send
        per direction.""",
    ], "dense4d", f"Dense 4D over {mesh(X=2, Y=2, C=2, stage=2)}, Llama 3 style: FSDP × TP × CP × PP on 16 devices."),
    (3, "Open MoE models: DeepSeek and Kimi", "moe4d", [
        f"""<strong>EP × CP × PP × ZeRO-1, and no TP.</strong> DeepSeek-V3 trained with PP 16
        (DualPipe), EP 64 across 8 nodes and ZeRO-1 data parallelism, "without costly tensor
        parallelism"; Kimi K2 with PP 16, EP 16 and ZeRO-1. Neither reports context parallelism
        (V3 reached 128K with YaRN), but the long-context work that followed does: GLM-4.5V adds
        CP 4 for its long-context stage, and
        {a("DeepSeek-V4", "https://arxiv.org/abs/2606.19348")} and
        {a("Kimi K3", "https://arxiv.org/abs/2607.24653")} each describe context-parallel
        attention for million-token training. Here {mesh(X=2, C=2, Z=2, stage=2)} and
        {t("In", BTD, {"B": "XZ", "T": "C"})}: the batch is split over the data axis and the
        expert axis, since outside the MoE block the expert axis is more data parallelism, and
        the sequence over {C}. The forward has no weight gathers at all: K and V gather over
        {C}, the {op("AllToAll", "Z")} pair runs per MoE layer, and the activations hop between
        stages. In the backward an attention weight gradient scatters over {X} (ZeRO-1) and sums
        over {Z} and {C}; an expert weight gradient scatters over {X} and sums over {C} only,
        because each expert owns its weights. Then each device steps on its shard and the
        updated weights gather back.""",
    ], "moe4d", f"MoE 4D over {mesh(X=2, C=2, Z=2, stage=2)}, DeepSeek / Kimi style: EP × CP × PP × ZeRO-1, no tensor parallelism, 16 devices."),
    (3, "Everything at once: Nemotron 3 and Qwen3-VL", "5d", [
        f"""<strong>FSDP × TP × CP × EP × PP.</strong> Nemotron 3 Nano's long-context phase ran
        CP 8 × TP 8 × EP 8 × PP 4 (Nemotron 3 Ultra pushes EP to 128), and Qwen3-VL lists TP, PP,
        CP, EP and ZeRO-1 DP on up to 10,000 GPUs. Here, two of everything: 32 devices,
        {mesh(X=2, Y=2, C=2, Z=2, stage=2)}, {t("In", BTD, {"B": "XZ", "T": "C", "D": "Y"})}.
        Each stage is a 4 × 4 grid, rows {X} then {C}, columns {Z} then {Y}. The tally is the
        summary: FSDP and TP do most of the talking, CP speaks only in attention, EP is two
        AllToAlls per MoE layer, PP one send per direction. In the backward a weight gradient is
        summed over every axis that split the tokens, {X}, {Z} and {C}, so one gradient takes
        three collectives to settle.""",
    ], "5d", f"Everything at once over {mesh(X=2, Y=2, C=2, Z=2, stage=2)}, Nemotron 3 / Qwen3-VL style: 32 devices, every collective colored by the axis that causes it."),
]


def clean(par: str) -> str:
    return " ".join(textwrap.dedent(par).split())


def render_md() -> str:
    out = [DEK, ""]
    for level, head, _id, paras, fig, cap in SECTIONS:
        out.append(f"{'#' * level} {head}\n")
        for par in paras:
            out.append(clean(par) + "\n")
        if fig:
            out.append(f'<figure class="tpv-outer">\n  <tpviz-figure strategy="{fig}"></tpviz-figure>\n'
                       f'  <figcaption>\n    {cap}\n  </figcaption>\n</figure>\n')
    return "\n".join(out).rstrip("\n") + "\n"


def render_html() -> str:
    out = ['  <h1>A Visual Guide to Parallel Transformers</h1>',
           '  <p class="dek">\n' + textwrap.indent(textwrap.fill(DEK, 88), "    ") + '\n  </p>\n']
    for level, head, _id, paras, fig, cap in SECTIONS:
        out.append(f'  <h{level} id="{_id}">{head}</h{level}>')
        for par in paras:
            out.append("  <p>\n" + textwrap.indent(textwrap.fill(clean(par), 88), "    ") + "\n  </p>")
        if fig:
            out.append(f'  <figure class="tpv-outer">\n    <tpviz-figure strategy="{fig}"></tpviz-figure>\n'
                       f'    <figcaption>\n      {cap}\n    </figcaption>\n  </figure>')
        out.append("")
    return "\n".join(out)


def main():
    html = ARTICLE.read_text()
    i, j = html.index("<article>\n") + len("<article>\n"), html.index("\n  <footer>")
    ARTICLE.write_text(html[:i] + render_html() + html[j:])
    if POST.exists():
        md = POST.read_text()
        head = md[: md.index('<script src="/tpviz/tpviz.js" defer></script>\n') + len('<script src="/tpviz/tpviz.js" defer></script>\n')]
        POST.write_text(head + "\n" + render_md())
        print("wrote", POST)
    print("wrote", ARTICLE)


if __name__ == "__main__":
    main()
