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

DEK = (f"How a transformer's forward and backward passes are sharded across devices, first one "
       f"strategy at a time and then in the combinations used to train recent models. The "
       f"notation follows the {a('JAX scaling book', BOOK)}.")

SUMMARY = ("How a transformer's forward and backward passes are sharded across devices, one strategy "
           "at a time and then in the combinations used to train recent models, in the notation of "
           "the JAX scaling book.")

# (level, heading, id, paragraphs, figure strategy or None, caption)
SECTIONS = [
    (2, "Notation", "notation", [
        f"""We use the {a("sharding notation", BOOK + "sharding/")} from the scaling book. A mesh
        such as {mesh(X=4)} is a set of four devices along one axis named {X}. A subscript on a
        dimension gives the mesh axis that dimension is sharded over. For example,
        {t("In", BTD, {"B": "X"})} is sharded along the batch over {X}, so each device holds an
        array of shape {m(r"[B/4,\, T,\, D]")}. A dimension with no subscript is replicated. A
        trailing {m(r"\{U_X\}")} marks a partial sum: each device holds an array of the full
        shape, and the true value is the sum of these arrays over {X}. AllReduce and
        ReduceScatter compute this sum.""",
        f"""The tooltips assume {m("B=8")}, {m("T=128")}, {m("D=1024")}, {m("F=4096")},
        {m("H=1024")} (all attention heads concatenated), {m("E=4")} experts with {m("S=256")}
        tokens routed to each, and bf16.""",
    ], None, None),
    (2, "Reading the figures", "figures", [
        """The model runs from left to right: <span class="meq"><span class="mu">In</span></span>,
        then the attention and MLP blocks of each layer, then
        <span class="meq"><span class="mu">Out</span></span>. Each device is a horizontal lane, so
        tensors sent between devices move vertically. Weights are <span class="sw wt"></span>
        blue, activations <span class="sw act"></span> amber, keys and values
        <span class="sw kv"></span> teal, and gradients <span class="sw grad"></span> rose. A
        dashed outline is the full array and the solid part is the shard held by that device.
        Hovering over a tensor shows its shape, sharding and memory.""",
        """Each segment of the step bar is one operation. Click a segment to see the state after
        that operation, or press play. For a collective, the panel below the canvas shows how it
        runs on a ring of devices. The program on the right lists every operation. The figures
        start with the forward pass; <em>+ Backward</em> adds the backward pass, including the
        activations saved for it, and links each backward operation to the forward operation it
        comes from.""",
    ], None, None),
    (2, "The axes of transformer sharding", "axes", [
        """Each strategy below shards one thing: the batch, the optimizer state, the weights,
        the feature dimensions, the sequence, the experts, or the layers. Each runs on four
        devices.""",
    ], None, None),
    (3, "Data parallelism", "dp", [
        f"""The batch is sharded, {t("In", BTD, {"B": "X"})}, and the weights are replicated. The
        forward pass needs no communication. In the backward pass, each device computes the
        weight gradients from its part of the batch, so each gradient
        {t("dW", ("D", "F"), partial="X", kind="grad")} is a partial sum. It is AllReduced over
        {X} before the optimizer step.""",
    ], "dp", f"Data parallelism over {mesh(X=4)}. The forward pass has no communication; the backward pass has one AllReduce per weight gradient."),
    (3, "ZeRO-1", "zero1", [
        f"""Data parallelism also replicates the optimizer state, which for Adam is two more
        arrays the size of the weights. ZeRO-1 shards the optimizer state over {X} and keeps the
        weights replicated. The forward pass is unchanged. In the backward pass, each gradient
        {t("dW", ("D", "F"), partial="X", kind="grad")} is ReduceScattered instead of AllReduced,
        giving {t("dW", ("D", "F"), {"D": "X"}, kind="grad")}. Each device updates its shard of
        the weights, {t("W", ("D", "F"), {"D": "X"}, kind="weight")}, and an AllGather gives every
        device the full updated weights. An AllReduce is a ReduceScatter followed by an
        AllGather, so ZeRO-1 has the same communication cost as data parallelism.""",
    ], "zero1", f"ZeRO-1 over {mesh(X=4)}. Each weight gradient is ReduceScattered, each device updates its shard, and the updated shards are AllGathered."),
    (3, "Fully-sharded data parallelism (ZeRO-3)", "fsdp", [
        f"""FSDP also shards the weights over {X}:
        {t("W_in", ("D", "F"), {"D": "X"}, kind="weight")} and
        {t("W_out", ("F", "D"), {"D": "X"}, kind="weight")}. Each weight is AllGathered just
        before its matmul and freed just after. The backward pass AllGathers each weight again,
        and each weight gradient is ReduceScattered so that it is sharded like its weight.""",
    ], "fsdp", f"FSDP over {mesh(X=4)}. Weights are AllGathered in both passes, and weight gradients are ReduceScattered in the backward pass."),
    (3, "Tensor parallelism", "tp", [
        f"""Tensor parallelism shards the attention heads,
        {t("W_qkv", ("D", "H"), {"H": "Y"}, kind="weight")} and
        {t("W_o", ("H", "D"), {"H": "Y"}, kind="weight")}, and the MLP hidden dimension,
        {t("W_in", ("D", "F"), {"F": "Y"}, kind="weight")} and
        {t("W_out", ("F", "D"), {"F": "Y"}, kind="weight")}. Activations are sharded along
        {m("D")}. Each block starts with an AllGather and ends with a ReduceScatter. The first
        matmul contracts over {m("D")}, so the activations are AllGathered first. The second
        matmul contracts over the sharded {m("H")} or {m("F")}, so each device holds a partial sum
        {m(r"\{U_Y\}")}. The ReduceScatter sums it and shards the result along {m("D")} for the
        next block. The backward pass does the same in reverse.""",
    ], "tp", f"Tensor parallelism over {mesh(Y=4)}. Each block has one AllGather and one ReduceScatter in each pass."),
    (3, "Context parallelism", "cp", [
        f"""Context parallelism shards the sequence: {t("In", BTD, {"T": "X"})}. The MLP acts on
        each token separately and needs no communication. In attention, each query needs every
        key and value, so {m("K")} and {m("V")} are AllGathered over {X} while {m("Q")} stays
        sharded. The backward pass AllGathers {m("K")} and {m("V")} again. Each device computes
        gradients for all keys and values, so {m(r"\mathrm{d}K")} and {m(r"\mathrm{d}V")} are
        partial sums and are ReduceScattered back over {m("T")}. The weight gradients are
        AllReduced over {X}, as in data parallelism.""",
    ], "cp", f"Context parallelism over {mesh(X=4)}. K and V are AllGathered in attention; dK and dV are ReduceScattered in the backward pass."),
    (3, "Expert parallelism", "ep", [
        f"""In a mixture-of-experts layer, the MLP is replaced by {m("E")} experts, one per
        device, {t("W_in", ("E", "D", "F"), {"E": "Z"}, kind="weight")}, and a router assigns each
        token to one expert. Tokens are drawn as squares colored by their expert. An AllToAll
        sends each token to the device that holds its expert, each expert runs its MLP, and a
        second AllToAll sends the tokens back. In the backward pass, the token gradients go
        through the same two AllToAlls. The expert weight gradients need no communication, since
        each device holds its own expert. The attention weight gradients are AllReduced over
        {Z}, as in data parallelism.""",
    ], "ep", f"Expert parallelism over {mesh(Z=4)}. Each MoE layer has two AllToAlls in each pass."),
    (3, "Pipeline parallelism", "pp", [
        """Pipeline parallelism shards the layers: stage 0 holds layer 1 and stage 1 holds
        layer 2. The activations are sent from one stage to the next with a point-to-point send.
        With a single batch, one stage is always idle, so the batch is split into four
        microbatches that go through the stages in turn. The grid below the lanes shows which
        microbatch each stage runs at each time step. The empty cells are idle time, called the
        bubble. The backward pass takes about twice the compute of the forward pass, so its
        cells are twice as wide.""",
    ], "pp", "Pipeline parallelism with 2 stages and 4 microbatches."),
    (2, "Examples from frontier models", "recipes", [
        f"""Large training runs combine several of these strategies. The technical reports for
        {a("DeepSeek-V3", "https://arxiv.org/abs/2412.19437")},
        {a("Kimi K2", "https://arxiv.org/abs/2507.20534")},
        {a("GLM-4.5V", "https://arxiv.org/abs/2507.01006")},
        {a("Nemotron-4", "https://arxiv.org/abs/2406.11704")},
        {a("Nemotron-H", "https://arxiv.org/abs/2504.03624")},
        {a("Nemotron 3", "https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf")},
        {a("Qwen3-VL", "https://arxiv.org/abs/2511.21631")} and
        {a("Llama 3", "https://arxiv.org/abs/2407.21783")} use three combinations. The figures use
        two devices per axis; the sizes used in the reports are given in the text.""",
        """In these figures, each pipeline stage is a grid of devices. The row and column labels
        give each device's coordinate along each mesh axis. Each collective runs over one axis
        and is drawn in that axis's color, and the label under the grid names the axis and the
        strategy. The row under the step bar counts the collectives on each axis.""",
    ], None, None),
    (3, "Dense models: Llama 3", "dense4d", [
        f"""<strong>FSDP × TP × CP × PP.</strong> Llama 3 405B used TP 8, PP 16 and FSDP 64 for
        8K-token sequences, and TP 8, CP 16, PP 16 and FSDP 8 for 128K-token sequences. TP is
        placed within a server, where bandwidth is highest, and data parallelism across servers.
        Nemotron-4 340B used TP 8, PP 12 and data parallelism; Nemotron-H used TP 8 and 768-way
        data parallelism. The figure uses {mesh(X=2, Y=2, C=2, stage=2)} with
        {t("In", BTD, {"B": "X", "T": "C", "D": "Y"})}. Each matmul has an FSDP AllGather of the
        weight and a TP AllGather of the activations. CP adds the K and V AllGathers in
        attention, and PP adds one send in each direction.""",
    ], "dense4d", f"FSDP × TP × CP × PP over {mesh(X=2, Y=2, C=2, stage=2)}, 16 devices."),
    (3, "Open MoE models: DeepSeek and Kimi", "moe4d", [
        f"""<strong>EP × CP × PP × ZeRO-1, without TP.</strong> DeepSeek-V3 used PP 16, EP 64 across
        8 nodes and ZeRO-1, and no tensor parallelism. Kimi K2 used PP 16, EP 16 and ZeRO-1.
        Neither report uses context parallelism; DeepSeek-V3 extends its context to 128K with
        YaRN. GLM-4.5V adds CP 4 for its long-context stage, and
        {a("DeepSeek-V4", "https://arxiv.org/abs/2606.19348")} and
        {a("Kimi K3", "https://arxiv.org/abs/2607.24653")} both use context parallelism for
        long-context training. The figure uses {mesh(X=2, C=2, Z=2, stage=2)} with
        {t("In", BTD, {"B": "XZ", "T": "C"})}. The batch is sharded over both {X} and {Z}, since
        outside the MoE layers the expert axis is another data axis. The forward pass has no
        weight AllGathers. It has the K and V AllGathers over {C}, two AllToAlls over {Z} per MoE
        layer, and the send between stages. In the backward pass, each attention weight gradient
        is ReduceScattered over {X} and AllReduced over {Z} and {C}. Each expert weight gradient
        is ReduceScattered over {X} and AllReduced over {C} only, since each expert is held at a
        single {Z} coordinate.""",
    ], "moe4d", f"EP × CP × PP × ZeRO-1 over {mesh(X=2, C=2, Z=2, stage=2)}, 16 devices."),
    (3, "All five: Nemotron 3 and Qwen3-VL", "5d", [
        f"""<strong>FSDP × TP × CP × EP × PP.</strong> The long-context stage of Nemotron 3 Nano
        used CP 8, TP 8, EP 8 and PP 4, and Nemotron 3 Ultra used EP 128. Qwen3-VL uses TP, PP,
        CP, EP and ZeRO-1 on up to 10,000 GPUs. The figure uses 32 devices,
        {mesh(X=2, Y=2, C=2, Z=2, stage=2)}, with
        {t("In", BTD, {"B": "XZ", "T": "C", "D": "Y"})}. Each stage is a 4 × 4 grid with rows
        {X} and {C} and columns {Z} and {Y}. Most collectives are the FSDP and TP AllGathers and
        ReduceScatters. CP communicates only in attention, EP adds two AllToAlls per MoE layer,
        and PP adds one send in each direction. In the backward pass, each attention weight
        gradient is summed over {X}, {Z} and {C}, the three axes that shard the tokens, which
        takes three collectives.""",
    ], "5d", f"FSDP × TP × CP × EP × PP over {mesh(X=2, Y=2, C=2, Z=2, stage=2)}, 32 devices."),
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
        head = re.sub(r'^summary: ".*"$', f'summary: "{SUMMARY}"', head, flags=re.M)
        POST.write_text(head + "\n" + render_md())
        print("wrote", POST)
    print("wrote", ARTICLE)


if __name__ == "__main__":
    main()
