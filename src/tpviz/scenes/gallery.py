"""Static component sheet for visual QA — render with -sql and inspect the PNG."""

from __future__ import annotations

from manim import DOWN, LEFT, RIGHT, UP, Scene, Text, VGroup

from tpviz import style
from tpviz.core.mesh import Mesh
from tpviz.core.tensors import LTensor
from tpviz.mobjects.device_grid import DeviceBox
from tpviz.mobjects.equation_strip import EquationStrip
from tpviz.mobjects.tensor_mobject import ActivationDeck, WeightRect
from tpviz.mobjects.labels import caption_text


def _cell(vis, label_text: str) -> VGroup:
    label = vis.make_label(font_size=20)
    tag = caption_text(label_text, font_size=16, color=style.MUTED_TEXT)
    tag.next_to(label, DOWN, buff=0.08)
    return VGroup(vis, label, tag)


class Gallery(Scene):
    def construct(self):
        mesh = Mesh({"X": 4})
        mesh_y = Mesh({"Y": 4})

        acts = [
            _cell(ActivationDeck(LTensor("In", ("B", "T", "D"), {}), mesh, sharded=False), "full"),
            _cell(ActivationDeck(LTensor("In", ("B", "T", "D"), {"B": "X"}), mesh), "B sharded (DP)"),
            _cell(ActivationDeck(LTensor("In", ("B", "T", "D"), {"T": "X"}), mesh), "T sharded (CP)"),
            _cell(ActivationDeck(LTensor("In", ("B", "T", "D"), {"D": "Y"}), mesh_y), "D sharded (TP)"),
            _cell(
                ActivationDeck(
                    LTensor("Out", ("B", "T", "D"), {}, partial=frozenset({"Y"})), mesh_y
                ),
                "partial sum",
            ),
            _cell(
                ActivationDeck(LTensor("K", ("B", "T", "H"), {"T": "X"}, kind="kv"), mesh),
                "K/V (kv kind)",
            ),
        ]
        row1 = VGroup(*acts).arrange(RIGHT, buff=0.7, aligned_edge=UP)

        weights = [
            _cell(WeightRect(LTensor("W_in", ("D", "F"), {}, kind="weight"), mesh, sharded=False), "full"),
            _cell(WeightRect(LTensor("W_in", ("D", "F"), {"D": "X"}, kind="weight"), mesh), "FSDP shard"),
            _cell(WeightRect(LTensor("W_in", ("D", "F"), {"F": "Y"}, kind="weight"), mesh_y), "TP shard"),
            _cell(WeightRect(LTensor("W_out", ("F", "D"), {}, kind="weight"), mesh, sharded=False), "tall [F, D]"),
        ]
        row2 = VGroup(*weights).arrange(RIGHT, buff=0.8, aligned_edge=UP)

        # per-device offsets: the SAME tensor on devices 0..3 shows different slices
        w_shard = LTensor("W_in", ("D", "F"), {"D": "X"}, kind="weight")
        t_shard = LTensor("In", ("B", "T", "D"), {"T": "X"})
        b_shard = LTensor("In", ("B", "T", "D"), {"B": "X"})
        offsets = (
            [_cell(WeightRect(w_shard, mesh, device=i), f"dev {i}") for i in range(4)]
            + [_cell(ActivationDeck(t_shard, mesh, device=i), f"T dev {i}") for i in range(4)]
            + [_cell(ActivationDeck(b_shard, mesh, device=i), f"B dev {i}") for i in range(4)]
        )
        row_offsets = VGroup(*offsets).arrange(RIGHT, buff=0.55, aligned_edge=UP)

        device = DeviceBox(1, width=2.6, height=3.0)
        w = WeightRect(LTensor("W_in", ("D", "F"), {"D": "X"}, kind="weight"), mesh)
        wl = w.make_label(font_size=18)
        device.fit_into(VGroup(w, wl), "weights", max_scale=0.9)
        a = ActivationDeck(LTensor("In", ("B", "T", "D"), {"B": "X"}), mesh)
        al = a.make_label(font_size=18)
        device.fit_into(VGroup(a, al), "acts", max_scale=0.9)
        row3 = VGroup(_cell_title("DeviceBox", device))

        sheet = VGroup(row1, row2, row_offsets, row3).arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        sheet.scale_to_fit_width(13.4)
        if sheet.height > 6.2:
            sheet.scale_to_fit_height(6.2)
        sheet.to_edge(UP, buff=0.35)
        self.add(sheet)

        strip = EquationStrip()
        self.add(strip)
        strip.show(
            r"\text{AllGather}_{Y}\; \mathrm{In}[B,\, T,\, D_{Y}] \,\to\, \mathrm{In}[B,\, T,\, D]",
            color=style.COMM_COLOR,
        )


def _cell_title(label_text: str, mobj) -> VGroup:
    tag = caption_text(label_text, font_size=16, color=style.MUTED_TEXT)
    tag.next_to(mobj, DOWN, buff=0.1)
    return VGroup(mobj, tag)
