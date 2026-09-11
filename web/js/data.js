"use strict";
/* Parse the embedded document and build fast runtime indexes. */

const DOC = JSON.parse(document.getElementById("tpviz-data").textContent);
const FRAME_W = DOC.frame[0], FRAME_H = DOC.frame[1];
const PAL = DOC.palette;

/* manim scene coords (centi-units, y-up, origin center) -> svg */
const SX = (v) => v + FRAME_W / 2;
const SY = (v) => FRAME_H / 2 - v;

function axisCoord(mesh, axis, device) {
  let rem = device;
  const names = Object.keys(mesh).reverse();
  for (const name of names) {
    const size = mesh[name];
    const coord = rem % size;
    if (name === axis) return coord;
    rem = Math.floor(rem / size);
  }
  return 0;
}

function shardFrac(tensor, dim) {
  const ax = tensor.shard[dim];
  if (!ax || !tensor.sharded) return [1, 0];
  const n = tensor.mesh[ax] || 1;
  return [1 / n, axisCoord(tensor.mesh, ax, tensor.device) / n];
}

/* Build a runtime timeline: per-object track lists sorted by t0. */
function buildTimeline(name) {
  const tl = DOC.timelines[name];
  const objs = tl.objects.map((spec) => ({
    spec,
    el: null,
    tracks: { x: [], y: [], w: [], o: [] },
  }));
  for (const row of tl.tracks) {
    const [idx, prop, t0, t1, vals] = row;
    objs[idx].tracks[prop].push({ t0, t1, vals });
  }
  return { name, tl, objs };
}

function fmtClock(cs) {
  const s = Math.max(0, Math.round(cs / 100));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
