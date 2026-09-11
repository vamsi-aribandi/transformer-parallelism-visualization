"use strict";
/* The dumb player: binary-search each object's keyframe segments at time t,
   set transform/opacity. No semantic logic lives here. */

class Player {
  constructor(sceneEl) {
    this.sceneEl = sceneEl;
    this.rt = null;
    this.t = 0;
    this.playing = false;
    this.speed = 1;
    this._raf = null;
    this._last = 0;
    this.onTick = () => {};
  }

  load(rt) {
    this.rt = rt;
    this.sceneEl.textContent = "";
    const layers = new Map();
    const order = rt.objs
      .map((o, i) => ({ o, i }))
      .sort((a, b) => (a.o.spec.z - b.o.spec.z) || (a.i - b.i));
    for (const { o } of order) {
      const el = drawObject(o.spec);
      el.classList.add("obj");
      if (o.spec.tip !== undefined) {
        el.classList.add("hit");
        el.dataset.tip = o.spec.tip;
        el.dataset.tex = o.spec.tex;
      }
      o.el = el;
      this.sceneEl.appendChild(el);
    }
    // deck/wrect groups need bbox-based fitting after DOM insertion
    for (const o of this.rt.objs) {
      if (o.el.dataset.fitW) applyFit(o.el);
    }
    this.t = 0;
    this.seek(0);
  }

  static sample(track, t, fallback) {
    // track: sorted [{t0,t1,vals[5]}]; returns value at t or fallback
    let lo = 0, hi = track.length - 1, best = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (track[mid].t0 <= t) { best = mid; lo = mid + 1; } else hi = mid - 1;
    }
    if (best < 0) return fallback;
    const seg = track[best];
    const vals = seg.vals;
    if (t >= seg.t1) return vals[vals.length - 1];
    const u = ((t - seg.t0) / (seg.t1 - seg.t0)) * (vals.length - 1);
    const i = Math.min(Math.floor(u), vals.length - 2);
    return vals[i] + (vals[i + 1] - vals[i]) * (u - i);
  }

  seek(t) {
    const rt = this.rt;
    if (!rt) return;
    this.t = Math.max(0, Math.min(t, rt.tl.dur));
    for (const o of rt.objs) {
      const s = o.spec;
      const alive = this.t >= s.t0 && (this.t < s.t1 || s.t1 >= rt.tl.dur);
      if (!alive) { if (o.el.style.display !== "none") o.el.style.display = "none"; continue; }
      if (o.el.style.display === "none") o.el.style.display = "";
      const x = Player.sample(o.tracks.x, this.t, s.x);
      const y = Player.sample(o.tracks.y, this.t, s.y);
      const w = Player.sample(o.tracks.w, this.t, 100) / 100;
      const op = Player.sample(o.tracks.o, this.t, s.o) / 100;
      o.el.setAttribute(
        "transform",
        `translate(${SX(x).toFixed(1)},${SY(y).toFixed(1)})${w !== 1 ? ` scale(${w.toFixed(4)})` : ""}`
      );
      o.el.setAttribute("opacity", op.toFixed(3));
    }
    this.onTick(this.t);
  }

  play() {
    if (this.playing) return;
    this.playing = true;
    this._last = performance.now();
    const loop = (now) => {
      if (!this.playing) return;
      const dt = ((now - this._last) / 1000) * 100 * this.speed; // cs
      this._last = now;
      let t = this.t + dt;
      if (t >= this.rt.tl.dur) { t = this.rt.tl.dur; this.pause(); }
      this.seek(t);
      if (this.playing) this._raf = requestAnimationFrame(loop);
    };
    this._raf = requestAnimationFrame(loop);
  }

  pause() {
    this.playing = false;
    if (this._raf) cancelAnimationFrame(this._raf);
    this.onTick(this.t);
  }

  toggle() { this.playing ? this.pause() : this.play(); }
}
