"use strict";
/* Transport bar: slider with forward/backward regions and phase ticks,
   play/pause, step buttons, speed, keyboard, clock. */

class Transport {
  constructor(player) {
    this.player = player;
    this.slider = document.getElementById("slider");
    this.clock = document.getElementById("clock");
    this.playBtn = document.getElementById("btn-play");
    this.regions = document.getElementById("slider-regions");
    this.ticks = document.getElementById("slider-ticks");
    this.scrubLabel = document.getElementById("scrub-label");
    this.steps = [];

    this.slider.addEventListener("input", () => {
      player.pause();
      player.seek(+this.slider.value);
    });
    this.playBtn.addEventListener("click", () => player.toggle());
    document.getElementById("btn-prev").addEventListener("click", () => this.step(-1));
    document.getElementById("btn-next").addEventListener("click", () => this.step(1));
    document.getElementById("speed").addEventListener("change", (e) => {
      player.speed = +e.target.value;
    });
    this.slider.addEventListener("pointermove", (ev) => this.preview(ev));
    this.slider.addEventListener("pointerleave", () => { this.scrubLabel.hidden = true; });

    document.addEventListener("keydown", (ev) => {
      if (ev.target.tagName === "SELECT" || ev.target.tagName === "INPUT") ev.target.blur();
      if (ev.code === "Space") { ev.preventDefault(); player.toggle(); }
      else if (ev.key === "ArrowLeft" && ev.shiftKey) player.seek(player.t - 100);
      else if (ev.key === "ArrowRight" && ev.shiftKey) player.seek(player.t + 100);
      else if (ev.key === "ArrowLeft") this.step(-1);
      else if (ev.key === "ArrowRight") this.step(1);
      else if (ev.key === "Home") player.seek(0);
      else if (ev.key === "End") player.seek(player.rt.tl.dur);
    });
  }

  load(tl) {
    this.steps = tl.steps;
    this.slider.max = tl.dur;
    this.slider.value = 0;
    // forward/backward regions
    this.regions.textContent = "";
    let bwdStart = null;
    for (const st of tl.steps) if (st.bwd) { bwdStart = st.t0; break; }
    const mk = (a, b, color, op) => {
      const d = document.createElement("div");
      d.className = "region";
      d.style.left = `${(a / tl.dur) * 100}%`;
      d.style.width = `${((b - a) / tl.dur) * 100}%`;
      d.style.background = color;
      d.style.opacity = op;
      this.regions.appendChild(d);
    };
    if (bwdStart !== null) {
      mk(0, bwdStart, "var(--act)", 0.3);
      mk(bwdStart, tl.dur, "var(--grad)", 0.35);
    } else mk(0, tl.dur, "var(--act)", 0.3);
    // phase ticks
    this.ticks.textContent = "";
    for (const [, t0] of tl.sections.map((s) => [s[0], s[1]])) {
      if (t0 <= 0) continue;
      const d = document.createElement("div");
      d.className = "tick";
      d.style.left = `${(t0 / tl.dur) * 100}%`;
      this.ticks.appendChild(d);
    }
  }

  step(dir) {
    const t = this.player.t;
    const ts = this.steps;
    if (!ts.length) return;
    let target;
    if (dir > 0) target = ts.find((s) => s.t0 > t + 1);
    else {
      const past = ts.filter((s) => s.t0 < t - 5);
      target = past[past.length - 1];
    }
    this.player.pause();
    this.player.seek(target ? target.t0 + 1 : dir > 0 ? this.player.rt.tl.dur : 0);
  }

  preview(ev) {
    const r = this.slider.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width));
    const t = frac * +this.slider.max;
    let label = "";
    for (const st of this.steps) if (st.t0 <= t) label = `${st.bwd ? "◀ " : ""}L${st.layer} · ${st.phase} · ${st.kind}`;
    if (!label) { this.scrubLabel.hidden = true; return; }
    this.scrubLabel.textContent = label;
    this.scrubLabel.style.left = `${frac * 100}%`;
    this.scrubLabel.hidden = false;
  }

  tick(t) {
    this.slider.value = t;
    this.clock.textContent = `${fmtClock(t)} / ${fmtClock(+this.slider.max)}`;
    this.playBtn.textContent = this.player.playing ? "❚❚" : "▶";
  }
}
