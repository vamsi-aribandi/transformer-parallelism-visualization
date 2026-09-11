"use strict";
/* Wire everything: tabs, mode toggle, counters, URL state, QA still mode. */

(function main() {
  installGlyphs();

  const sceneEl = document.getElementById("scene");
  const stageEl = document.getElementById("stage");
  const player = new Player(sceneEl);
  const program = new Program(document.getElementById("program-list"), (t) => {
    player.pause();
    player.seek(t);
  });
  const transport = new Transport(player);
  initTooltip(stageEl, document.getElementById("canvas"));

  const strategies = ["dp", "fsdp", "tp", "cp", "pp", "ep"];
  const tabsEl = document.getElementById("tabs");
  const modeEl = document.getElementById("mode-toggle");
  const chipColl = document.querySelector("#chip-coll b");
  const chipMM = document.querySelector("#chip-mm b");
  const meshEl = document.getElementById("mesh-eq");
  const state = { s: "tp", m: "train" };

  function available(s, m) {
    return DOC.timelines[`${s}_${m}`] !== undefined;
  }

  for (const s of strategies) {
    const b = document.createElement("button");
    b.textContent = s.toUpperCase();
    b.dataset.s = s;
    b.disabled = !available(s, "fwd") && !available(s, "train");
    if (b.disabled) b.title = "coming soon";
    b.addEventListener("click", () => setState(s, state.m));
    tabsEl.appendChild(b);
  }
  modeEl.querySelectorAll("button").forEach((b) =>
    b.addEventListener("click", () => setState(state.s, b.dataset.mode))
  );

  function setState(s, m) {
    if (!available(s, m)) m = available(s, "train") ? "train" : "fwd";
    if (!available(s, m)) return;
    state.s = s;
    state.m = m;
    tabsEl.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b.dataset.s === s));
    modeEl.querySelectorAll("button").forEach((b) => {
      b.classList.toggle("active", b.dataset.mode === m);
      b.disabled = !available(s, b.dataset.mode);
    });
    const rt = buildTimeline(`${s}_${m}`);
    player.pause();
    player.load(rt);
    program.build(rt.tl);
    transport.load(rt.tl);
    const meta = DOC.meta.strategies[s];
    meshEl.textContent = "";
    if (meta) meshEl.appendChild(eqSvg(meta.mesh, 15));
    document.getElementById("program-sub").textContent = m === "train" ? "forward + backward" : "forward";
    writeHash();
  }

  player.onTick = (t) => {
    if (!player.rt) return;
    transport.tick(t);
    program.update(t);
    const cs = player.rt.tl.counters;
    let coll = 0, mm = 0;
    for (const [tc, c, m] of cs) { if (tc <= t) { coll = c; mm = m; } else break; }
    chipColl.textContent = coll;
    chipMM.textContent = mm;
  };

  // legend
  const legend = document.getElementById("legend");
  document.getElementById("legend-btn").addEventListener("click", () => {
    legend.hidden = !legend.hidden;
  });

  // URL state + QA still mode
  function readHash() {
    const p = new URLSearchParams(location.hash.slice(1));
    return {
      s: p.get("s") || "tp",
      m: p.get("m") || "train",
      t: p.get("t") ? Math.round(parseFloat(p.get("t")) * 100) : 0,
      still: p.get("still") === "1",
      autoplay: p.get("play") === "1",
    };
  }
  function writeHash() {
    const t = (player.t / 100).toFixed(1);
    history.replaceState(null, "", `#s=${state.s}&m=${state.m}&t=${t}`);
  }
  let hashTimer = null;
  const origPause = player.pause.bind(player);
  player.pause = () => { origPause(); clearTimeout(hashTimer); hashTimer = setTimeout(writeHash, 150); };

  const h = readHash();
  setState(h.s, h.m);
  if (h.t) player.seek(h.t);
  if (h.autoplay) player.play();
  if (h.still) {
    document.fonts.ready.then(() => {
      player.pause();
      document.title = "ready";
    });
  } else if (!h.t && !h.autoplay) {
    // gentle default: start playing after fonts settle
    document.fonts.ready.then(() => setTimeout(() => player.play(), 400));
  }

  // keyboard: strategy/mode switching
  document.addEventListener("keydown", (ev) => {
    const i = "123456".indexOf(ev.key);
    if (i >= 0 && !tabsEl.children[i].disabled) setState(strategies[i], state.m);
    else if (ev.key === "f") setState(state.s, "fwd");
    else if (ev.key === "t") setState(state.s, "train");
  });
})();
