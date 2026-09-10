import sys, av, pathlib
src, outdir, every = sys.argv[1], pathlib.Path(sys.argv[2]), float(sys.argv[3])
outdir.mkdir(parents=True, exist_ok=True)
container = av.open(src)
stream = container.streams.video[0]
next_t, i = 0.0, 0
for frame in container.decode(stream):
    t = float(frame.pts * stream.time_base)
    if t >= next_t:
        frame.to_image().save(outdir / f"f{i:03d}_t{t:05.1f}.png")
        next_t += every; i += 1
print(f"{i} frames -> {outdir}")
