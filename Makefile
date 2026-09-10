export PATH := $(HOME)/Library/TinyTeX/bin/universal-darwin:$(PATH)

SCENES = dp:DPScene fsdp:FSDPScene tp:TPScene cp:CPScene pp:PPScene ep:EPScene

# usage: make lq S=dp C=DPScene
lq:
	uv run manim render -ql src/tpviz/scenes/$(S).py $(C)

snap:
	uv run manim render -sqh src/tpviz/scenes/$(S).py $(C)

gallery:
	uv run manim render -sql src/tpviz/scenes/gallery.py Gallery

# extract a frame every 3s from the last -ql render for visual QA
frames:
	uv run python scripts_extract_frames.py media/videos/$(S)/480p15/$(C).mp4 qa/$(S) 3.0

test:
	uv run pytest -q

hq-all:
	mkdir -p renders
	for s in $(SCENES); do \
		short=$${s%%:*}; cls=$${s##*:}; \
		uv run manim render -qh src/tpviz/scenes/$$short.py $$cls || exit 1; \
		cp media/videos/$$short/1080p60/$$cls.mp4 renders/$$short.mp4; \
	done

.PHONY: lq snap gallery frames test hq-all
