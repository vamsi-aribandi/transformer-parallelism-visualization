export PATH := $(HOME)/Library/TinyTeX/bin/universal-darwin:$(PATH)

SCENES = dp:DPScene fsdp:FSDPScene tp:TPScene cp:CPScene pp:PPScene ep:EPScene

# usage: make lq S=dp C=DPScene
lq:
	uv run manim render -ql --disable_caching src/tpviz/scenes/$(S).py $(C)

snap:
	uv run manim render -sqh src/tpviz/scenes/$(S).py $(C)

gallery:
	uv run manim render -sql src/tpviz/scenes/gallery.py Gallery

# extract a frame every 3s from the last -ql render for visual QA
frames:
	uv run python scripts/extract_frames.py media/videos/$(S)/480p15/$(C).mp4 qa/$(S) 3.0

test:
	uv run pytest -q

hq-all:
	mkdir -p renders
	for s in $(SCENES); do \
		short=$${s%%:*}; cls=$${s##*:}; \
		uv run manim render -qh --disable_caching src/tpviz/scenes/$$short.py $$cls || exit 1; \
		cp media/videos/$$short/1080p60/$$cls.mp4 renders/$$short.mp4; \
	done

.PHONY: lq snap gallery frames test hq-all

TRAIN_SCENES = dp_train:DPTrainScene fsdp_train:FSDPTrainScene tp_train:TPTrainScene cp_train:CPTrainScene pp_train:PPTrainScene ep_train:EPTrainScene

# forward+backward videos: 480p only until the look is signed off
lq-train-all:
	for s in $(TRAIN_SCENES); do \
		short=$${s%%:*}; cls=$${s##*:}; \
		uv run manim render -ql --disable_caching src/tpviz/scenes/$$short.py $$cls || exit 1; \
	done

.PHONY: lq-train-all

# ---- interactive web player ----
web:
	uv run python -m tpviz.web.export --only dp_train,fsdp_train,tp_train,cp_train,ep_train,pp_train

web-all:
	uv run python -m tpviz.web.export

CHROME = /Applications/Google Chrome.app/Contents/MacOS/Google Chrome
# usage: make web-qa S=tp M=train T=42.5
web-qa:
	mkdir -p qa/web
	"$(CHROME)" --headless --disable-gpu --window-size=1600,1000 \
		--screenshot=qa/web/step$(T).png --virtual-time-budget=4000 \
		"file://$(PWD)/web/dist/index.html#step=$(T)&theme=dark"

.PHONY: web web-all web-qa
