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
	uv run python -m tpviz.web.export --only dp_train,zero1_train,fsdp_train,tp_train,cp_train,ep_train,pp_train,dense4d_train,moe3d_train,5d_train

# re-record ONE timeline and merge it into the existing dist/data.json
# usage: make web-one T=5d_train
web-one:
	uv run python -m tpviz.web.export --only $(T)

web-all:
	uv run python -m tpviz.web.export

# rebuild dist/index.html + data.js from the existing data.json (no re-record)
web-page:
	uv run python -m tpviz.web.export --page-only

# sync the embeddable figure assets into the personal site's blog post
SITE = ../vamsi-aribandi.github.io
site: web-page
	cp web/style.css   $(SITE)/static/tpviz/tpviz.css
	cp web/tpviz.js    $(SITE)/static/tpviz/tpviz.js
	cp web/dist/data.js $(SITE)/static/tpviz/data.js

CHROME = /Applications/Google Chrome.app/Contents/MacOS/Google Chrome
# usage: make web-qa S=5d M=train T=42 [THEME=dark]  — screenshots ONE figure
# (fig= isolates it on the page) at step T in the given mode
M ?= fwd
THEME ?= dark
web-qa:
	mkdir -p qa/web
	"$(CHROME)" --headless --disable-gpu --window-size=1500,1150 \
		--screenshot=qa/web/$(S)-$(M)-step$(T)-$(THEME).png --virtual-time-budget=6000 \
		"file://$(PWD)/web/dist/index.html#fig=$(S)&mode=$(M)&step=$(T)&theme=$(THEME)"

# the 5D combination: 480p iteration render / 1080p60 final
lq-5d:
	uv run manim render -ql --disable_caching src/tpviz/scenes/five_d.py FiveDScene

hq-5d:
	mkdir -p renders
	uv run manim render -qh --disable_caching src/tpviz/scenes/five_d.py FiveDScene
	cp media/videos/five_d/1080p60/FiveDScene.mp4 renders/5d.mp4

.PHONY: web web-one web-all web-page site web-qa lq-5d hq-5d
