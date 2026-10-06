#!/usr/bin/env bash
# Rebuild every brand asset from source. Run from anywhere:  bash docs/brand/scripts/build.sh
# Needs: the repo .venv (numpy, contourpy via matplotlib, fontTools), node + web/node_modules
# (Playwright, KaTeX) for PNG renders, and ffmpeg (libwebp) for the optional animated WebP.
# Raster fallbacks of the hero (PNG, WebP) go to docs/brand/build/, which git ignores.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
$PY docs/brand/scripts/palette_check.py > /dev/null && echo "palette: all gates pass"
$PY docs/brand/scripts/make_logo.py
$PY docs/brand/scripts/make_hero.py "$@"
$PY docs/brand/scripts/make_social.py
$PY docs/brand/scripts/make_sheet.py
# Live counts for the portal concept and the README (never typed by hand).
$PY docs/brand/scripts/readme_facts.py --portal-js docs/brand/portal/facts.js
node docs/brand/scripts/render.mjs docs/brand/portal/home-concept.html docs/brand/portal/home-concept.png 1440 1900 --scale 1
echo; echo "README 'What's inside' table:"; $PY docs/brand/scripts/readme_facts.py
echo; echo "README 'Research' table:"; $PY docs/brand/scripts/readme_facts.py --research
