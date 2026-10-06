"""Shrink the committed research figures without a visible change.

Run it after a study's ``run.py`` has rewritten its figures, before you commit them:

    .venv/bin/python research/shrink_figures.py                       # every research/*/figures/*
    .venv/bin/python research/shrink_figures.py research/<slug>/figures/x.svg ...

* SVG (matplotlib output): removes the ``<metadata>`` block and the comments, joins the lines, and
  rounds path data and ``x``/``y`` positions to 0.01 pt. Transforms and scale factors are not
  changed. A Chromium render of the result differs from the original only by sub-pixel shifts of
  antialiased edges.
* PNG: stores an opaque image with a 256-color palette (median cut, no dithering) when that file is
  smaller. The mean change per channel is below 0.2 of 255 on the study figures.

The script is idempotent: a second run changes nothing. It needs Pillow (a matplotlib dependency).
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parent
_NUM = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?|-?\d+(?:[eE][-+]?\d+)?")


def _round(m: re.Match[str]) -> str:
    v = round(float(m.group(0)), 2)
    return "0" if v == 0 else f"{v:.2f}".rstrip("0").rstrip(".")


def _numbers(m: re.Match[str]) -> str:
    body = _NUM.sub(_round, re.sub(r"\s+", " ", m.group(2)).strip())
    return f'{m.group(1)}"{body}"'


def shrink_svg(text: str) -> str:
    text = re.sub(r"\s*<metadata>.*?</metadata>", "", text, flags=re.S)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r'(\s(?:d|points|x|y)=)"([^"]*)"', _numbers, text)
    return re.sub(r">\s*\n\s*<", "><", text)


def shrink_png(data: bytes) -> bytes:
    from PIL import Image

    im = Image.open(io.BytesIO(data))
    if im.mode == "P":
        return data  # already a palette image (for example, from an earlier run)
    rgba = im.convert("RGBA")
    alpha = rgba.getchannel("A")
    if alpha.getextrema() != (255, 255):
        return data  # transparent: keep it lossless
    pal = rgba.convert("RGB").quantize(
        colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
    )
    out = io.BytesIO()
    kwargs = {"dpi": im.info["dpi"]} if "dpi" in im.info else {}
    pal.save(out, "PNG", optimize=True, **kwargs)
    return out.getvalue() if out.tell() < len(data) else data


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv] or sorted(
        p for p in RESEARCH.glob("*/figures/*") if p.suffix in (".svg", ".png")
    )
    before = after = 0
    for path in paths:
        data = path.read_bytes()
        if path.suffix == ".svg":
            new = shrink_svg(data.decode("utf-8")).encode("utf-8")
        elif path.suffix == ".png":
            new = shrink_png(data)
        else:
            continue
        before += len(data)
        after += len(new)
        if new != data:
            path.write_bytes(new)
            print(f"{len(data):>9,} -> {len(new):>9,}  {path}")
    print(f"{before:,} -> {after:,} bytes in {len(paths)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
