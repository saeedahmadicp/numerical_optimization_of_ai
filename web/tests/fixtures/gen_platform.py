"""Write tests/fixtures/platform_python.json: does this Python round like the canonical one?

Run from the repo root (``npm run gen:test-fixtures`` runs it with the other generators)::

    .venv/bin/python web/tests/fixtures/gen_platform.py

The TypeScript ports replay the arithmetic of the Python environment that wrote the committed
parity fixtures in web/src/generated/ (aarch64 Linux, the OpenBLAS of the NumPy wheel), some of
it bit for bit (the dgemv lane order of OpenBLAS, LAPACK's dgelsd and dsteqr). The Python
reference dumps under web/tests/**/fixtures are written on the machine that runs the tests, so
they match the ports bit for bit only when that machine rounds like the canonical one. The
test: a fresh ``numopt export`` is byte-identical to the committed files. ``canonical`` is then
true and the web tests compare the dumps exactly; otherwise (an x86-64 CI runner, another BLAS
or NumPy) they compare them with the tolerances each test states (tests/fixtures/platform.ts).
"""

from __future__ import annotations

import json
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np

from numopt.export import export

WEB = Path(__file__).resolve().parents[2]
COMMITTED = WEB / "src" / "generated"
OUT = Path(__file__).with_name("platform_python.json")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        written = [p.relative_to(tmp) for p in export(Path(tmp))]
        differ = sorted(
            str(rel)
            for rel in written
            if not (COMMITTED / rel).exists()
            or (COMMITTED / rel).read_bytes() != (Path(tmp) / rel).read_bytes()
        )
    info = {
        "canonical": not differ,
        "machine": platform.machine(),
        "system": platform.system(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "export_files": len(written),
        "export_files_that_differ": differ,
    }
    OUT.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    state = "canonical" if info["canonical"] else f"not canonical ({len(differ)} files differ)"
    print(f"wrote {OUT} ({platform.machine()}, NumPy {np.__version__}: {state})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
