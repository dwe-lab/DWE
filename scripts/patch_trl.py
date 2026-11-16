"""Apply the two-line loss hook to the pinned TRL release, with source verification."""

import hashlib
import importlib.metadata
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
assert importlib.metadata.version("trl") == "1.13.0", "Install the locked TRL version"
dist = importlib.metadata.distribution("trl")
source = Path(dist.locate_file("trl/trainer/distillation_trainer.py"))
before, after = (root / "patches/trl/source.sha256").read_text().splitlines()
actual = hashlib.sha256(source.read_bytes()).hexdigest()
if actual != after:
    if actual != before:
        raise SystemExit("TRL source differs from the reviewed release; refusing to patch")
    subprocess.run(
        ["patch", "--batch", "--forward", str(source), str(root / "patches/trl/loss-hook.patch")],
        check=True,
    )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == after
print("TRL 1.13.0 loss hook verified")
