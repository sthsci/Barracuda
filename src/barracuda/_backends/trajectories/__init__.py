"""Source-checkout path for the packaged trajectory backend."""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[4] / "section_3" / "src")]
