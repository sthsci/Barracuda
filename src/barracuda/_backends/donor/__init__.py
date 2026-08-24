"""Source-checkout path for the packaged donor-aware backend."""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[4] / "section_2" / "src")]
