"""Create a deterministic zero-dependency SVG badge from coverage.py JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def color_for(percent: float) -> str:
    if percent >= 90:
        return "#4c1"
    if percent >= 80:
        return "#97ca00"
    if percent >= 70:
        return "#a4a61d"
    if percent >= 60:
        return "#dfb317"
    return "#e05d44"


def render_badge(percent: float) -> str:
    value = f"{percent:.0f}%"
    color = color_for(percent)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="106" height="20" role="img" aria-label="coverage: {value}">
  <title>coverage: {value}</title>
  <linearGradient id="s" x2="0" y2="100%"><stop offset="0" stop-color="#bbb" stop-opacity=".1"/><stop offset="1" stop-opacity=".1"/></linearGradient>
  <clipPath id="r"><rect width="106" height="20" rx="3" fill="#fff"/></clipPath>
  <g clip-path="url(#r)"><path fill="#555" d="M0 0h67v20H0z"/><path fill="{color}" d="M67 0h39v20H67z"/><path fill="url(#s)" d="M0 0h106v20H0z"/></g>
  <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="11">
    <text x="34.5" y="15" fill="#010101" fill-opacity=".3">coverage</text><text x="34.5" y="14">coverage</text>
    <text x="85.5" y="15" fill="#010101" fill-opacity=".3">{value}</text><text x="85.5" y="14">{value}</text>
  </g>
</svg>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="coverage.json")
    parser.add_argument("--output", default="docs/coverage.svg")
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    percent = float(payload["totals"]["percent_covered"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_badge(percent), encoding="utf-8")


if __name__ == "__main__":
    main()
