"""Render the deterministic README terminal demo GIF."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1200, 675
BACKGROUND = "#08111f"
PANEL = "#101c2e"
TEXT = "#dbeafe"
MUTED = "#7f93ad"
GREEN = "#4ade80"
BLUE = "#60a5fa"
ORANGE = "#fb923c"

SCENES = (
    (
        "1 / 3   LEAKAGE-SAFE RESEARCH",
        "$ quantbot robustness --start 2010-01-01 --end 2024-12-31",
        (
            "Nested ML + momentum   Sharpe 0.717   ROC AUC 0.501",
            "SPY buy and hold      Sharpe 0.789",
            "Conclusion: no deployable alpha - reported honestly.",
        ),
    ),
    (
        "2 / 3   DURABLE PAPER MONITORING",
        "$ quantbot paper-report --database state/paper_trading.sqlite3",
        (
            '{  "status": "healthy",',
            '   "decision_count": 24,  "duplicate_signal_dates": 0  }',
            "SQLite + HTML + CSV + JSON audit history",
        ),
    ),
    (
        "3 / 3   CLONE, TEST, RUN",
        "$ docker run --rm quantbot:latest --help",
        (
            "backtest | walk-forward | robustness | trade | paper-report",
            "76 tests   Python 3.11-3.13   80% enforced coverage",
            "Paper execution is dry-run by default.",
        ),
    ),
)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf",
        "C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf",
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
        ),
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _frame(scene_index: int, reveal: int) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font = _font(30, bold=True)
    code_font = _font(25)
    small_font = _font(19, bold=True)

    draw.text((70, 42), "quantbot", font=title_font, fill=TEXT)
    draw.text((225, 50), "systematic research -> paper execution", font=small_font, fill=MUTED)
    draw.rounded_rectangle((65, 105, 1135, 580), radius=18, fill=PANEL, outline="#263a54", width=2)
    for x, color in ((94, "#f87171"), (122, "#fbbf24"), (150, GREEN)):
        draw.ellipse((x, 130, x + 14, 144), fill=color)

    label, command, output = SCENES[scene_index]
    draw.text((92, 178), label, font=small_font, fill=BLUE)
    lines = (command, *output)
    colors = (TEXT, GREEN, GREEN, ORANGE)
    for line_number, line in enumerate(lines[:reveal]):
        draw.text((92, 245 + line_number * 62), line, font=code_font, fill=colors[line_number])
    if reveal <= len(lines):
        y = 245 + max(0, reveal - 1) * 62
        prefix = lines[max(0, reveal - 1)] if reveal else ""
        x = 92 + draw.textlength(prefix, font=code_font)
        draw.rectangle((x + 3, y + 3, x + 16, y + 31), fill=TEXT)

    for step in range(len(SCENES)):
        color = BLUE if step == scene_index else "#30445f"
        draw.rounded_rectangle((495 + step * 72, 620, 553 + step * 72, 628), radius=4, fill=color)
    return image


def main() -> None:
    frames: list[Image.Image] = []
    durations: list[int] = []
    for scene_index, (_, _, output) in enumerate(SCENES):
        line_count = 1 + len(output)
        for reveal in range(1, line_count + 1):
            frames.append(_frame(scene_index, reveal))
            durations.append(650 if reveal < line_count else 2_200)
    output_path = Path(__file__).resolve().parent.parent / "docs" / "demo.gif"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"Demo written to {output_path}")


if __name__ == "__main__":
    main()
