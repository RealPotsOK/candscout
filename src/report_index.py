#!/usr/bin/env python3
"""Generate a small report index page for the local reports server."""

from __future__ import annotations

import argparse
import html
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate data/reports index.html")
    parser.add_argument("--output", default="data/reports/index.html", help="Index HTML output path")
    parser.add_argument("--title", default="cryptopred reports", help="Index page title")
    parser.add_argument("--model-url", required=True, help="Model visualization URL path")
    parser.add_argument("--sim-url", required=True, help="Simulation visualization URL path")
    parser.add_argument("--lr-model-url", default="", help="Optional LR model visualization URL path")
    parser.add_argument("--lr-sim-url", default="", help="Optional LR simulation visualization URL path")
    return parser.parse_args()


def link(label: str, url: str, description: str) -> str:
    if not url:
        return ""
    return (
        "<a class=\"card\" href=\""
        + html.escape(url, quote=True)
        + "\"><strong>"
        + html.escape(label)
        + "</strong><span>"
        + html.escape(description)
        + "</span></a>"
    )


def build_html(args: argparse.Namespace) -> str:
    title = html.escape(args.title)
    cards = [
        link("Model Visualization", args.model_url, "Price, predictions, outcomes, equity vs buy-and-hold"),
        link("Simulation Visualization", args.sim_url, "Bank simulation trades and active invested dollars"),
        link("LR Model Visualization", args.lr_model_url, "Older logistic-regression visualization"),
        link("LR Simulation Visualization", args.lr_sim_url, "Older logistic-regression bank simulation"),
    ]
    cards_html = "\n".join(card for card in cards if card)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{
      margin: 0;
      min-height: 100vh;
      background: radial-gradient(circle at top left, #fff8df 0, #f4f0e7 38%, #e7ece8 100%);
      color: #17201a;
      font-family: Georgia, "Times New Roman", serif;
    }}
    header {{
      padding: 28px 32px 12px;
      border-bottom: 1px solid #d6d0bf;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
    }}
    p {{
      margin: 0;
      color: #5d665d;
      font-family: "Courier New", monospace;
      font-size: 13px;
    }}
    main {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
      padding: 20px 24px;
    }}
    .card {{
      display: block;
      padding: 18px;
      border: 1px solid #c8c1ad;
      background: rgba(255, 253, 246, 0.86);
      color: #17201a;
      text-decoration: none;
      box-shadow: 0 10px 28px rgba(41, 36, 18, 0.08);
    }}
    .card:hover {{
      border-color: #17201a;
      transform: translateY(-1px);
    }}
    .card strong {{
      display: block;
      margin-bottom: 9px;
      font-size: 18px;
    }}
    .card span {{
      display: block;
      color: #5d665d;
      font-family: "Courier New", monospace;
      font-size: 13px;
      line-height: 1.35;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <p>Local report navigation for the current generated experiment outputs.</p>
  </header>
  <main>
    {cards_html}
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_html(args), encoding="utf-8")
    print(f"Saved report index to {output}")


if __name__ == "__main__":
    main()
