"""Build PDF from TaxGuard methodology Markdown (KaTeX + Playwright)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MD = ROOT / "docs" / "TaxGuard_Corporate_Tax_Risk_Scoring_Methodology.md"
DEFAULT_PDF = ROOT / "docs" / "TaxGuard_Corporate_Tax_Risk_Scoring_Methodology.pdf"
DEFAULT_HTML = ROOT / "docs" / "_methodology_build.html"

KATEX_HEAD = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
"""

KATEX_INIT = """
<script>
document.addEventListener("DOMContentLoaded", function() {
  renderMathInElement(document.body, {
    delimiters: [
      {left: "$$", right: "$$", display: true},
      {left: "\\\\[", right: "\\\\]", display: true},
      {left: "$", right: "$", display: false},
      {left: "\\\\(", right: "\\\\)", display: false}
    ],
    throwOnError: false
  });
});
</script>
"""

CSS = """
<style>
  @page { size: A4; margin: 22mm 18mm; }
  body {
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.45;
    color: #1a1a1a;
    max-width: 170mm;
    margin: 0 auto;
  }
  h1 { font-size: 18pt; border-bottom: 2px solid #1e3a5f; padding-bottom: 6px; }
  h2 { font-size: 14pt; color: #1e3a5f; margin-top: 1.2em; }
  h3 { font-size: 12pt; color: #2c5282; }
  h4 { font-size: 11pt; }
  table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 10pt; }
  th, td { border: 1px solid #cbd5e0; padding: 6px 8px; text-align: left; }
  th { background: #edf2f7; font-weight: 600; }
  pre, code { font-family: Consolas, "Courier New", monospace; font-size: 9pt; }
  pre {
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    padding: 10px;
    overflow-x: auto;
    white-space: pre-wrap;
  }
  blockquote {
    border-left: 4px solid #2b6cb0;
    margin: 12px 0;
    padding: 8px 14px;
    background: #ebf8ff;
    font-size: 10.5pt;
  }
  .katex-display { margin: 14px 0; overflow-x: auto; }
</style>
"""


def md_to_html(md_path: Path, html_path: Path) -> None:
    import pypandoc

    body = pypandoc.convert_file(
        str(md_path),
        "html",
        format="markdown+tex_math_dollars+raw_tex",
        extra_args=[
            "--standalone",
            "--toc",
            "--toc-depth=3",
            "--number-sections",
            "--mathjax",
        ],
    )
    # Inject KaTeX into <head>
    if "<head>" in body:
        body = body.replace("<head>", f"<head>{KATEX_HEAD}{CSS}", 1)
    else:
        body = f"<html><head>{KATEX_HEAD}{CSS}</head><body>{body}</body></html>"
    if "</body>" in body:
        body = body.replace("</body>", f"{KATEX_INIT}</body>", 1)
    html_path.write_text(body, encoding="utf-8")


def html_to_pdf(html_path: Path, pdf_path: Path) -> None:
    from playwright.sync_api import sync_playwright

    uri = html_path.resolve().as_uri()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(uri, wait_until="networkidle")
        page.wait_for_timeout(2500)
        page.pdf(
            path=str(pdf_path),
            format="A4",
            print_background=True,
            margin={"top": "18mm", "bottom": "18mm", "left": "16mm", "right": "16mm"},
        )
        browser.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build TaxGuard methodology PDF")
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--html-only", action="store_true")
    args = parser.parse_args()

    if not args.md.exists():
        print(f"Missing: {args.md}", file=sys.stderr)
        return 1

    print(f"Converting {args.md.name} -> HTML ...")
    md_to_html(args.md, DEFAULT_HTML)
    print(f"Wrote {DEFAULT_HTML}")

    if args.html_only:
        return 0

    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except ImportError:
        print(
            "Install Playwright for PDF output:\n"
            "  pip install playwright\n"
            "  playwright install chromium",
            file=sys.stderr,
        )
        return 1

    print(f"Rendering PDF -> {args.pdf} ...")
    html_to_pdf(DEFAULT_HTML, args.pdf)
    print(f"Done: {args.pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
