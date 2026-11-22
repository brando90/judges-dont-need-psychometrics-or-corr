#!/usr/bin/env python3
"""Package judge-objectives-status.html and its referenced experiment PNGs into a self-contained zip bundle."""
import argparse
import os
import re
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
HTML_SRC = HERE / "judge-objectives-status.html"

HEADER = (
    "<!doctype html><html><head><meta charset=utf8>"
    "<meta name=viewport content=\"width=device-width,initial-scale=1,viewport-fit=cover\">"
    "<style>:root{color-scheme:light;box-sizing:border-box;"
    "padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}"
    "html{scroll-padding-top:env(safe-area-inset-top,0px)}"
    "body{margin:0;padding:0;font:14px -apple-system,BlinkMacSystemFont,sans-serif;background:#fff;color:#000}"
    "img{max-width:100%}[hidden]:not([hidden=until-found i]){display:none!important}</style></head><body>\n"
)
TRAILER = "\n</body></html>\n"

IMAGE_PATTERNS = [
    "experiments/06_*/*.png",
    "experiments/07_*/*.png",
    "experiments/08_*/*.png",
    "experiments/09_*/figures/*.png",
    "experiments/10_*/figures/*.png",
    "experiments/11_*/*.png",
    "experiments/24_trained_open_weight_judge_helpsteer2_ce_vs_pearson_start_21Nov2026/figures/*.png",
    "experiments/25_helpsteer2_lora_scale_and_second_family_ce_vs_pearson_start_21Nov2026/figures/*.png",
    "experiments/archive/13_*/*.png",
]


def discover_images() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for pat in IMAGE_PATTERNS:
        for path in sorted(REPO.glob(pat)):
            found[path.name] = path
    return found


def build_bundle(out_zip: Path) -> list[str]:
    body = HTML_SRC.read_text()
    referenced = list(dict.fromkeys(re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', body)))
    available = discover_images()
    missing = [name for name in referenced if name not in available]
    if missing:
        raise FileNotFoundError(f"Missing referenced image(s): {missing}")

    full_html = HEADER + body + TRAILER
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("index.html", full_html)
        for name in referenced:
            zf.write(available[name], arcname=name)
    return referenced


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=HERE / "_dist" / "Correlation_vs_Loss_Judges_bundle.zip",
        help="Output zip path (default outside tracked git or explicit artifact path).",
    )
    args = ap.parse_args()
    refs = build_bundle(args.out)
    print(f"Wrote {args.out} ({os.path.getsize(args.out)} bytes) with index.html + {len(refs)} images: {refs}")


if __name__ == "__main__":
    main()
