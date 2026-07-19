#!/usr/bin/env python3
"""Build a deployable Pages tree from the calibrated reference bundles."""

from __future__ import annotations

import argparse
import json
import shutil

from g1cal.paths import project_root, resolve_inside_root
from g1cal.rendering import render_clip


def _publish_clip(clip: str, *, work_root: str, site_root: str) -> dict:
    report = render_clip(clip, output_root=work_root)
    source = resolve_inside_root(report["artifacts"]["meshcat_html"]["path"])
    destination_root = resolve_inside_root(
        f"{site_root}/media", must_exist=False
    )
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / f"{clip}_calibrated.html"
    shutil.copy2(source, destination)
    return {
        "clip": clip,
        "source_result": report["source_result"],
        "se3_log_loss": report["se3_log_loss"],
        "interactive_html": str(destination.relative_to(project_root())),
        "bytes": destination.stat().st_size,
        "default_playback_rate": 0.5,
        "available_playback_rates": [0.5, 1.0],
        "normal_speed_url": f"{destination.name}?speed=1",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", action="append", choices=("run1", "run2"))
    parser.add_argument("--work-root", default="out/pages_build")
    parser.add_argument("--site-root", default="out/pages_site")
    args = parser.parse_args()
    site_root = resolve_inside_root(args.site_root, must_exist=False)
    site_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolve_inside_root("docs/index.html"), site_root / "index.html")
    (site_root / ".nojekyll").touch()
    reports = [
        _publish_clip(
            clip,
            work_root=args.work_root,
            site_root=args.site_root,
        )
        for clip in (args.clip or ["run1", "run2"])
    ]
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
