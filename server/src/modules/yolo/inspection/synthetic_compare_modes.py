from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


MODES = ("fallback", "yolo", "auto")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run synthetic_context_test in three comparable pose modes: "
            "fallback-only, yolo-only/prefer, and auto arbiter."
        )
    )
    parser.add_argument("--out", required=True, help="Base output directory")
    parser.add_argument("--profile", default="nightmare")
    parser.add_argument("--cases", type=int, default=240)
    parser.add_argument("--projection-pipeline", default="v2")
    parser.add_argument("--synthetic-yolo-anchor-noise-px", type=float, default=1.5)
    parser.add_argument("--synthetic-yolo-anchor-dropout", type=float, default=0.06)
    parser.add_argument("--synthetic-yolo-anchor-min-anchors", type=int, default=2)
    parser.add_argument("--synthetic-yolo-anchor-ransac-px", type=float, default=6.0)
    parser.add_argument(
        "--stress",
        action="store_true",
        help="Use harsher default anchor noise/dropout unless values are explicitly passed.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra argument passed through to synthetic_context_test; repeat as needed.",
    )
    args = parser.parse_args(argv)

    base_out = Path(args.out)
    base_out.mkdir(parents=True, exist_ok=True)

    noise = args.synthetic_yolo_anchor_noise_px
    dropout = args.synthetic_yolo_anchor_dropout
    ransac = args.synthetic_yolo_anchor_ransac_px
    if args.stress:
        noise = 3.0 if noise == 1.5 else noise
        dropout = 0.15 if dropout == 0.06 else dropout
        ransac = 8.0 if ransac == 6.0 else ransac

    results: dict[str, dict[str, Any]] = {}
    failures: dict[str, int] = {}

    for mode in MODES:
        mode_out = base_out / mode
        cmd = [
            sys.executable,
            "-m",
            "modules.yolo.inspection.synthetic_context_test",
            "--projection-pipeline",
            str(args.projection_pipeline),
            "--profile",
            str(args.profile),
            "--cases",
            str(args.cases),
            "--out",
            str(mode_out),
        ]

        env = os.environ.copy()
        if mode == "fallback":
            env["INSPECTION_YOLO_ANCHOR_POSE_MODE"] = "off"
        else:
            env["INSPECTION_YOLO_ANCHOR_POSE_MODE"] = "prefer" if mode == "yolo" else "auto"
            cmd.extend(
                [
                    "--synthetic-yolo-anchor-pose",
                    "--synthetic-yolo-anchor-noise-px",
                    str(noise),
                    "--synthetic-yolo-anchor-dropout",
                    str(dropout),
                    "--synthetic-yolo-anchor-min-anchors",
                    str(args.synthetic_yolo_anchor_min_anchors),
                    "--synthetic-yolo-anchor-ransac-px",
                    str(ransac),
                ]
            )

        cmd.extend(args.extra_arg)
        print("\n=== running", mode, "===")
        print("$", " ".join(cmd))
        proc = subprocess.run(cmd, env=env, text=True)
        failures[mode] = int(proc.returncode)
        summary_path = mode_out / "summary.json"
        if summary_path.exists():
            results[mode] = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            results[mode] = {"error": f"summary not found; exit={proc.returncode}"}

    compare = {
        "modes": results,
        "exit_codes": failures,
        "winner_by_object_accuracy": _winner(results, "object_accuracy_rate"),
        "winner_by_safety": _winner(results, "object_safety_rate"),
    }
    (base_out / "compare_summary.json").write_text(
        json.dumps(compare, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (base_out / "compare_report.html").write_text(
        _render_compare_html(compare),
        encoding="utf-8",
    )

    print("\n=== compare summary ===")
    for mode in MODES:
        summary = results.get(mode, {})
        print(
            mode,
            "case=", _fmt(summary.get("case_pass_rate")),
            "object=", _fmt(summary.get("object_accuracy_rate")),
            "safety=", _fmt(summary.get("object_safety_rate")),
            "danger=", summary.get("object_dangerous_projection_count", summary.get("dangerous_projection_count", "—")),
            "hidden=", summary.get("object_unsafe_hidden_count", summary.get("unsafe_hidden_count", "—")),
        )
    print("\nwrote:", base_out / "compare_summary.json")
    print("wrote:", base_out / "compare_report.html")
    return 0 if all(code == 0 for code in failures.values()) else 1


def _winner(results: dict[str, dict[str, Any]], metric: str) -> str | None:
    best_mode: str | None = None
    best_value: float | None = None
    for mode, summary in results.items():
        try:
            value = float(summary.get(metric))
        except (TypeError, ValueError):
            continue
        if best_value is None or value > best_value:
            best_mode = mode
            best_value = value
    return best_mode


def _fmt(value: object) -> str:
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "—"


def _render_compare_html(compare: dict[str, Any]) -> str:
    rows = []
    for mode in MODES:
        summary = compare.get("modes", {}).get(mode, {})
        rows.append(
            "<tr>"
            f"<td>{mode}</td>"
            f"<td>{_fmt(summary.get('case_pass_rate'))}</td>"
            f"<td>{_fmt(summary.get('object_accuracy_rate'))}</td>"
            f"<td>{_fmt(summary.get('object_safety_rate'))}</td>"
            f"<td>{summary.get('object_dangerous_projection_count', summary.get('dangerous_projection_count', '—'))}</td>"
            f"<td>{summary.get('object_unsafe_hidden_count', summary.get('unsafe_hidden_count', '—'))}</td>"
            f"<td><a href='{mode}/report.html'>report</a></td>"
            "</tr>"
        )
    return """<!doctype html>
<html lang=\"ru\">
<head>
<meta charset=\"utf-8\">
<title>Synthetic pose mode comparison</title>
<style>
body { font-family: system-ui, sans-serif; margin: 24px; background: #111; color: #eee; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #333; padding: 8px 10px; text-align: left; }
th { background: #242424; }
a { color: #8ecbff; }
</style>
</head>
<body>
<h1>Synthetic pose mode comparison</h1>
<p>fallback = LightGlue/feature-only, yolo = YOLO-anchor prefer, auto = arbiter between YOLO-anchor and LightGlue fallback.</p>
<table>
<tr><th>mode</th><th>case pass</th><th>object accuracy</th><th>object safety</th><th>danger</th><th>hidden</th><th>report</th></tr>
""" + "\n".join(rows) + """
</table>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
