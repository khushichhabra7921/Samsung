"""Command-line entry points for local analysis and the Studio."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from egovision.ingest import VideoError

from .contracts import ROOT, write_json
from .pipeline import Options, analyze
from .recognition import ModelSetupError
from .render import RenderError, render

log = logging.getLogger(__name__)


def _options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=("preview", "semantic"), default="preview")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights", help="Directory containing local model assets")
    parser.add_argument("--vocabulary", type=Path, help="Track 04 labels as verbs and nouns arrays")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")


def main() -> int:
    parser = argparse.ArgumentParser(description="HandScribe XR offline video summarization")
    commands = parser.add_subparsers(dest="command", required=True)
    summarize = commands.add_parser("summarize", help="Analyze one first-person MP4")
    summarize.add_argument("video", type=Path)
    summarize.add_argument("output", type=Path, help="Development JSON report")
    summarize.add_argument("--highlight", type=Path, help="Optional silent highlight MP4")
    summarize.add_argument("--trace", type=Path, help="Optional motion and landmark trace JSON")
    summarize.add_argument("--allow-short", action="store_true", help="Allow a sub-four-second highlight in preview")
    _options(summarize)
    serve = commands.add_parser("serve", help="Start the Studio and local API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    _options(serve)
    args = parser.parse_args()
    options = Options(mode=args.mode, weights=args.weights, vocabulary=args.vocabulary, device=args.device)

    try:
        if args.command == "serve":
            import uvicorn

            from .studio import create_app

            options = replace(options, allow_short=True)
            options.models()
            uvicorn.run(create_app(options), host=args.host, port=args.port, workers=1)
            return 0

        outputs = [item.resolve() for item in (args.output, args.trace, args.highlight) if item is not None]
        if args.video.resolve() in outputs or len(set(outputs)) != len(outputs):
            parser.error("Input, report, trace and highlight destinations must be different files.")
        analysis = analyze(args.video, args.video.stem, replace(options, allow_short=args.allow_short))
        report = analysis.report
        write_json(args.output, report.model_dump(mode="json"))
        if args.trace is not None:
            write_json(args.trace, {"motion": analysis.motion, "hands": analysis.hands}, indent=None)
        if args.highlight is not None and report.highlights:
            render(args.video, args.highlight, report)
        print(
            f"{report.video_id} | {report.status} | {len(report.highlights)} highlights | "
            f"{report.highlight_duration_ms / 1000:.3f} / {report.budget_ms / 1000:.3f}s budget"
        )
        return 0
    except (VideoError, ModelSetupError, RenderError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        if isinstance(exc, ModelSetupError) and exc.__cause__:
            log.error("%s", exc.__cause__)
        return 2


if __name__ == "__main__":
    sys.exit(main())
