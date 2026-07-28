#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scaling.dashboard_data import build_run_payload, list_runs


class DashboardHandler(BaseHTTPRequestHandler):
    dashboard_dir = PROJECT_ROOT / "dashboard"
    runs_dir = PROJECT_ROOT / "runs"

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, requested: str) -> None:
        relative = requested.lstrip("/") or "index.html"
        candidate = (self.dashboard_dir / relative).resolve()
        try:
            candidate.relative_to(self.dashboard_dir.resolve())
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = candidate.read_bytes()
        content_type, _ = mimetypes.guess_type(candidate.name)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        if path == "/api/runs":
            self._send_json({"runs": list_runs(self.runs_dir)})
            return
        if path.startswith("/api/run/"):
            run_name = unquote(path[len("/api/run/") :])
            if not run_name or "/" in run_name or "\\" in run_name:
                self._send_json({"error": "invalid run name"}, HTTPStatus.BAD_REQUEST)
                return
            run_dir = self.runs_dir / run_name
            if not run_dir.is_dir():
                self._send_json({"error": "run not found"}, HTTPStatus.NOT_FOUND)
                return
            self._send_json(build_run_payload(run_dir))
            return
        self._serve_file(path)

    def log_message(self, format: str, *args) -> None:
        if self.server.verbose:  # type: ignore[attr-defined]
            super().log_message(format, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the live training dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--runs-dir",
        help="Runs directory to watch (defaults to scaling_runs/runs)",
    )
    args = parser.parse_args()

    if args.runs_dir:
        DashboardHandler.runs_dir = Path(args.runs_dir).expanduser().resolve()

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    server.verbose = args.verbose  # type: ignore[attr-defined]
    print(f"Dashboard: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
