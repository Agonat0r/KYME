#!/usr/bin/env python3
"""
Quick launcher for the KYMA server.

Usage
─────
  python run.py               # real hardware (Cyton + Arduino)
  python run.py --mock        # simulated hardware, no board/Arduino needed
  python run.py --mock --port 9000

Flags
─────
  --mock          Use SimulatedCytonStream + MockArduinoBridge
  --cyton  COMx   Override Cyton serial port   (default: COM6)
  --arduino COMx  Override Arduino serial port (default: COM4)
  --port   N      HTTP port                    (default: 8000)
  --host   H      Bind host                    (default: 0.0.0.0)
"""
import argparse
import os
import sys

# Ensure server/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))


def main():
    parser = argparse.ArgumentParser(description="KYMA — Biosignal Control Platform")
    parser.add_argument("--mock",    action="store_true", help="Use simulated hardware")
    parser.add_argument("--cyton",   default=None,        metavar="PORT")
    parser.add_argument("--arduino", default=None,        metavar="PORT")
    parser.add_argument("--port",    type=int, default=8000)
    parser.add_argument("--host",    default="0.0.0.0")
    args = parser.parse_args()

    # Pass settings via environment variables picked up by config.py / main.py
    if args.mock:
        os.environ["EMG_MOCK"] = "1"
    if args.cyton:
        os.environ["CYTON_PORT"] = args.cyton
    if args.arduino:
        os.environ["ARDUINO_PORT"] = args.arduino
    os.environ["PORT"] = str(args.port)

    import uvicorn
    uvicorn.run(
        "main:app",
        app_dir="server",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
