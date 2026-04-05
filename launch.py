#!/usr/bin/env python3
"""
KYMA Desktop Launcher

Opens KYMA in a native desktop window (no browser needed).
Falls back to browser if pywebview is not installed.

Usage:
    python launch.py              # default (COM8)
    python launch.py --cyton COM3 # specify Cyton port
    python launch.py --browser    # force browser instead of native window
"""
import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))


def start_server(host, port):
    """Start the FastAPI server in a background thread."""
    import uvicorn
    uvicorn.run(
        "main:app",
        app_dir="server",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


def wait_for_server(host, port, timeout=30):
    """Block until the server is responding."""
    import urllib.request
    url = f"http://{host}:{port}/api/status"
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(0.3)
    return False


def main():
    parser = argparse.ArgumentParser(description="KYMA Desktop Launcher")
    parser.add_argument("--cyton", default=None, metavar="PORT", help="Cyton COM port (default: COM8)")
    parser.add_argument("--arduino", default=None, metavar="PORT", help="Arduino COM port")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--browser", action="store_true", help="Open in browser instead of native window")
    args = parser.parse_args()

    if args.cyton:
        os.environ["CYTON_PORT"] = args.cyton
    if args.arduino:
        os.environ["ARDUINO_PORT"] = args.arduino
    os.environ["PORT"] = str(args.port)

    url = f"http://{args.host}:{args.port}"

    # Start server in background thread
    server_thread = threading.Thread(
        target=start_server, args=(args.host, args.port), daemon=True
    )
    server_thread.start()

    print(f"Starting KYMA server on {url} ...")
    if not wait_for_server(args.host, args.port):
        print("ERROR: Server failed to start within 30 seconds.")
        sys.exit(1)

    print("Server ready.")

    # Try native window, fall back to browser
    if not args.browser:
        try:
            import webview
            print("Opening native desktop window...")
            webview.create_window(
                "KYMA",
                url,
                width=1400,
                height=900,
                min_size=(1000, 600),
                resizable=True,
                text_select=True,
            )
            webview.start()
            return  # Window closed, exit
        except ImportError:
            print("pywebview not installed, falling back to browser.")
            print("  Install it with: pip install pywebview")
        except Exception as e:
            print(f"Desktop window failed: {e}, falling back to browser.")

    # Browser fallback
    import webbrowser
    webbrowser.open(url)
    print(f"Opened {url} in your browser.")
    print("Press Ctrl+C to stop the server.")
    try:
        server_thread.join()
    except KeyboardInterrupt:
        print("\nStopping KYMA.")


if __name__ == "__main__":
    main()
