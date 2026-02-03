import os
import subprocess
import sys


def main() -> int:
    port = os.environ.get("PORT", "8501")
    cmd = [
        "streamlit",
        "run",
        "chatbot/streamlit_chatbot.py",
        "--server.port",
        port,
        "--server.address",
        "0.0.0.0",
        "--server.headless",
        "true",
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
