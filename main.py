"""OpsMind — entry point.

Local:  python main.py
Replit: the Run button uses this file (see .replit)
Prod:   gunicorn "main:app" --bind 0.0.0.0:8000 --workers 2
"""

import os

from dotenv import load_dotenv

load_dotenv()

from app import create_app  # noqa: E402  (after load_dotenv so config sees the vars)

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
