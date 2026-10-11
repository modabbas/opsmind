#!/usr/bin/env bash
# Install once, then run. Replit re-runs this on every Run click; pip is a
# no-op when the packages are already present.
set -e
pip install -q -r requirements.txt
exec python main.py
