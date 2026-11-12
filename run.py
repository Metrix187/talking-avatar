"""Start the talking-avatar server from any working directory:

    .venv\Scripts\python run.py                       ->  http://127.0.0.1:8770
    .venv\Scripts\python run.py --compile --steps 4   (slower start, faster chunks)
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser(description="Talking avatar server")
ap.add_argument("--compile", action="store_true",
                help="torch.compile the models: a few minutes longer to start, ~15%% faster chunks")
ap.add_argument("--steps", type=int, help="denoise steps per chunk (default 2)")
args = ap.parse_args()
# flags win over .env, which wins over nothing: the app reads these at import
if args.compile:
    os.environ["AVATAR_COMPILE"] = "1"
if args.steps:
    os.environ["AVATAR_STEPS"] = str(args.steps)

# API keys can live in talking-avatar/.env (MODEL_API_KEY=... or ANTHROPIC_API_KEY=...)
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from app.server import main  # noqa: E402

main()
