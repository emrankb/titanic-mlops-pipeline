"""Serve a trusted local Titanic model over FastAPI."""
from __future__ import annotations
import argparse
from pathlib import Path
import uvicorn
from titanic_mlops.api import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/survival-model/model.joblib"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    app = create_app(model_path=args.model)
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
