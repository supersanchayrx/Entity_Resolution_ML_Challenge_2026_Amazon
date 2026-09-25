"""Entry point (also used by SageMaker Processing): python run.py <step> [options]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
