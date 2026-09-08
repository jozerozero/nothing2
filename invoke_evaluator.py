"""Importable entry point so spawned evaluator workers inherit the override."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'source_loop/src'))
from loop_runtime import install
install()
sys.path.insert(0, str(HERE / 'evaluator'))
import talent_eval_online

if __name__ == '__main__':
    talent_eval_online.main()
