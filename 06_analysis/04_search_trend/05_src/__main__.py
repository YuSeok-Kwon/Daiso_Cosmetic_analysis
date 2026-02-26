"""직접 실행 지원: python -c "..." 또는 패키지 import 후 호출"""

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))

from importlib import import_module

_run = import_module("04_search_trend.run_search_trend")
_run.main()
