"""네이버 DataLab 검색어 트렌드 API 설정"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── 경로 ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Why-pi/
CONFIG_DIR = PROJECT_ROOT / "config"
FINAL_DIR = PROJECT_ROOT / "02_processed_data" / "csv" / "final"

MODULE_DIR = Path(__file__).resolve().parent  # 04_search_trend/
CACHE_DIR = MODULE_DIR / "cache"
OUTPUT_DIR = MODULE_DIR / "output"

for d in [CACHE_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── 환경변수 ────────────────────────────────────────────
load_dotenv(CONFIG_DIR / ".env")

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID", "")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET", "")

# 복수 키 리스트 (키 로테이션용)
# NAVER_CLIENT_ID + NAVER_CLIENT_SECRET → 키 1
# NAVER_CLIENT_ID2 + NAVER_CLIENT_SECRET2 → 키 2, ...
NAVER_API_KEYS: list[dict[str, str]] = []
_key_idx = 1
while True:
    _suffix = "" if _key_idx == 1 else str(_key_idx)
    _cid = os.getenv(f"NAVER_CLIENT_ID{_suffix}", "")
    _csec = os.getenv(f"NAVER_CLIENT_SECRET{_suffix}", "")
    if not _cid or not _csec:
        break
    NAVER_API_KEYS.append({"client_id": _cid, "client_secret": _csec})
    _key_idx += 1

# ── API ─────────────────────────────────────────────────
NAVER_DATALAB_URL = "https://openapi.naver.com/v1/datalab/search"

# ── 성별 코드 ───────────────────────────────────────────
GENDER_MAP = {
    "남성": "m",
    "여성": "f",
}

# ── 연령대 코드 (네이버 DataLab 기준) ───────────────────
# "1": 0~12, "2": 13~18, "3": 19~24, "4": 25~29,
# "5": 30~34, "6": 35~39, "7": 40~44, "8": 45~49,
# "9": 50~54, "10": 55~59, "11": 60+
AGE_MAP = {
    "10대(13-18)": ["2"],
    "20대(19-24)": ["3"],
    "20대(25-29)": ["4"],
    "30대(30-34)": ["5"],
    "30대(35-39)": ["6"],
    "40대(40-44)": ["7"],
    "40대(45-49)": ["8"],
    "50대(50-54)": ["9"],
    "50대(55-59)": ["10"],
    "60대(60+)": ["11"],
}

# 분석용 그룹 (10대 단위)
AGE_GROUPS = {
    "10대": ["2"],
    "20대": ["3", "4"],
    "30대": ["5", "6"],
    "40대": ["7", "8"],
    "50대": ["9", "10"],
    "60대": ["11"],
}

# ── 기기 코드 ───────────────────────────────────────────
DEVICE_MAP = {
    "PC": "pc",
    "모바일": "mo",
}

# ── 기본값 ──────────────────────────────────────────────
DEFAULT_START_DATE = "2024-01-01"
DEFAULT_END_DATE = "2026-01-31"
DEFAULT_TIME_UNIT = "month"
DEFAULT_TOP_N = 50
