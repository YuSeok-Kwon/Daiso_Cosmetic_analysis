"""MFDS API 설정"""

import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "02_processed_data"
CACHE_DIR = PROJECT_ROOT / "05_src" / "03_mfds_api" / "cache"

load_dotenv(CONFIG_DIR / ".env")

MFDS_API_KEY = os.getenv("MFDS_API_KEY", "")

# API 엔드포인트
ENDPOINTS = {
    "functional": "http://apis.data.go.kr/1471000/FtnltCosmRptPrdlstInfoService/getRptPrdlstInq",
    "ingredient": "https://apis.data.go.kr/1471000/CsmtcsIngdCpntInfoService01/getCsmtcsIngdCpntInfoService01",
    "restricted": "https://apis.data.go.kr/1471000/CsmtcsUseRstrcInfoService/getCsmtcsUseRstrcInfoService",
}

# 기능성화장품 효능 매핑 (화장품법 시행규칙 제10조)
EFFECT_FLAG_MAP = {
    "1호": ("is_whitening", "미백"),
    "2호": ("is_anti_wrinkle", "주름개선"),
    "3호": ("is_sunscreen", "자외선차단"),
    "4호": ("is_hair_color", "모발 색상"),
    "5호": ("is_hair_removal", "체모 제거"),
    "6호": ("is_acne", "여드름 완화"),
    "7호": ("is_atopic", "아토피 보습"),
    "8호": ("is_stretch_marks", "튼살 완화"),
    "9호": ("is_hair_loss", "탈모 완화"),
    "10호": ("is_eye_wrinkle", "눈가 개선"),
    "11호": ("is_tanning", "선탠"),
}
