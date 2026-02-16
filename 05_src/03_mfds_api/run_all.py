"""MFDS API 전체 실행 스크립트

사용법:
    # 프로젝트 루트에서 실행
    cd Why-pi/

    # 전체 실행
    python 05_src/03_mfds_api/run_all.py

    # 개별 실행
    python 05_src/03_mfds_api/run_all.py --target functional
    python 05_src/03_mfds_api/run_all.py --target ingredient
    python 05_src/03_mfds_api/run_all.py --target restricted

    # 단건 테스트
    python 05_src/03_mfds_api/run_all.py --test
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트 및 모듈 경로 설정
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from functional_cosmetics import FunctionalCosmeticsAPI
from ingredient_info import IngredientInfoAPI
from restricted_ingredients import RestrictedIngredientsAPI


def run_functional():
    """기능성화장품 보고품목 매칭"""
    print("=" * 60)
    print("[1/3] 기능성화장품 보고품목 API")
    print("=" * 60)

    api = FunctionalCosmeticsAPI()
    df = api.match_all_products()
    api.save_results(df)
    return df


def run_ingredient():
    """원료성분정보 보강"""
    print("=" * 60)
    print("[2/3] 화장품 원료성분정보 API")
    print("=" * 60)

    api = IngredientInfoAPI()
    df = api.enrich_ingredients()
    api.save_results(df)
    return df


def run_restricted():
    """사용제한 원료 매칭"""
    print("=" * 60)
    print("[3/3] 화장품 사용제한 원료정보 API")
    print("=" * 60)

    api = RestrictedIngredientsAPI()
    df = api.match_ingredients()
    api.save_results(df)
    return df


def run_test():
    """단건 테스트"""
    print("=" * 60)
    print("MFDS API 단건 테스트")
    print("=" * 60)

    # 1. 기능성화장품
    print("\n[1] 기능성화장품 검색")
    func_api = FunctionalCosmeticsAPI()
    result = func_api.match_product("해서린 스팟 케어 클리어 젤 10ml", "TEST")
    matched = result.get("mfds_matched", False)
    print(f"  매칭: {matched}")
    if matched:
        print(f"  제품명: {result.get('mfds_item_name')}")
        print(f"  업체명: {result.get('mfds_entp_name')}")
        print(f"  미백: {result.get('is_whitening')}")
        print(f"  주름개선: {result.get('is_anti_wrinkle')}")
        print(f"  자외선차단: {result.get('is_sunscreen')}")

    # 2. 원료성분정보
    print("\n[2] 원료성분정보 검색")
    ingr_api = IngredientInfoAPI()
    result = ingr_api.search_by_name("토코페릴아세테이트")
    print(f"  영문명: {result['name_eng']}")
    print(f"  CAS No: {result['cas_no']}")

    # 3. 사용제한 원료
    print("\n[3] 사용제한 원료 (캐시 확인)")
    rest_api = RestrictedIngredientsAPI()
    if "all_restricted" in rest_api.cache:
        print(f"  캐시 데이터: {len(rest_api.cache['all_restricted'])}건")
    else:
        print("  캐시 없음 (--target restricted 로 먼저 다운로드 필요)")

    print("\n테스트 완료!")


def main():
    parser = argparse.ArgumentParser(description="MFDS API 데이터 수집")
    parser.add_argument(
        "--target",
        choices=["functional", "ingredient", "restricted", "all"],
        default="all",
        help="실행 대상 (기본: all)",
    )
    parser.add_argument("--test", action="store_true", help="단건 테스트 모드")
    args = parser.parse_args()

    if args.test:
        run_test()
        return

    if args.target in ("functional", "all"):
        run_functional()

    if args.target in ("ingredient", "all"):
        run_ingredient()

    if args.target in ("restricted", "all"):
        run_restricted()

    print("\n" + "=" * 60)
    print("전체 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
