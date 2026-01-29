"""
다이소몰 스킨케어 제품만 크롤링하는 스크립트
"""
from daiso_beauty import crawl_category
from driver_setup import create_driver, quit_driver
from utils import *
from config import DAISO_BEAUTY_CATEGORIES, DAISO_CONFIG
import traceback

logger = setup_logger('daiso_skincare', 'daiso_skincare.log')


def crawl_skincare():
    """스킨케어 카테고리만 크롤링"""

    logger.info("=" * 60)
    logger.info("다이소 스킨케어 전용 크롤링 시작")
    logger.info("=" * 60)

    driver = None
    all_products = []

    try:
        # 드라이버 생성
        logger.info("Chrome 드라이버 생성 중")
        driver = create_driver()
        logger.info("드라이버 생성 완료")

        # 스킨케어 카테고리 정보
        skincare_data = DAISO_BEAUTY_CATEGORIES['스킨케어']
        middle_code = skincare_data['중분류코드']
        subcategories = skincare_data['소분류']

        max_products = DAISO_CONFIG['max_products_per_category']

        logger.info(f"\n스킨케어 소분류: {len(subcategories)}개")
        for code, name in subcategories.items():
            logger.info(f"  - {name} ({code})")

        # 각 소분류 크롤링
        for sub_code, sub_name in subcategories.items():
            logger.info(f"\n{'='*20} 스킨케어/{sub_name} {'='*20}")

            products = crawl_category(
                driver,
                main_category="스킨케어",
                sub_category_name=sub_name,
                middle_code=middle_code,
                sub_code=sub_code,
                max_products=max_products
            )

            all_products.extend(products)
            logger.info(f"{sub_name} 완료: {len(products)}개 수집")

            # 딜레이
            random_delay(2, 3)

        # 데이터 저장
        if all_products:
            date_str = get_date_string()
            filename = f'daiso_skincare_{date_str}.csv'
            filepath = save_to_csv(all_products, filename)

            logger.info(f"\n{'='*60}")
            logger.info(f"데이터 저장 완료: {filepath}")
            logger.info(f"총 {len(all_products)}개 스킨케어 제품 수집")
            logger.info(f"{'='*60}")

            # 소분류별 통계
            stats = {}
            for p in all_products:
                sub = p['sub_category']
                stats[sub] = stats.get(sub, 0) + 1

            logger.info("\n=== 소분류별 수집 현황 ===")
            for sub, count in sorted(stats.items()):
                logger.info(f"  - {sub}: {count}개")

            print(f"\n" + "=" * 70)
            print(f"✅ 스킨케어 크롤링 완료!")
            print(f"=" * 70)
            print(f"총 제품 수: {len(all_products)}개")
            print(f"\n소분류별:")
            for sub, count in sorted(stats.items()):
                print(f"  - {sub}: {count}개")
            print(f"\n저장 파일: {filepath}")
            print(f"=" * 70)

            return filepath
        else:
            logger.warning("수집된 데이터가 없습니다.")
            print("\n수집된 데이터가 없습니다.")
            return None

    except Exception as e:
        logger.error(f"크롤링 실패: {e}")
        logger.error(traceback.format_exc())
        print(f"\n크롤링 실패: {e}")
        return None

    finally:
        if driver:
            logger.info("브라우저 종료 중")
            quit_driver(driver)
            logger.info("브라우저 종료 완료")


if __name__ == "__main__":
    import sys

    print("\n" + "=" * 70)
    print("다이소몰 스킨케어 전용 크롤러")
    print("=" * 70)
    print("\n크롤링 대상:")
    print("  - 스킨케어 카테고리 (5개 소분류)")
    print("    ├ 기초스킨케어")
    print("    ├ 립케어")
    print("    ├ 팩/마스크")
    print("    ├ 자외선차단제")
    print("    └ 클렌징/필링")
    print("\n주의: 교육/연구 목적으로만 사용하세요.")
    print("=" * 70)

    # 백그라운드 실행 체크
    if sys.stdin.isatty():
        input("\n시작하려면 Enter를 누르세요...")
    else:
        print("\n자동 시작...")

    csv_path = crawl_skincare()

    if csv_path:
        print("\n다음 단계:")
        print(f"1. 전성분 추출 크롤링을 실행하세요:")
        print(f"   python daiso_ingredients.py")
        print(f"2. 입력 파일 경로: {csv_path}")
