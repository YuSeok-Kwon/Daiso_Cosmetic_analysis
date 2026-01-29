"""
전성분 추출 로직 테스트
실제 크롤링 없이 alt 텍스트 파싱만 테스트
"""
from daiso_ingredients import extract_ingredients_from_alt
import json


def test_extract_ingredients():
    """제공된 예시 alt 텍스트로 테스트"""

    # 사용자가 제공한 alt 텍스트 예시
    test_alt = """데일리콤마 모스트 오 드 퍼퓸 우디 가든 30ml/1.0fl.oz (우디가든) 30mlPRODUCT INFO 데일리콤마모스트 퍼퓸 프랑스에센셜오일이 향수로 일상의향기를 특별하게 만들어줍니다. 부드럽고 은은함을 풍부하게 만들어주는 프랑스에센셜오일은 한국 더해중과 목련소시에이 함유되어 있습니다. Most eau de parfum 오랫동안 지속되는 은은한 잔향으로 특별함 속에 머물러보세요. Base ndalwood.cedarwood MUSK MUGUET 문성학에서 나는 MUSK darwood. TENDER PARIS 호환특성이 발표될 위험회주민를 분말을 하요. orange-grape.cass Middle cedarwood.vanilla MUSK MUGUET 향이 결합해 달콤하면서도 Middle powdery sweet vanilla⌀y [사용방법] 예금을 두고 미스트를 사용하여 얼굴을 제외한 원하는 부위에 분사해줍니다. [전성분] 우디가든 정제수, 프로필렌글라이콜 라벤더의 오일, 향료, 제라니올, 리모넨, 에칠헥실메톡시신나메이트 룽글 머수쿠, 폴리, 에틸렌스, 콘테스, 프로폴리아 아픔, 라벤더오일, 항로 라남을 하이드록시시트로벌일, Tender Paris: 에탄올, 정제수 프로필렌글라이콜라벤더오일, 향료 벤질알코올, 시트랄, 벤질살리실레이트 시그널베리 최강미녀언급출장샵강추출장샵콜걸출장안마오일, 장로 부틸페닐메틸프로피오날, 리날롤, 코튼화이트홀에 ⌀10mmx⌀10mm 즈에이트, 헥실신남알, 리모넨,알파-아이: 메틸아이 ⌀1000000ftx30ft:1ft 있는 경우 전문의 동화장구매생겼 21상처가 있는 부위등에는 사용의자제할것 備lyttly是k首位百货 가족관계 않는곳에 보관할 것 가) 어린 4)눈에서 금과일의사용할 * 본 상품 정보(상품 상세, 상품 설명등)의 내용은 협력사가 ·직접 등록한 것입니다. *향특성상 계옆에따라 색상이반할수있느냐들을및사용하시는하는 전하여성이없습니다. [품번] 1043751 [품명] 데일리콤마 모스트 오 드 퍼퓸 (우디가든) 30 ml [용량] 30 ml [화장품제조업자 및 화장품책임판매업자] 에이디인터 내셔날(주), 경기도 파주시 탄현면 축현산단로 56-14 [반품 및 교환장소] 구입처 및 판매원 [소비자상담실] 1522-4400"""

    print("=" * 70)
    print("전성분 추출 테스트")
    print("=" * 70)

    # 전성분 추출
    result = extract_ingredients_from_alt(test_alt)

    # 결과 출력
    print(f"\n[제품명]: {result['product_name']}")
    print(f"[용량]: {result['volume']}")
    print(f"[제조업자]: {result['manufacturer']}")

    print(f"\n[전성분 원문]:")
    print(result['raw_ingredients'][:200] + "..." if len(result['raw_ingredients']) > 200 else result['raw_ingredients'])

    print(f"\n[추출된 성분] (총 {len(result['ingredients'])}개):")
    if result['ingredients']:
        for idx, ing_data in enumerate(result['ingredients'][:20], 1):  # 처음 20개만 표시
            variant = f"[{ing_data['product_variant']}]" if ing_data['product_variant'] else ""
            print(f"  {idx}. {variant} {ing_data['ingredient']}")

        if len(result['ingredients']) > 20:
            print(f"  ... 외 {len(result['ingredients']) - 20}개")
    else:
        print("  (성분을 추출하지 못했습니다)")

    print("\n" + "=" * 70)

    # JSON 형태로도 출력
    print("\n[JSON 형식]:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    return result


if __name__ == "__main__":
    test_extract_ingredients()
