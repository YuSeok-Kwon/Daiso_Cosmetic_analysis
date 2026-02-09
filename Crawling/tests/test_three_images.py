"""
세 가지 이미지 테스트 스크립트
- 1005699: 중간에 성분표 (이전 실패 케이스)
- 1061921: 하단에 성분표 (regression 확인)
- 1071595: 새 이미지
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from PIL import Image
from io import BytesIO
import numpy as np
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_single_image(image_url: str, product_id: str, expected_keywords: list = None):
    """단일 이미지 테스트"""

    save_dir = f"test_results/{product_id}"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"테스트: {product_id}")
    print(f"{'='*70}")

    try:
        # 1. 이미지 다운로드
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Referer': 'https://www.daisomall.co.kr/',
            'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8'
        }
        response = requests.get(image_url, headers=headers, timeout=30)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content))
        if image.mode != 'RGB':
            image = image.convert('RGB')

        print(f"원본 이미지 크기: {image.size[0]}x{image.size[1]}")

        # 원본 저장
        image.save(os.path.join(save_dir, "1_original.jpg"))

        # 2. 성분표 영역 감지
        from modules.ingredient_detector import crop_ingredient_region, detect_ingredient_region, visualize_detected_regions

        img_array = np.array(image)

        # 감지 영역 시각화
        visualize_detected_regions(img_array, os.path.join(save_dir, "2_detected_regions.jpg"))

        # 영역 감지 (상세 정보)
        detected = detect_ingredient_region(img_array)
        if detected:
            x, y, w, h = detected
            print(f"감지된 영역: x={x}, y={y}, w={w}, h={h}")
            print(f"  - y 위치: {y}/{img_array.shape[0]} ({y/img_array.shape[0]*100:.1f}%)")
        else:
            print("⚠️ 성분표 영역 감지 실패")

        # 3. 크롭
        cropped = crop_ingredient_region(image, auto_detect=True, expand_margin=30)
        cropped.save(os.path.join(save_dir, "3_cropped.jpg"))
        print(f"크롭 크기: {cropped.size[0]}x{cropped.size[1]}")

        # 4. OCR 수행
        ocr_text = None

        # Clova OCR 시도
        try:
            from modules.clova_ocr import ClovaOCR
            clova = ClovaOCR()

            if clova.api_url and clova.secret_key:
                img_buffer = BytesIO()
                cropped.save(img_buffer, format='JPEG')
                img_buffer.seek(0)

                ocr_text = clova.extract_text_from_bytes(img_buffer.read())
                if ocr_text:
                    print(f"✅ Clova OCR 성공 ({len(ocr_text)}자)")
        except Exception as e:
            print(f"Clova OCR 실패: {e}")

        # EasyOCR 폴백
        if not ocr_text:
            try:
                import easyocr
                reader = easyocr.Reader(['ko', 'en'], gpu=False, verbose=False)
                result = reader.readtext(np.array(cropped), detail=0, paragraph=True)
                ocr_text = ' '.join(result)
                print(f"✅ EasyOCR 성공 ({len(ocr_text)}자)")
            except Exception as e:
                print(f"❌ EasyOCR 실패: {e}")

        # 5. 결과 저장
        if ocr_text:
            with open(os.path.join(save_dir, "4_ocr_result.txt"), 'w', encoding='utf-8') as f:
                f.write(ocr_text)

            # 주요 키워드 확인
            keywords_found = []
            check_keywords = ['전성분', '[전성분]', '성분', 'INGREDIENTS', '정제수', '글리세린']
            for kw in check_keywords:
                if kw in ocr_text:
                    keywords_found.append(kw)

            print(f"발견된 키워드: {keywords_found}")
            print(f"\nOCR 결과 (처음 300자):\n{ocr_text[:300]}...")

            # 예상 키워드 검증
            if expected_keywords:
                found = sum(1 for kw in expected_keywords if kw in ocr_text)
                print(f"\n예상 성분 매칭: {found}/{len(expected_keywords)}")

        return True, ocr_text

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    # 테스트 이미지 목록
    test_cases = [
        {
            "id": "1005699",
            "url": "https://cdn.daisomall.co.kr/file/PD/20240722/8i4VYpbDp5lI8BbCQZn31005699_01_07.jpg",
            "description": "중간에 성분표 (이전 실패 케이스)",
            "expected": ["전성분", "정제수"]
        },
        {
            "id": "1061921",
            "url": "https://cdn.daisomall.co.kr/file/PD/20250113/SYfYo0BhM6oFdBd0Txz71061921_01_07SYfYo0BhM6oFdBd0Txz7.jpg",
            "description": "하단에 성분표 (regression 확인)",
            "expected": ["전성분", "정제수"]
        },
        {
            "id": "1071595",
            "url": "https://cdn.daisomall.co.kr/file/PD/20251121/dFlpwkzivPuW6av0YRGO1071595_01_07dFlpwkzivPuW6av0YRGO.jpg",
            "description": "새 이미지",
            "expected": ["정제수", "글리세린", "스테아릭애씨드", "숯가루", "병풀잎추출물"]
        }
    ]

    print("="*70)
    print("성분표 영역 감지 테스트 (3개 이미지)")
    print("="*70)

    results = {}

    for case in test_cases:
        print(f"\n📋 {case['id']}: {case['description']}")
        success, ocr_text = test_single_image(
            case["url"],
            case["id"],
            case.get("expected", [])
        )
        results[case["id"]] = {
            "success": success,
            "has_text": ocr_text is not None and len(ocr_text) > 50
        }

    # 최종 결과 요약
    print("\n" + "="*70)
    print("테스트 결과 요약")
    print("="*70)

    for case in test_cases:
        pid = case["id"]
        r = results[pid]
        status = "✅ 성공" if r["success"] and r["has_text"] else "❌ 실패"
        print(f"{pid}: {status} - {case['description']}")


if __name__ == "__main__":
    main()
