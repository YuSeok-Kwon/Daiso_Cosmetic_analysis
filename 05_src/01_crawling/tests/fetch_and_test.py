"""
Selenium으로 제품 페이지에서 이미지 URL 추출 후 테스트
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
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_image_url_from_page(product_id: str) -> str:
    """제품 페이지에서 상세 이미지 URL 추출"""
    url = f"https://www.daisomall.co.kr/pd/pdr/SCR_PDR_0001?pdNo={product_id}&recmYn=N"

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        print(f"페이지 접속: {url}")
        driver.get(url)
        time.sleep(3)

        # 스크롤 다운
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        # 상세 이미지 찾기
        images = driver.find_elements(By.CSS_SELECTOR, "div.editor-content picture img")

        if images:
            # 마지막 이미지 (보통 성분표)
            last_img = images[-1]
            src = last_img.get_attribute("src")
            print(f"이미지 URL 발견: {src[:80]}...")
            return src
        else:
            print("이미지를 찾을 수 없습니다")
            return None

    finally:
        driver.quit()


def test_image(image_url: str, product_id: str, session=None):
    """이미지 테스트"""
    save_dir = f"test_results/{product_id}"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"테스트: {product_id}")
    print(f"{'='*70}")

    try:
        # 이미지 다운로드
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Referer': 'https://www.daisomall.co.kr/'
        }

        if session:
            response = session.get(image_url, headers=headers, timeout=30)
        else:
            response = requests.get(image_url, headers=headers, timeout=30)

        response.raise_for_status()

        image = Image.open(BytesIO(response.content))
        if image.mode != 'RGB':
            image = image.convert('RGB')

        print(f"원본 이미지 크기: {image.size[0]}x{image.size[1]}")
        image.save(os.path.join(save_dir, "1_original.jpg"))

        # 성분표 영역 감지
        from modules.ingredient_detector import crop_ingredient_region, detect_ingredient_region, visualize_detected_regions

        img_array = np.array(image)
        visualize_detected_regions(img_array, os.path.join(save_dir, "2_detected_regions.jpg"))

        detected = detect_ingredient_region(img_array)
        if detected:
            x, y, w, h = detected
            print(f"감지된 영역: x={x}, y={y}, w={w}, h={h}")
            print(f"  - y 위치: {y}/{img_array.shape[0]} ({y/img_array.shape[0]*100:.1f}%)")
        else:
            print("⚠️ 성분표 영역 감지 실패")

        # 크롭
        cropped = crop_ingredient_region(image, auto_detect=True, expand_margin=30)
        cropped.save(os.path.join(save_dir, "3_cropped.jpg"))
        print(f"크롭 크기: {cropped.size[0]}x{cropped.size[1]}")

        # OCR
        ocr_text = None

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

        if not ocr_text:
            try:
                import easyocr
                reader = easyocr.Reader(['ko', 'en'], gpu=False, verbose=False)
                result = reader.readtext(np.array(cropped), detail=0, paragraph=True)
                ocr_text = ' '.join(result)
                print(f"✅ EasyOCR 성공 ({len(ocr_text)}자)")
            except Exception as e:
                print(f"❌ EasyOCR 실패: {e}")

        if ocr_text:
            with open(os.path.join(save_dir, "4_ocr_result.txt"), 'w', encoding='utf-8') as f:
                f.write(ocr_text)

            keywords_found = []
            for kw in ['전성분', '[전성분]', '성분', 'INGREDIENTS', '정제수', '글리세린']:
                if kw in ocr_text:
                    keywords_found.append(kw)

            print(f"발견된 키워드: {keywords_found}")
            print(f"\nOCR 결과 (처음 500자):\n{ocr_text[:500]}...")

        return True, ocr_text, detected

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def main():
    test_products = [
        {"id": "1005699", "description": "중간에 성분표"},
        {"id": "1061921", "description": "하단에 성분표"},
        {"id": "1071595", "description": "새 이미지 (URL 직접 사용)",
         "url": "https://cdn.daisomall.co.kr/file/PD/20251121/dFlpwkzivPuW6av0YRGO1071595_01_07dFlpwkzivPuW6av0YRGO.jpg"}
    ]

    print("="*70)
    print("성분표 영역 감지 테스트")
    print("="*70)

    results = {}

    for product in test_products:
        pid = product["id"]

        # URL이 직접 제공된 경우
        if "url" in product:
            image_url = product["url"]
        else:
            # Selenium으로 URL 추출
            print(f"\n📋 {pid}: Selenium으로 이미지 URL 추출 중...")
            image_url = get_image_url_from_page(pid)

        if image_url:
            success, ocr_text, detected = test_image(image_url, pid)
            results[pid] = {
                "success": success,
                "has_text": ocr_text is not None and len(ocr_text) > 50,
                "detected": detected,
                "description": product["description"]
            }
        else:
            results[pid] = {
                "success": False,
                "has_text": False,
                "detected": None,
                "description": product["description"]
            }

    # 결과 요약
    print("\n" + "="*70)
    print("테스트 결과 요약")
    print("="*70)

    for pid, r in results.items():
        if r["success"] and r["has_text"]:
            status = "✅ 성공"
            if r["detected"]:
                x, y, w, h = r["detected"]
                status += f" (영역: y={y})"
        else:
            status = "❌ 실패"
        print(f"{pid}: {status} - {r['description']}")


if __name__ == "__main__":
    main()
