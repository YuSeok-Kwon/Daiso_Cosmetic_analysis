"""
성분표 자동 감지 테스트 스크립트
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import logging
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Selenium 설정
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time


def get_product_images(url: str):
    """제품 페이지에서 이미지 URL 추출"""
    logger.info(f"제품 페이지 접속: {url}")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    try:
        driver.get(url)

        # 페이지 로딩 대기
        time.sleep(3)

        # 스크롤 다운 (상세 이미지 로드)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        # 상세 이미지 찾기
        images = driver.find_elements(By.CSS_SELECTOR, "div.editor-content picture img")

        image_urls = []
        for img in images:
            src = img.get_attribute("src")
            if src:
                image_urls.append(src)

        logger.info(f"이미지 {len(image_urls)}개 발견")

        return image_urls

    finally:
        driver.quit()


def test_ingredient_detection(image_url: str, save_dir: str = "test_results"):
    """성분표 영역 감지 테스트"""

    os.makedirs(save_dir, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"테스트 이미지: {image_url[:80]}...")
    logger.info(f"{'='*60}")

    # 1. 이미지 다운로드
    response = requests.get(image_url, timeout=30)
    image = Image.open(BytesIO(response.content))

    logger.info(f"원본 이미지 크기: {image.size[0]}x{image.size[1]}")

    # 원본 이미지 저장
    original_path = os.path.join(save_dir, "1_original.jpg")
    image.save(original_path)
    logger.info(f"원본 저장: {original_path}")

    # 2. 성분표 영역 감지
    from modules.ingredient_detector import crop_ingredient_region, visualize_detected_regions
    import numpy as np

    # 감지된 영역 시각화
    img_array = np.array(image)
    vis_path = os.path.join(save_dir, "2_detected_regions.jpg")
    visualize_detected_regions(img_array, vis_path)
    logger.info(f"영역 감지 시각화: {vis_path}")

    # 3. 성분표 영역 크롭
    cropped = crop_ingredient_region(image, auto_detect=True, expand_margin=30)

    crop_path = os.path.join(save_dir, "3_cropped_ingredient.jpg")
    cropped.save(crop_path)

    logger.info(f"크롭된 이미지 크기: {cropped.size[0]}x{cropped.size[1]}")
    logger.info(f"크롭 이미지 저장: {crop_path}")

    # 4. OCR 수행 (Clova OCR 우선, 실패 시 EasyOCR)
    ocr_text = None
    ocr_method = None

    # 4-1. Clova OCR 시도
    try:
        from modules.clova_ocr import ClovaOCR
        clova = ClovaOCR()

        if clova.api_url and clova.secret_key:
            logger.info("Clova OCR 시도 중...")

            # PIL Image를 bytes로 변환
            img_buffer = BytesIO()
            cropped.save(img_buffer, format='JPEG')
            img_buffer.seek(0)

            ocr_text = clova.extract_text_from_bytes(img_buffer.read())

            if ocr_text:
                ocr_method = "Clova OCR"
                logger.info(f"✅ Clova OCR 성공! ({len(ocr_text)}자)")
        else:
            logger.warning("Clova OCR API 키가 설정되지 않았습니다.")
    except Exception as e:
        logger.warning(f"Clova OCR 실패: {str(e)}")

    # 4-2. Clova OCR 실패 시 EasyOCR 폴백
    if not ocr_text:
        try:
            logger.info("EasyOCR로 폴백...")
            import easyocr
            reader = easyocr.Reader(['ko', 'en'], gpu=False)

            result = reader.readtext(np.array(cropped), detail=0, paragraph=True)
            ocr_text = ' '.join(result)
            ocr_method = "EasyOCR"
            logger.info(f"✅ EasyOCR 성공! ({len(ocr_text)}자)")
        except Exception as e:
            logger.error(f"EasyOCR도 실패: {str(e)}")
            return

    if ocr_text:
        logger.info(f"\n{'='*60}")
        logger.info(f"OCR 결과 ({ocr_method}, {len(ocr_text)}자):")
        logger.info(f"{'='*60}")
        logger.info(ocr_text[:500])

        # OCR 결과 저장
        ocr_result_path = os.path.join(save_dir, "4_ocr_result.txt")
        with open(ocr_result_path, 'w', encoding='utf-8') as f:
            f.write(f"[{ocr_method}]\n\n")
            f.write(ocr_text)
        logger.info(f"OCR 결과 저장: {ocr_result_path}")

        # 5. 후처리
        try:
            from modules.ingredient_postprocessor import postprocess_ingredients

            ingredients = postprocess_ingredients(ocr_text, use_dictionary=True)

            logger.info(f"\n{'='*60}")
            logger.info(f"추출된 성분 ({len(ingredients)}개):")
            logger.info(f"{'='*60}")

            for idx, ing in enumerate(ingredients[:20], 1):
                logger.info(f"{idx}. {ing['corrected']} (신뢰도: {ing['confidence']:.2f})")

            if len(ingredients) > 20:
                logger.info(f"... 외 {len(ingredients) - 20}개")

            # 성분 리스트 저장
            ingredient_list_path = os.path.join(save_dir, "5_ingredients.txt")
            with open(ingredient_list_path, 'w', encoding='utf-8') as f:
                for ing in ingredients:
                    f.write(f"{ing['corrected']}\n")
            logger.info(f"성분 리스트 저장: {ingredient_list_path}")

        except Exception as e:
            logger.error(f"후처리 실패: {str(e)}")
    else:
        logger.error("OCR 결과가 없습니다.")

    logger.info(f"\n{'='*60}")
    logger.info(f"테스트 완료! 결과 폴더: {save_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    # 사용자가 제공한 제품 URL
    product_url = "https://www.daisomall.co.kr/pd/pdr/SCR_PDR_0001?pdNo=1018161&recmYn=N"

    print("="*60)
    print("성분표 자동 감지 테스트")
    print("="*60)

    # 1. 제품 이미지 URL 추출
    image_urls = get_product_images(product_url)

    if not image_urls:
        print("❌ 이미지를 찾을 수 없습니다")
    else:
        print(f"\n✅ 이미지 {len(image_urls)}개 발견")

        # 2. 마지막 이미지 테스트 (성분표는 보통 마지막에 위치)
        print(f"\n마지막 이미지로 테스트합니다 (성분표는 보통 마지막 이미지)")
        test_ingredient_detection(image_urls[-1], save_dir="test_results")
