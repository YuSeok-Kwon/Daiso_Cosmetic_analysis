"""
이미지 분할 OCR 유틸리티
긴 이미지를 여러 섹션으로 나눠서 OCR 처리
"""
import easyocr
import requests
import numpy as np
from PIL import Image, ImageEnhance
from io import BytesIO
import logging
import cv2
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

logger = logging.getLogger(__name__)

# EasyOCR Reader 싱글톤
_reader = None


def get_ocr_reader():
    """EasyOCR Reader 인스턴스 반환 (싱글톤)"""
    global _reader
    if _reader is None:
        logger.info("EasyOCR Reader 초기화 중 (한국어, 영어)")
        _reader = easyocr.Reader(['ko', 'en'], gpu=False)
        logger.info("EasyOCR Reader 초기화 완료")
    return _reader


def preprocess_image_for_ocr(image):
    """
    OCR 정확도 향상을 위한 이미지 전처리

    Args:
        image: PIL Image 객체

    Returns:
        PIL Image: 전처리된 이미지
    """
    # 1. RGB 변환
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # 2. 작은 이미지는 확대 (1.5배)
    if image.width < 1500:
        scale = 1.5
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        logger.debug(f"이미지 확대: {new_width}x{new_height}")

    # 3. 대비 향상 (1.6배)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.6)

    # 4. 선명도 향상 (1.4배)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.4)

    # 5. 밝기 약간 증가 (1.1배)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.1)

    logger.debug("이미지 전처리 완료 (대비, 선명도, 밝기 향상)")
    return image


def split_image_vertically(image, num_sections=3):
    """
    이미지를 수직으로 N등분

    Args:
        image: PIL Image 객체
        num_sections: 분할할 섹션 수 (기본 3)

    Returns:
        list: PIL Image 섹션 리스트
    """
    width, height = image.size
    section_height = height // num_sections

    sections = []
    for i in range(num_sections):
        top = i * section_height
        bottom = height if i == num_sections - 1 else (i + 1) * section_height

        section = image.crop((0, top, width, bottom))
        sections.append(section)
        logger.info(f"섹션 {i+1}/{num_sections}: {width}x{bottom-top} (y: {top}-{bottom})")

    return sections


def extract_text_from_image_url_split(url, num_sections=3, use_clova=True, auto_crop=True):
    """
    이미지 URL에서 텍스트 추출 (분할 방식 + Clova OCR + 자동 크롭)

    Args:
        url: 이미지 URL
        num_sections: 분할할 섹션 수 (기본 3)
        use_clova: Naver Clova OCR 사용 여부
        auto_crop: 성분표 영역 자동 크롭 여부

    Returns:
        list: 각 섹션의 OCR 텍스트 딕셔너리 리스트
              [{'section': 1, 'text': '...', 'height_range': (0, 1000), 'method': 'clova'}, ...]
    """
    try:
        # 이미지 다운로드
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # PIL Image로 변환
        image = Image.open(BytesIO(response.content))
        logger.info(f"이미지 다운로드 완료: {image.size[0]}x{image.size[1]}")

        # 1. 성분표 영역 자동 크롭 (선택적)
        if auto_crop:
            try:
                from ingredient_detector import crop_ingredient_region
                original_image = image
                image = crop_ingredient_region(image, auto_detect=True, expand_margin=30)

                if image.size != original_image.size:
                    logger.info(f"성분표 영역 크롭 완료: {original_image.size} → {image.size}")
            except Exception as e:
                logger.warning(f"성분표 영역 크롭 실패 (원본 사용): {str(e)}")

        # 2. Clova OCR 시도 (전체 이미지)
        clova_text = None
        if use_clova:
            try:
                from clova_ocr import get_clova_client
                clova_client = get_clova_client()

                if clova_client.is_available():
                    # URL 직접 전달 (Clova는 URL 지원)
                    clova_text = clova_client.extract_text_from_url(url)

                    if clova_text and len(clova_text) > 100:
                        logger.info(f"Clova OCR 성공: {len(clova_text)}자 추출")

                        # Clova 결과를 단일 섹션으로 반환
                        return [{
                            'section': 1,
                            'text': clova_text,
                            'char_count': len(clova_text),
                            'height_range': (0, image.size[1]),
                            'method': 'clova_ocr'
                        }]
                    else:
                        logger.warning("Clova OCR 결과가 부족합니다. EasyOCR로 폴백합니다.")
            except Exception as e:
                logger.warning(f"Clova OCR 실패 (EasyOCR로 폴백): {str(e)}")

        # 3. 이미지 전처리 (OCR 정확도 향상)
        image = preprocess_image_for_ocr(image)

        # 이미지 분할
        sections = split_image_vertically(image, num_sections)

        # 각 섹션 OCR
        reader = get_ocr_reader()
        results = []

        for idx, section in enumerate(sections, 1):
            logger.info(f"섹션 {idx} OCR 시작...")

            # PIL Image를 numpy array로 변환
            section_array = np.array(section)

            # 1. EasyOCR 실행
            ocr_result = reader.readtext(section_array, detail=0, paragraph=True)
            easyocr_text = ' '.join(ocr_result) if ocr_result else ''

            # 2. Tesseract OCR 실행 (사용 가능한 경우)
            tesseract_text = ""
            if TESSERACT_AVAILABLE:
                tesseract_text = extract_text_with_tesseract(section)

            # 3. 더 긴 결과 선택 (보통 더 많이 추출한 것이 더 좋음)
            text = easyocr_text if len(easyocr_text) >= len(tesseract_text) else tesseract_text

            if len(tesseract_text) > len(easyocr_text):
                logger.info(f"섹션 {idx}: Tesseract가 더 많은 텍스트 추출 ({len(tesseract_text)} vs {len(easyocr_text)})")

            # 높이 범위 계산
            section_height = image.size[1] // num_sections
            top = (idx - 1) * section_height
            bottom = image.size[1] if idx == num_sections else idx * section_height

            results.append({
                'section': idx,
                'text': text,
                'char_count': len(text),
                'height_range': (top, bottom)
            })

            logger.info(f"섹션 {idx} OCR 완료: {len(text)}자 추출")

        return results

    except Exception as e:
        logger.error(f"이미지 분할 OCR 실패: {str(e)}")
        return []


def extract_text_with_tesseract(image):
    """
    Tesseract OCR로 텍스트 추출

    Args:
        image: PIL Image 객체

    Returns:
        str: 추출된 텍스트
    """
    if not TESSERACT_AVAILABLE:
        logger.warning("Tesseract를 사용할 수 없습니다")
        return ""

    try:
        # Tesseract 설정 (한국어 + 영어)
        custom_config = r'--oem 3 --psm 6 -l kor+eng'

        # 이미지를 numpy array로 변환
        img_array = np.array(image)

        # OCR 실행
        text = pytesseract.image_to_string(img_array, config=custom_config)

        logger.debug(f"Tesseract OCR 완료: {len(text)}자 추출")
        return text

    except Exception as e:
        logger.error(f"Tesseract OCR 실패: {str(e)}")
        return ""


def extract_text_from_image_url(url):
    """
    이미지 URL에서 텍스트 추출 (기존 방식 - 호환성 유지)

    Args:
        url: 이미지 URL

    Returns:
        str: 추출된 텍스트 (모든 섹션 통합)
    """
    sections = extract_text_from_image_url_split(url, num_sections=3)

    if not sections:
        return ''

    # 모든 섹션의 텍스트 통합
    full_text = '\n\n'.join([s['text'] for s in sections if s['text']])
    return full_text
