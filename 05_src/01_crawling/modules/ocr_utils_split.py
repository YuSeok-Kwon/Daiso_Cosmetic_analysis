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


def extract_text_bottom_up_3920(url, use_clova=True):
    """
    하단에서 위로 3920px씩 OCR (긴 단일 이미지용)

    - 1차: 하단 1960px OCR → 키워드 찾으면 반환
    - 2차: 그 위 1960px OCR → 키워드 찾으면 반환 (총 3920px)
    - 3차: 그 위 1960px OCR → 키워드 찾으면 반환
    - 4차: 그 위 1960px OCR → 키워드 찾으면 반환 (총 7840px)

    Args:
        url: 이미지 URL
        use_clova: Naver Clova OCR 사용 여부

    Returns:
        list: OCR 결과 리스트 [{'section': 1, 'text': '...', 'method': 'clova_bottom_up'}, ...]
    """
    CLOVA_LIMIT = 1960
    MAX_CROPS = 4  # 최대 4번 크롭 (7840px)

    # 성분 키워드 (전성분 헤더)
    HEADER_KEYWORDS = ['전성분', '성분:', '모든성분', '화장품법', 'INGREDIENTS', 'ingredients', '성분명', '[전성분]']
    # 대표 성분명 (헤더 없이도 감지)
    COMMON_INGREDIENTS = [
        '정제수', '글리세린', '부틸렌글라이콜', '프로판다이올', '나이아신아마이드',
        '히알루론산', '판테놀', '알란토인', '토코페롤', '카보머', '페녹시에탄올',
        '향료', '시트릭애씨드', '다이메티콘', '스쿠알란', '세틸알코올',
    ]

    try:
        # 이미지 다운로드
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        original_image = Image.open(BytesIO(response.content))
        height = original_image.size[1]
        width = original_image.size[0]
        logger.info(f"[Bottom-Up OCR] 이미지 다운로드: {width}x{height}")

        all_results = []

        for crop_idx in range(MAX_CROPS):
            # 하단에서 위로 크롭 위치 계산
            bottom = height - (crop_idx * CLOVA_LIMIT)
            top = max(0, bottom - CLOVA_LIMIT)

            if bottom <= 0:
                logger.info(f"[Bottom-Up OCR] 이미지 상단 도달, 크롭 중단")
                break

            # 크롭
            cropped = original_image.crop((0, top, width, bottom))
            crop_height = bottom - top
            logger.info(f"[Bottom-Up OCR] {crop_idx+1}차 크롭: y {top}-{bottom} ({crop_height}px)")

            # Clova OCR 호출
            clova_text = None
            if use_clova:
                try:
                    from .clova_ocr import get_clova_client
                    clova_client = get_clova_client()

                    if clova_client.is_available():
                        img_byte_arr = BytesIO()
                        cropped.save(img_byte_arr, format='PNG')
                        img_bytes = img_byte_arr.getvalue()
                        clova_text = clova_client.extract_text_from_bytes(img_bytes, image_format='png')
                except Exception as e:
                    logger.warning(f"[Bottom-Up OCR] Clova 실패: {str(e)}")

            if not clova_text:
                # EasyOCR 폴백
                cropped = preprocess_image_for_ocr(cropped)
                reader = get_ocr_reader()
                ocr_result = reader.readtext(np.array(cropped), detail=0, paragraph=True)
                clova_text = ' '.join(ocr_result) if ocr_result else ''

            # 결과 저장
            all_results.append({
                'section': crop_idx + 1,
                'text': clova_text or '',
                'char_count': len(clova_text) if clova_text else 0,
                'height_range': (top, bottom),
                'method': 'clova_bottom_up'
            })

            # 키워드 검증
            if clova_text:
                text_flat = clova_text.replace('\n', ' ').replace('  ', ' ')
                text_nospace = clova_text.replace('\n', '').replace(' ', '')

                has_header = any(kw.replace(' ', '') in text_nospace.lower() for kw in HEADER_KEYWORDS)
                ingredient_count = sum(1 for ing in COMMON_INGREDIENTS if ing in text_flat)

                if has_header or ingredient_count >= 3:
                    logger.info(f"[Bottom-Up OCR] {crop_idx+1}차에서 성분 키워드 발견! (헤더: {has_header}, 성분: {ingredient_count}개)")
                    return all_results
                else:
                    logger.info(f"[Bottom-Up OCR] {crop_idx+1}차: 키워드 미발견 (헤더: {has_header}, 성분: {ingredient_count}개) → 위 영역 탐색")

        logger.info(f"[Bottom-Up OCR] 총 {len(all_results)}개 영역 OCR 완료")
        return all_results

    except Exception as e:
        logger.error(f"[Bottom-Up OCR] 실패: {str(e)}")
        return []


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
        crop_success = False
        original_image = image
        if auto_crop:
            try:
                from .ingredient_detector import crop_ingredient_region
                image = crop_ingredient_region(image, auto_detect=True, expand_margin=30)

                # 크롭이 성공적으로 수행되었는지 확인 (크기가 줄어들어야 함)
                if image.size[0] < original_image.size[0] * 0.9 or image.size[1] < original_image.size[1] * 0.9:
                    crop_success = True
                    logger.info(f"성분표 영역 크롭 완료: {original_image.size} → {image.size}")
                else:
                    logger.warning("크롭 영역이 원본과 비슷합니다. 성분표 감지 실패로 판단.")
                    image = original_image
            except Exception as e:
                logger.warning(f"성분표 영역 크롭 실패 (원본 사용): {str(e)}")

        # 1-1. Clova OCR 장축 제한 1960px 처리
        # 성분표 영역이 이미 크롭되었으면 그 영역 기준으로 처리
        exceeds_clova_limit = image.size[1] > 1960

        if exceeds_clova_limit:
            # 이미 성분표 영역이 크롭되었으면 (crop_success=True), 그 영역 내에서 추가 크롭
            # 성분표 크롭이 실패했으면, 원본에서 하단 크롭
            if crop_success:
                # 성분표 영역 내에서 Clova 제한에 맞게 크롭
                # 매우 긴 이미지(>5000px)의 하단 스캔에서 발견된 경우:
                #   - '전성분' 키워드는 3500px 영역의 하단에 있을 가능성 높음
                #   - 따라서 하단 1960px를 크롭
                # 일반 이미지: 상단 우선 (기존 로직)
                height = image.size[1]

                if original_image.size[1] > 5000:
                    # 매우 긴 이미지 → 하단 크롭 (성분표가 하단 스캔에서 발견됨)
                    bottom_height = min(1960, height)
                    bottom_start = height - bottom_height
                    image = image.crop((0, bottom_start, image.size[0], height))
                    logger.info(f"성분표 영역 내 Clova 크롭 (하단): → {image.size} (y: {bottom_start}-{height})")
                else:
                    # 일반 이미지 → 상단 크롭
                    top_height = min(1960, height)
                    image = image.crop((0, 0, image.size[0], top_height))
                    logger.info(f"성분표 영역 내 Clova 크롭 (상단): → {image.size} (y: 0-{top_height})")
            else:
                # 성분표 감지 실패 시 원본에서 하단 크롭
                height = original_image.size[1]
                bottom_height = min(int(height * 0.15), 1960)  # 하단 15% 또는 최대 1960px
                bottom_start = height - bottom_height
                image = original_image.crop((0, bottom_start, original_image.size[0], height))
                logger.info(f"Clova 제한 초과 → 하단 크롭: {original_image.size} → {image.size} (y: {bottom_start}-{height})")

        # 2. Clova OCR 시도
        clova_text = None
        if use_clova:
            try:
                from .clova_ocr import get_clova_client
                clova_client = get_clova_client()

                if clova_client.is_available():
                    # 크롭/분할된 이미지는 바이트로 전달, 원본 그대로면 URL 전달
                    image_was_modified = (image.size != original_image.size)

                    if image_was_modified:
                        img_byte_arr = BytesIO()
                        image.save(img_byte_arr, format='PNG')
                        img_bytes = img_byte_arr.getvalue()
                        logger.info(f"Clova OCR 호출 (크롭/분할 이미지: {image.size[0]}x{image.size[1]})")
                        clova_text = clova_client.extract_text_from_bytes(img_bytes, image_format='png')
                    else:
                        # 원본 그대로인 경우 URL로 호출 (더 효율적)
                        logger.info(f"Clova OCR 호출 (원본 URL 사용)")
                        clova_text = clova_client.extract_text_from_url(url)

                    # 성분 키워드 + 실제 성분명이 함께 있어야 함
                    # 다양한 화장품 유형에서 자주 등장하는 성분들 (스킨케어, 클렌징, 헤어, 바디 등)
                    ACTUAL_INGREDIENTS = [
                        # 기본 성분
                        '정제수', '글리세린', '부틸렌글라이콜', '프로판다이올',
                        # 스킨케어 성분
                        '나이아신아마이드', '토코페롤', '히알루론산', '판테놀', '알란토인',
                        # 계면활성제/클렌징
                        '라우릴', '설페이트', '코카미도', '베타인', '올레핀',
                        # 유화제/안정제
                        '폴리소르베이트', '스테아릭', '트리에탄올아민', '카보머',
                        # 보존제/향료
                        '벤질알코올', '페녹시에탄올', '멘톨', '향료', '리모넨',
                        # 오일/왁스
                        '미네랄오일', '시어버터', '호호바', '스쿠알란',
                        # 쉐이빙 제품 성분 추가
                        '이소부탄', '팔미틱', '소르비톨', '코코넛', '프로판', '라우릭',
                        '포타슘', '하이드록사이드', '피브이피',
                    ]

                    # 줄바꿈/공백 제거한 텍스트로 검증 (Clova OCR이 단어별로 줄바꿈하는 경우 대응)
                    clova_text_flat = clova_text.replace('\n', ' ').replace('  ', ' ') if clova_text else ''
                    # 공백 완전 제거 버전 (헤더 키워드 검증용)
                    clova_text_nospace = clova_text.replace('\n', '').replace(' ', '') if clova_text else ''

                    # 헤더 키워드 검증 (공백 제거 후 검증 - "전 성분", "전\n성분" 등 대응)
                    HEADER_KEYWORDS_NOSPACE = ['전성분', '성분:', '모든성분', '화장품법', 'INGREDIENTS', 'ingredients', '성분명', '[성분명]', '성분은']
                    has_header = any(kw.replace(' ', '') in clova_text_nospace.lower() for kw in HEADER_KEYWORDS_NOSPACE) if clova_text_nospace else False

                    # 실제 성분 개수 (공백 포함 텍스트에서)
                    ingredient_count = sum(1 for ing in ACTUAL_INGREDIENTS if ing in clova_text_flat) if clova_text_flat else 0

                    # 디버깅 로그 (첫 100자 미리보기 추가)
                    preview = clova_text_flat[:100].replace('\n', ' ') if clova_text_flat else ''
                    logger.info(f"Clova 검증: has_header={has_header}, ingredient_count={ingredient_count}, text_len={len(clova_text_flat)}")
                    logger.debug(f"Clova 텍스트 미리보기: {preview}...")

                    # Clova 결과 검증 (완화됨):
                    # 1) 헤더 + 1개 이상 성분, 또는
                    # 2) 2개 이상 성분, 또는
                    # 3) 성분표 크롭 성공 + 텍스트 200자 이상 (크롭이 성공했으면 성분표일 가능성 높음)
                    has_ingredient_keywords = (
                        (has_header and ingredient_count >= 1) or
                        ingredient_count >= 2 or
                        (crop_success and len(clova_text_flat) > 200)
                    ) if clova_text_flat else False

                    if clova_text and len(clova_text) > 50 and has_ingredient_keywords:
                        logger.info(f"Clova OCR 성공: {len(clova_text)}자 추출 (성분 키워드 확인됨)")

                        # Clova 결과를 단일 섹션으로 반환
                        return [{
                            'section': 1,
                            'text': clova_text,
                            'char_count': len(clova_text),
                            'height_range': (0, image.size[1]),
                            'method': 'clova_ocr'
                        }]
                    else:
                        reason = "성분 키워드 없음" if clova_text and len(clova_text) > 50 else f"{len(clova_text) if clova_text else 0}자"
                        logger.warning(f"Clova OCR 결과 부적합 ({reason}). EasyOCR로 폴백합니다.")
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
