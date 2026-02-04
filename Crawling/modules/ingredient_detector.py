"""
성분표 영역 자동 감지 및 크롭 모듈
- 이미지에서 성분표가 있는 영역만 자동으로 찾아서 잘라냄
- OCR 정확도 향상
"""

import cv2
import numpy as np
from PIL import Image
import io
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


def detect_text_regions(image_array: np.ndarray,
                        min_area: int = 5000,
                        aspect_ratio_range: Tuple[float, float] = (0.1, 10.0)) -> List[Tuple[int, int, int, int]]:
    """
    이미지에서 텍스트 영역 감지

    Args:
        image_array: NumPy 이미지 배열
        min_area: 최소 영역 크기
        aspect_ratio_range: 가로세로 비율 범위

    Returns:
        List of (x, y, w, h) 텍스트 영역 바운딩 박스
    """
    try:
        # 그레이스케일 변환
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array

        # 이진화 (Otsu's method)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 모폴로지 연산 (텍스트 영역 확장)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))  # 가로로 긴 커널
        dilated = cv2.dilate(binary, kernel, iterations=3)

        # 컨투어 찾기
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 텍스트 영역 필터링
        text_regions = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / h if h > 0 else 0

            # 필터 조건
            if (area > min_area and
                aspect_ratio_range[0] < aspect_ratio < aspect_ratio_range[1]):
                text_regions.append((x, y, w, h))

        # 면적 기준 정렬 (큰 것부터)
        text_regions.sort(key=lambda r: r[2] * r[3], reverse=True)

        logger.info(f"텍스트 영역 {len(text_regions)}개 감지")

        return text_regions

    except Exception as e:
        logger.error(f"텍스트 영역 감지 실패: {str(e)}")
        return []


def detect_ingredient_region(image_array: np.ndarray,
                             keywords: List[str] = None,
                             prefer_bottom: bool = True) -> Optional[Tuple[int, int, int, int]]:
    """
    성분표 영역 감지 (키워드 기반)

    전략:
    1. 이미지를 여러 영역으로 분할
    2. 각 영역에서 간단한 OCR 수행
    3. "전성분", "성분:", "INGREDIENTS" 키워드 포함 여부 확인
    4. 해당 영역의 바운딩 박스 반환

    Args:
        image_array: NumPy 이미지 배열
        keywords: 성분표 키워드 리스트
        prefer_bottom: 하단 영역 우선 검사 (성분표는 보통 아래쪽에 위치)

    Returns:
        (x, y, w, h) 성분표 영역 또는 None
    """
    if keywords is None:
        keywords = ['전성분', '성분:', 'INGREDIENTS', 'Ingredients', '화장품법']

    try:
        # 텍스트 영역 감지
        text_regions = detect_text_regions(image_array, min_area=3000)

        if not text_regions:
            logger.warning("텍스트 영역을 찾을 수 없습니다")
            return None

        # 하단 영역 우선: y 좌표 기준으로 정렬 (아래쪽부터)
        if prefer_bottom:
            image_height = image_array.shape[0]
            # y + h/2 (중심점의 y 좌표) 기준으로 내림차순 정렬
            text_regions.sort(key=lambda r: r[1] + r[3]/2, reverse=True)
            logger.info("하단 영역부터 검사합니다 (성분표는 보통 아래쪽에 위치)")

        # EasyOCR로 각 영역 빠르게 스캔
        try:
            import easyocr
            reader = easyocr.Reader(['ko', 'en'], gpu=False, verbose=False)
        except ImportError:
            logger.warning("EasyOCR을 사용할 수 없습니다. 전체 영역을 반환합니다.")
            # EasyOCR 없으면 가장 큰 영역 반환
            return text_regions[0] if text_regions else None

        # 각 영역을 스캔하여 키워드 찾기
        for region in text_regions[:5]:  # 상위 5개 영역만 검사
            x, y, w, h = region

            # 영역 크롭
            cropped = image_array[y:y+h, x:x+w]

            # OCR 수행 (빠른 모드)
            try:
                result = reader.readtext(cropped, detail=0, paragraph=False)
                text = ' '.join(result)

                logger.debug(f"영역 ({x}, {y}, {w}, {h}): {text[:50]}")

                # 키워드 체크
                for keyword in keywords:
                    if keyword in text:
                        logger.info(f"성분표 영역 발견! 키워드: '{keyword}' at ({x}, {y}, {w}, {h})")
                        return region

            except Exception as e:
                logger.debug(f"영역 OCR 실패: {str(e)}")
                continue

        # 키워드를 못 찾았으면 가장 큰 영역 반환 (폴백)
        logger.warning("성분표 키워드를 찾지 못했습니다. 가장 큰 텍스트 영역을 반환합니다.")
        return text_regions[0] if text_regions else None

    except Exception as e:
        logger.error(f"성분표 영역 감지 실패: {str(e)}")
        return None


def crop_ingredient_region(image: Image.Image,
                           auto_detect: bool = True,
                           region: Tuple[int, int, int, int] = None,
                           expand_margin: int = 20) -> Image.Image:
    """
    성분표 영역 크롭

    Args:
        image: PIL Image 객체
        auto_detect: 자동 감지 여부
        region: 수동 지정 영역 (x, y, w, h)
        expand_margin: 크롭 시 여백 (픽셀)

    Returns:
        크롭된 PIL Image
    """
    try:
        # NumPy 배열로 변환
        img_array = np.array(image)

        if auto_detect:
            # 자동 감지
            detected_region = detect_ingredient_region(img_array)

            if detected_region is None:
                logger.warning("성분표 영역 자동 감지 실패 - 원본 이미지 반환")
                return image

            x, y, w, h = detected_region
        else:
            if region is None:
                logger.warning("수동 영역이 지정되지 않았습니다 - 원본 이미지 반환")
                return image
            x, y, w, h = region

        # 여백 추가
        x = max(0, x - expand_margin)
        y = max(0, y - expand_margin)
        w = min(img_array.shape[1] - x, w + 2 * expand_margin)
        h = min(img_array.shape[0] - y, h + 2 * expand_margin)

        # 크롭
        cropped_array = img_array[y:y+h, x:x+w]
        cropped_image = Image.fromarray(cropped_array)

        logger.info(f"성분표 영역 크롭 완료: {w}x{h} (원본: {image.width}x{image.height})")

        return cropped_image

    except Exception as e:
        logger.error(f"성분표 영역 크롭 실패: {str(e)}")
        return image


def visualize_detected_regions(image_array: np.ndarray,
                               save_path: str = 'detected_regions.jpg') -> None:
    """
    감지된 영역 시각화 (디버깅용)

    Args:
        image_array: NumPy 이미지 배열
        save_path: 저장 경로
    """
    try:
        # 텍스트 영역 감지
        regions = detect_text_regions(image_array)

        # 이미지 복사
        vis_img = image_array.copy()

        # 영역 그리기
        for idx, (x, y, w, h) in enumerate(regions[:5]):
            # 바운딩 박스
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 3)

            # 번호 표시
            cv2.putText(vis_img, f"#{idx+1}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 저장
        cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        logger.info(f"시각화 이미지 저장: {save_path}")

    except Exception as e:
        logger.error(f"시각화 실패: {str(e)}")


# 테스트 코드
if __name__ == "__main__":
    import requests
    from io import BytesIO

    logging.basicConfig(level=logging.INFO)

    # 테스트 이미지 URL (예시)
    # test_url = "https://example.com/product_image.jpg"

    print("성분표 영역 감지 테스트")
    print("=" * 60)

    # 사용 예시
    """
    # 1. 이미지 다운로드
    response = requests.get(test_url)
    image = Image.open(BytesIO(response.content))

    # 2. 성분표 영역 크롭
    cropped = crop_ingredient_region(image, auto_detect=True, expand_margin=30)

    # 3. 크롭된 이미지로 OCR 수행
    # ... (OCR 코드)

    print(f"원본 크기: {image.width}x{image.height}")
    print(f"크롭 크기: {cropped.width}x{cropped.height}")
    """

    print("\n✅ 모듈 로드 완료")
    print("\n사용 방법:")
    print("  from ingredient_detector import crop_ingredient_region")
    print("  cropped_image = crop_ingredient_region(image, auto_detect=True)")
