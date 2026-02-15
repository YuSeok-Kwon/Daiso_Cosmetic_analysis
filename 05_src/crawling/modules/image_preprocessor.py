"""
이미지 전처리 모듈 - OCR 정확도 향상을 위한 이미지 처리
"""
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io


def preprocess_image_for_ocr(image_bytes, method='enhanced'):
    """
    OCR 정확도를 높이기 위한 이미지 전처리

    Args:
        image_bytes: 원본 이미지 바이트
        method: 전처리 방법
            - 'basic': 기본 전처리 (그레이스케일)
            - 'enhanced': 향상된 전처리 (대비, 샤프닝)
            - 'aggressive': 공격적 전처리 (이진화, 노이즈 제거)

    Returns:
        처리된 이미지 바이트
    """
    # PIL 이미지로 변환
    img = Image.open(io.BytesIO(image_bytes))

    # RGB로 변환 (RGBA 등 다른 모드 처리)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    if method == 'basic':
        # 기본: 그레이스케일만
        img = img.convert('L')

    elif method == 'enhanced':
        # 향상: 대비 + 샤프닝
        # 1. 대비 향상 (1.5배)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)

        # 2. 선명도 향상
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.3)

        # 3. 그레이스케일
        img = img.convert('L')

    elif method == 'aggressive':
        # 공격적: 이진화 + 노이즈 제거
        # 1. 그레이스케일 변환
        img = img.convert('L')

        # 2. NumPy 배열로 변환
        img_array = np.array(img)

        # 3. 대비 향상 (CLAHE - Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_array = clahe.apply(img_array)

        # 4. 노이즈 제거 (가우시안 블러)
        img_array = cv2.GaussianBlur(img_array, (3, 3), 0)

        # 5. 적응형 이진화 (Adaptive Thresholding)
        img_array = cv2.adaptiveThreshold(
            img_array, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # 6. PIL 이미지로 다시 변환
        img = Image.fromarray(img_array)

    # 바이트로 변환
    output = io.BytesIO()
    img.save(output, format='PNG', optimize=True)
    return output.getvalue()


def upscale_image(image_bytes, scale_factor=2.0):
    """
    이미지 해상도 향상

    Args:
        image_bytes: 원본 이미지 바이트
        scale_factor: 확대 비율 (기본 2배)

    Returns:
        확대된 이미지 바이트
    """
    img = Image.open(io.BytesIO(image_bytes))

    # 새 크기 계산
    new_width = int(img.width * scale_factor)
    new_height = int(img.height * scale_factor)

    # 고품질 리샘플링으로 확대
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 바이트로 변환
    output = io.BytesIO()
    img.save(output, format='PNG', optimize=True)
    return output.getvalue()


def preprocess_for_korean_ocr(image_bytes):
    """
    한국어 OCR에 최적화된 전처리
    - 대비 향상
    - 선명도 향상
    - 약간의 확대
    """
    img = Image.open(io.BytesIO(image_bytes))

    # RGB 변환
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # 1. 약간 확대 (1.5배) - 작은 글자 개선
    if img.width < 1500:
        scale = 1.5
        new_width = int(img.width * scale)
        new_height = int(img.height * scale)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 2. 대비 향상 (한국어는 대비가 중요)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.6)

    # 3. 선명도 향상
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.4)

    # 4. 밝기 약간 조정 (너무 어두운 이미지 개선)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.1)

    # 바이트로 변환
    output = io.BytesIO()
    img.save(output, format='PNG', optimize=True, quality=95)
    return output.getvalue()
