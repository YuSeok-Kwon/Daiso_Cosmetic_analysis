"""
Naver Clova OCR API 연동 모듈
- 한국어 OCR 특화
- 화장품 성분표 인식 최적화
"""

import requests
import json
import os
import uuid
import time
import logging
from typing import Optional, Dict, List
from io import BytesIO
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

logger = logging.getLogger(__name__)


class ClovaOCR:
    """Naver Clova OCR 클라이언트"""

    def __init__(self, api_url: str = None, secret_key: str = None):
        """
        Args:
            api_url: Clova OCR API URL (환경변수 CLOVA_OCR_URL 또는 직접 지정)
            secret_key: Clova OCR Secret Key (환경변수 CLOVA_OCR_SECRET 또는 직접 지정)
        """
        self.api_url = api_url or os.getenv('CLOVA_OCR_URL')
        self.secret_key = secret_key or os.getenv('CLOVA_OCR_SECRET')

        if not self.api_url or not self.secret_key:
            logger.warning("Clova OCR API 키가 설정되지 않았습니다. 환경변수를 확인하세요.")
            logger.warning("CLOVA_OCR_URL, CLOVA_OCR_SECRET")

    def is_available(self) -> bool:
        """API 사용 가능 여부 확인"""
        return bool(self.api_url and self.secret_key)

    def extract_text_from_url(self, image_url: str) -> Optional[str]:
        """
        이미지 URL에서 텍스트 추출

        Args:
            image_url: 이미지 URL

        Returns:
            추출된 텍스트 (실패 시 None)
        """
        if not self.is_available():
            logger.warning("Clova OCR을 사용할 수 없습니다")
            return None

        try:
            # 요청 JSON 생성
            request_json = {
                'images': [
                    {
                        'format': 'jpg',  # jpg, png 등
                        'name': 'ingredient_image',
                        'url': image_url
                    }
                ],
                'requestId': str(uuid.uuid4()),
                'version': 'V2',
                'timestamp': int(round(time.time() * 1000))
            }

            headers = {
                'X-OCR-SECRET': self.secret_key,
                'Content-Type': 'application/json'
            }

            # API 호출
            logger.info(f"Clova OCR 호출: {image_url[:80]}...")
            response = requests.post(
                self.api_url,
                headers=headers,
                json=request_json,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                text = self._parse_response(result)
                logger.info(f"Clova OCR 성공: {len(text)}자 추출")
                return text
            else:
                logger.error(f"Clova OCR 실패: {response.status_code}")
                logger.error(f"응답: {response.text}")
                return None

        except Exception as e:
            logger.error(f"Clova OCR 오류: {str(e)}")
            return None

    def extract_text_from_bytes(self, image_bytes: bytes, image_format: str = 'png') -> Optional[str]:
        """
        이미지 바이트에서 텍스트 추출

        Args:
            image_bytes: 이미지 바이트 데이터
            image_format: 이미지 포맷 (jpg, png 등)

        Returns:
            추출된 텍스트 (실패 시 None)
        """
        if not self.is_available():
            logger.warning("Clova OCR을 사용할 수 없습니다")
            return None

        try:
            # multipart/form-data로 전송
            request_json = {
                'images': [
                    {
                        'format': image_format,
                        'name': 'ingredient_image'
                    }
                ],
                'requestId': str(uuid.uuid4()),
                'version': 'V2',
                'timestamp': int(round(time.time() * 1000))
            }

            # 파일 데이터 준비
            files = {
                'message': (None, json.dumps(request_json), 'application/json'),
                'file': ('image.' + image_format, BytesIO(image_bytes), 'image/' + image_format)
            }

            headers = {
                'X-OCR-SECRET': self.secret_key
            }

            # API 호출
            logger.info("Clova OCR 호출 (이미지 바이트)")
            response = requests.post(
                self.api_url,
                headers=headers,
                files=files,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                text = self._parse_response(result)
                logger.info(f"Clova OCR 성공: {len(text)}자 추출")
                return text
            else:
                logger.error(f"Clova OCR 실패: {response.status_code}")
                logger.error(f"응답: {response.text}")
                return None

        except Exception as e:
            logger.error(f"Clova OCR 오류: {str(e)}")
            return None

    def _parse_response(self, response_json: Dict) -> str:
        """
        Clova OCR 응답 파싱

        Args:
            response_json: API 응답 JSON

        Returns:
            추출된 텍스트 (줄바꿈 포함)
        """
        try:
            images = response_json.get('images', [])
            if not images:
                return ''

            # 첫 번째 이미지 결과
            fields = images[0].get('fields', [])

            # 각 필드의 텍스트를 줄바꿈으로 연결
            # fields는 이미 좌표 순서대로 정렬되어 있음
            texts = []

            for field in fields:
                infer_text = field.get('inferText', '')
                if infer_text:
                    texts.append(infer_text)

            # 줄바꿈으로 연결 (성분 리스트는 보통 한 줄에 하나씩)
            full_text = '\n'.join(texts)

            return full_text

        except Exception as e:
            logger.error(f"Clova OCR 응답 파싱 오류: {str(e)}")
            return ''

    def extract_structured_data(self, image_url: str) -> Optional[Dict]:
        """
        구조화된 데이터 추출 (위치 정보 포함)

        Args:
            image_url: 이미지 URL

        Returns:
            {
                'full_text': '전체 텍스트',
                'fields': [
                    {'text': '성분명', 'confidence': 0.95, 'bounding_box': {...}},
                    ...
                ]
            }
        """
        if not self.is_available():
            return None

        try:
            request_json = {
                'images': [
                    {
                        'format': 'jpg',
                        'name': 'ingredient_image',
                        'url': image_url
                    }
                ],
                'requestId': str(uuid.uuid4()),
                'version': 'V2',
                'timestamp': int(round(time.time() * 1000))
            }

            headers = {
                'X-OCR-SECRET': self.secret_key,
                'Content-Type': 'application/json'
            }

            response = requests.post(
                self.api_url,
                headers=headers,
                json=request_json,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                images = result.get('images', [])

                if not images:
                    return None

                fields_data = images[0].get('fields', [])

                # 구조화된 데이터 추출
                structured_fields = []
                all_texts = []

                for field in fields_data:
                    text = field.get('inferText', '')
                    confidence = field.get('inferConfidence', 0)

                    # bounding box 정보
                    vertices = field.get('boundingPoly', {}).get('vertices', [])

                    if text:
                        all_texts.append(text)
                        structured_fields.append({
                            'text': text,
                            'confidence': confidence,
                            'bounding_box': vertices
                        })

                return {
                    'full_text': '\n'.join(all_texts),
                    'fields': structured_fields,
                    'field_count': len(structured_fields)
                }

            return None

        except Exception as e:
            logger.error(f"구조화된 데이터 추출 오류: {str(e)}")
            return None


# 싱글톤 인스턴스
_clova_client = None


def get_clova_client() -> ClovaOCR:
    """Clova OCR 클라이언트 싱글톤"""
    global _clova_client
    if _clova_client is None:
        _clova_client = ClovaOCR()
    return _clova_client


# 사용 예시
if __name__ == "__main__":
    # 환경변수 설정 예시
    print("Naver Clova OCR 테스트")
    print("=" * 60)

    # 방법 1: 환경변수로 설정
    # export CLOVA_OCR_URL="https://********.apigw.ntruss.com/custom/v1/****/infer"
    # export CLOVA_OCR_SECRET="****************"

    # 방법 2: 직접 지정
    # client = ClovaOCR(
    #     api_url="https://********.apigw.ntruss.com/custom/v1/****/infer",
    #     secret_key="****************"
    # )

    client = get_clova_client()

    if not client.is_available():
        print("\n⚠️  Clova OCR API 키가 설정되지 않았습니다!")
        print("\n다음 명령어로 환경변수를 설정하세요:")
        print("\nexport CLOVA_OCR_URL='https://********.apigw.ntruss.com/custom/v1/****/infer'")
        print("export CLOVA_OCR_SECRET='****************'")
        print("\n또는 .env 파일에 추가:")
        print("CLOVA_OCR_URL=https://********.apigw.ntruss.com/custom/v1/****/infer")
        print("CLOVA_OCR_SECRET=****************")
    else:
        print("✅ Clova OCR API 설정 완료")
        print(f"API URL: {client.api_url[:50]}...")

        # 테스트 이미지 URL (예시)
        # test_url = "https://example.com/ingredient_image.jpg"
        # text = client.extract_text_from_url(test_url)
        # print(f"\n추출된 텍스트:\n{text}")
