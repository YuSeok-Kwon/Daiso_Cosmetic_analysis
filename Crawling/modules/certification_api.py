"""
공공데이터 인증 API 모듈
- 농식품 해외인증종류 API (할랄, 비건 등)
- 캐싱 기능으로 API 호출 최소화
"""

import os
import json
import requests
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta


class CertificationAPIClient:
    """농식품 해외인증 API 클라이언트"""

    def __init__(self, api_key: str = None, cache_dir: str = "cache"):
        """
        Args:
            api_key: 공공데이터포털 API 키 (None이면 환경변수에서 읽음)
            cache_dir: 캐시 디렉토리
        """
        self.api_key = api_key or os.getenv('FOOD_CERTIFICATION_API_KEY', '')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "certification_cache.json"

        # API 엔드포인트 (실제 URL은 API 문서 확인 후 수정 필요)
        self.base_url = "https://api.odcloud.kr/api/15102890/v1/uddi:b7f89484-470a-46de-a824-a6bf14f088eb"

        # 캐시 로드
        self.cache = self._load_cache()

        # 캐시 유효 기간 (일)
        self.cache_ttl_days = 30

    def _load_cache(self) -> dict:
        """캐시 파일 로드"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_cache(self):
        """캐시 파일 저장"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _is_cache_valid(self, cache_entry: dict) -> bool:
        """캐시 유효성 확인"""
        if 'timestamp' not in cache_entry:
            return False

        cached_time = datetime.fromisoformat(cache_entry['timestamp'])
        expired_time = datetime.now() - timedelta(days=self.cache_ttl_days)

        return cached_time > expired_time

    def get_certifications(self, page: int = 1, per_page: int = 100) -> Optional[List[Dict]]:
        """
        인증 목록 조회

        Args:
            page: 페이지 번호
            per_page: 페이지당 결과 수

        Returns:
            list: 인증 정보 리스트 또는 None
        """
        if not self.api_key:
            print("경고: API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")
            return None

        # 캐시 확인
        cache_key = f"certifications_page_{page}"
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            print(f"캐시에서 인증 목록 로드 (페이지 {page})")
            return self.cache[cache_key]['data']

        try:
            # API 호출
            params = {
                'serviceKey': self.api_key,
                'page': page,
                'perPage': per_page,
                'returnType': 'json'
            }

            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            # 응답 파싱 (실제 구조에 맞게 수정 필요)
            if 'data' in data:
                certifications = data['data']

                # 캐시 저장
                self.cache[cache_key] = {
                    'data': certifications,
                    'timestamp': datetime.now().isoformat()
                }
                self._save_cache()

                print(f"API에서 인증 목록 조회 완료 (페이지 {page}): {len(certifications)}개")
                return certifications
            else:
                print(f"API 응답 형식 오류: {data}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"API 호출 실패: {str(e)}")
            return None
        except Exception as e:
            print(f"예외 발생: {str(e)}")
            return None

    def search_halal_certifications(self) -> Optional[List[Dict]]:
        """
        할랄 인증 검색

        Returns:
            list: 할랄 인증 정보 리스트
        """
        cache_key = "halal_certifications"
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            print("캐시에서 할랄 인증 목록 로드")
            return self.cache[cache_key]['data']

        all_certs = self.get_certifications(page=1, per_page=1000)

        if all_certs:
            # 할랄 인증만 필터링
            halal_certs = [
                cert for cert in all_certs
                if any(keyword in str(cert).lower() for keyword in ['halal', '할랄'])
            ]

            # 캐시 저장
            self.cache[cache_key] = {
                'data': halal_certs,
                'timestamp': datetime.now().isoformat()
            }
            self._save_cache()

            print(f"할랄 인증 검색 완료: {len(halal_certs)}개")
            return halal_certs

        return None

    def search_vegan_certifications(self) -> Optional[List[Dict]]:
        """
        비건 인증 검색

        Returns:
            list: 비건 인증 정보 리스트
        """
        cache_key = "vegan_certifications"
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            print("캐시에서 비건 인증 목록 로드")
            return self.cache[cache_key]['data']

        all_certs = self.get_certifications(page=1, per_page=1000)

        if all_certs:
            # 비건 인증만 필터링
            vegan_certs = [
                cert for cert in all_certs
                if any(keyword in str(cert).lower() for keyword in ['vegan', '비건', 'vegetarian'])
            ]

            # 캐시 저장
            self.cache[cache_key] = {
                'data': vegan_certs,
                'timestamp': datetime.now().isoformat()
            }
            self._save_cache()

            print(f"비건 인증 검색 완료: {len(vegan_certs)}개")
            return vegan_certs

        return None

    def check_product_certification(self, product_name: str, certification_type: str = 'halal') -> Dict:
        """
        제품의 인증 여부 확인 (제품명 기반 매칭)

        Args:
            product_name: 제품명
            certification_type: 'halal' 또는 'vegan'

        Returns:
            dict: {
                'is_certified': bool,
                'certification_info': dict or None,
                'source': 'api' or 'cache'
            }
        """
        if certification_type == 'halal':
            certs = self.search_halal_certifications()
        elif certification_type == 'vegan':
            certs = self.search_vegan_certifications()
        else:
            return {'is_certified': False, 'certification_info': None, 'source': None}

        if not certs:
            return {'is_certified': False, 'certification_info': None, 'source': 'api'}

        # 제품명으로 매칭 시도 (간단한 부분 문자열 매칭)
        for cert in certs:
            # 실제 필드명은 API 응답 구조에 맞게 수정 필요
            cert_product_name = cert.get('product_name', cert.get('name', ''))

            if product_name.lower() in cert_product_name.lower() or \
               cert_product_name.lower() in product_name.lower():
                return {
                    'is_certified': True,
                    'certification_info': cert,
                    'source': 'cache' if 'halal_certifications' in self.cache else 'api'
                }

        return {'is_certified': False, 'certification_info': None, 'source': 'api'}


# 사용 예시
if __name__ == "__main__":
    # .env 파일에서 API 키 로드
    from dotenv import load_dotenv
    load_dotenv()

    client = CertificationAPIClient()

    print("=" * 60)
    print("공공데이터 인증 API 테스트")
    print("=" * 60)

    # 할랄 인증 검색
    print("\n[1] 할랄 인증 검색")
    halal_certs = client.search_halal_certifications()
    if halal_certs:
        print(f"할랄 인증 {len(halal_certs)}개 발견")
        if len(halal_certs) > 0:
            print(f"예시: {halal_certs[0]}")
    else:
        print("API 키가 없거나 응답 없음")

    # 비건 인증 검색
    print("\n[2] 비건 인증 검색")
    vegan_certs = client.search_vegan_certifications()
    if vegan_certs:
        print(f"비건 인증 {len(vegan_certs)}개 발견")
        if len(vegan_certs) > 0:
            print(f"예시: {vegan_certs[0]}")
    else:
        print("API 키가 없거나 응답 없음")

    # 제품 인증 확인
    print("\n[3] 제품 인증 확인 테스트")
    test_product = "다이소 립스틱"
    result = client.check_product_certification(test_product, 'halal')
    print(f"제품: {test_product}")
    print(f"할랄 인증 여부: {result['is_certified']}")
    if result['certification_info']:
        print(f"인증 정보: {result['certification_info']}")
