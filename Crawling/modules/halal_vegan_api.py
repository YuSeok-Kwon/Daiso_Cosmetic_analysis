"""
할랄/비건 성분 검증 API 모듈

- 식약처 API 연동 (화장품 성분 검증)
- 외부 할랄/비건 데이터베이스 조회
- 캐싱 기능으로 API 호출 최소화
"""

import requests
import json
import os
from pathlib import Path
from typing import Optional, Dict


class HalalVeganChecker:
    """할랄/비건 성분 검증 클래스"""

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "halal_vegan_cache.json"

        # 캐시 로드
        self.cache = self._load_cache()

        # API 설정 (필요시 환경변수로 관리)
        self.mfds_api_key = os.getenv('MFDS_API_KEY', '')  # 식약처 API 키

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

    def check_ingredient_mfds(self, ingredient: str) -> Optional[Dict]:
        """
        식약처 API로 성분 정보 조회

        Returns:
            dict: {name, cas_no, function, is_safe, ...} or None
        """
        # 캐시 확인
        cache_key = f"mfds_{ingredient}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        if not self.mfds_api_key:
            return None

        try:
            # 식약처 화장품 안전정보 API 호출
            # (실제 API 엔드포인트와 파라미터는 식약처 문서 참고)
            url = "https://openapi.mfds.go.kr/api/cosmetic/ingredients"
            params = {
                'serviceKey': self.mfds_api_key,
                'pageNo': 1,
                'numOfRows': 10,
                'ingredient_name': ingredient,
                'type': 'json'
            }

            response = requests.get(url, params=params, timeout=5)

            if response.status_code == 200:
                data = response.json()

                # API 응답 파싱 (실제 구조에 맞게 수정 필요)
                if data.get('body') and data['body'].get('items'):
                    item = data['body']['items'][0]

                    result = {
                        'name': item.get('ingredient_name'),
                        'cas_no': item.get('cas_no'),
                        'function': item.get('function'),
                        'is_safe': item.get('safety_grade', 'Unknown'),
                        'source': 'mfds_api'
                    }

                    # 캐시 저장
                    self.cache[cache_key] = result
                    self._save_cache()

                    return result

        except Exception as e:
            print(f"식약처 API 호출 실패: {ingredient} - {str(e)}")

        return None

    def check_vegan_status(self, ingredient: str) -> str:
        """
        비건 적합성 확인

        Returns:
            'Yes', 'No', 'Unknown', 'Questionable'
        """
        # 1. 로컬 데이터베이스 체크
        from daiso_beauty_crawler import (
            VEGAN_SAFE_INGREDIENTS,
            ANIMAL_DERIVED_INGREDIENTS
        )

        if ingredient in VEGAN_SAFE_INGREDIENTS:
            return 'Yes'

        if ingredient in ANIMAL_DERIVED_INGREDIENTS:
            return 'No'

        # 2. 식약처 API 체크
        mfds_data = self.check_ingredient_mfds(ingredient)

        if mfds_data:
            function = mfds_data.get('function', '').lower()

            # 동물성 키워드 체크
            animal_keywords = ['동물', '밀크', '콜라겐', '케라틴', '실크', '밀랍', '꿀']
            if any(kw in function for kw in animal_keywords):
                return 'No'

            # 식물성 키워드 체크
            plant_keywords = ['식물', '추출물', '씨앗', '뿌리', '잎', '꽃']
            if any(kw in function for kw in plant_keywords):
                return 'Yes'

        # 3. 외부 API 또는 데이터베이스 연동 (확장 가능)
        # TODO: 외부 비건 인증 데이터베이스 API 연동

        return 'Unknown'

    def check_halal_status(self, ingredient: str) -> str:
        """
        할랄 적합성 확인

        Returns:
            'Yes', 'No', 'Unknown', 'Questionable'
        """
        # 1. 로컬 데이터베이스 체크
        from daiso_beauty_crawler import HARAM_INGREDIENTS, ANIMAL_DERIVED_INGREDIENTS

        if ingredient in HARAM_INGREDIENTS:
            return 'No'

        # 돼지 유래 성분 체크
        pig_keywords = ['돼지', '포신', 'porcine']
        if any(kw in ingredient.lower() for kw in pig_keywords):
            return 'No'

        # 동물성 성분은 원료 확인 필요
        if ingredient in ANIMAL_DERIVED_INGREDIENTS:
            return 'Questionable'

        # 2. 식약처 API 체크
        mfds_data = self.check_ingredient_mfds(ingredient)

        if mfds_data:
            function = mfds_data.get('function', '').lower()

            # 알코올 키워드 체크
            if '알코올' in function or 'alcohol' in function:
                return 'No'

            # 돼지 유래 체크
            if '돼지' in function or 'porcine' in function:
                return 'No'

        return 'Unknown'

    def analyze_product(self, ingredients: list) -> dict:
        """
        제품 전체 성분 분석

        Args:
            ingredients: 성분 리스트

        Returns:
            dict: {
                'is_vegan': bool,
                'is_halal': bool,
                'vegan_issues': list,
                'halal_issues': list,
                'summary': str
            }
        """
        vegan_issues = []
        halal_issues = []

        for ing in ingredients:
            vegan_status = self.check_vegan_status(ing)
            halal_status = self.check_halal_status(ing)

            if vegan_status == 'No':
                vegan_issues.append(f"{ing} (동물성)")
            elif vegan_status == 'Questionable':
                vegan_issues.append(f"{ing} (의심)")

            if halal_status == 'No':
                halal_issues.append(f"{ing} (부적합)")
            elif halal_status == 'Questionable':
                halal_issues.append(f"{ing} (원료확인필요)")

        # 최종 판정
        is_vegan = len(vegan_issues) == 0
        is_halal = len([x for x in halal_issues if '부적합' in x]) == 0

        summary = f"비건: {'적합' if is_vegan else '부적합'}, 할랄: {'적합' if is_halal else '부적합 또는 의심'}"

        return {
            'is_vegan': is_vegan,
            'is_halal': is_halal,
            'vegan_issues': vegan_issues,
            'halal_issues': halal_issues,
            'summary': summary
        }


# 사용 예시
if __name__ == "__main__":
    checker = HalalVeganChecker()

    # 테스트 성분
    test_ingredients = [
        '글리세린',
        '알로에베라잎추출물',
        '밀랍',
        '에탄올',
        '토코페롤'
    ]

    print("할랄/비건 성분 분석 테스트")
    print("=" * 60)

    for ing in test_ingredients:
        vegan = checker.check_vegan_status(ing)
        halal = checker.check_halal_status(ing)

        print(f"\n성분: {ing}")
        print(f"  비건: {vegan}")
        print(f"  할랄: {halal}")

    # 제품 전체 분석
    print("\n" + "=" * 60)
    print("제품 전체 분석:")
    result = checker.analyze_product(test_ingredients)
    print(json.dumps(result, ensure_ascii=False, indent=2))
