"""
OpenAI Batch API를 사용한 대량 ABSA 라벨링
- 비용 50% 절감 (GPT-4o 기준)
- 24시간 내 처리
"""
import json
import time
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# .env 파일 로드
load_dotenv(Path(__file__).parent / ".env")


class BatchLabeler:
    """OpenAI Batch API를 사용한 대량 라벨링"""

    def __init__(self, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.data_dir = Path(__file__).parent / "data"
        self.batch_dir = self.data_dir / "batch"
        self.batch_dir.mkdir(parents=True, exist_ok=True)

    def _build_prompt(self, row: pd.Series) -> str:
        """프롬프트 생성 (openai_client.py와 동일)"""
        # 제품 정보 구성
        if row.get('name'):
            if row.get('category_2'):
                product_info = f"{row['name']} (카테고리: {row['category_2']})"
            else:
                product_info = row['name']
        else:
            product_info = f"제품코드 {row['product_code']}"

        prompt = f"""당신은 한국어 화장품 리뷰를 분석하는 '다이소 뷰티 특화' ABSA 전문가입니다.
초저가 화장품 시장과 고객의 다이소 소비 패턴(가성비, 듀프, 품절 대란 등)의 문맥을 파악해야 합니다.

**[리뷰 정보]**
- 제품: {product_info}
- 평점: {row['rating']}/5
- 리뷰: "{row['text']}"

**[핵심 분석 규칙]**
1. **확실한 Aspect만 추출**: confidence ≥ 0.7인 Aspect만 추출하세요. 억지로 추출하지 마세요.
2. **재구매 과잉 추론 금지**: "좋아요", "맘에 들어요" 같은 만족 표현만으로 재구매를 추론하지 마세요. 재구매는 "또 살", "재구매", "N번째 구매" 같은 **명시적 키워드**가 있을 때만 추출합니다.
3. **미분류 처리**: 확실한 Aspect가 없으면 "미분류" Aspect + neutral sentiment로 반환
4. **혼합 감성 분리**: "내용물은 좋은데 용기가 샌다"처럼 역접(근데, 하지만)이 있으면 각 Aspect를 분리 추출
5. **별점-텍스트 불일치**: "구하기 힘들어서 짜증 (5점)"은 품절 대란으로 인한 아쉬움이지만 CS/응대가 아님 → 미분류 neutral
6. **재구매 시그널**: 아래 키워드가 **긍정적 문맥**에서 사용될 때만 재구매 positive (confidence ≥ 0.9)
   - "쟁여둠, 또 삼, 정착, 보이면 무조건, N번째 구매, 재구매했어요"
   - 단, "정착할랬는데 안맞아" 같은 부정 문맥이면 추출 금지
7. **미사용 리뷰**: 사용 전/기대감만 있으면 성능/품질/재질/재구매 Aspect 추출 금지

**[⚠️ 평점(Rating)을 활용한 Sentiment 판단]**
- 평점은 sentiment 판단의 **보조 지표**로 활용하세요 (텍스트 내용이 최우선)
- **평점 기준**:
  - 1~2점: negative 가능성 높음
  - 3점: neutral 가능성 높음
  - 4~5점: positive 가능성 높음
- **적용 원칙**:
  - 텍스트가 모호하거나 짧을 때 평점을 참고하여 sentiment 결정
  - 예: "그냥 그래요" (3점) → neutral / "그냥 그래요" (1점) → negative
  - 예: "괜찮아요" (5점) → positive / "괜찮아요" (2점) → neutral~negative
- **주의**: 텍스트가 명확한 감성을 표현하면 평점과 불일치해도 텍스트 우선

**[⚠️ 중요: Aspect 혼동 방지 규칙]**

**1. 색상/발색 vs 사용감/성능 (가장 많이 혼동됨!)**
- **색상/발색**: "발색, 색, 컬러, 톤, 쿨톤, 웜톤, 예쁜색, 이쁜색, 착색" 키워드가 있으면 무조건 색상/발색
  - O: "발색이 진해요", "색이 예뻐요", "발색이 별로", "색이 탁해요", "발색 안됨"
  - O: "광고 그렇게 이쁜거같지않은데" → 색상에 대한 불만 → 색상/발색
  - X: "색상 잘못왔다고 문의" → 오배송 → **배송/포장** (색상 불만 아님)
  - X: "화끈화끈해요... 색상은 누디합니다" → 화끈함=**사용감/성능**, 색상 언급은 부수적
- **사용감/성능**: 발림성, 지속력, 커버력, 보습력, 효과 등 '사용 경험'
  - O: "발림 좋아요", "커버력 별로", "보습력 최고", "효과 있어요"
- **핵심**: "발색" ≠ "발림". 발색=색상, 발림=사용감
- **[색상/발색 sentiment 판단 — neutral 남발 금지]**
  - **결론 우선 원칙**: 혼합 감성("A는 좋은데 B는 아쉬워요")에서 **리뷰 결론/주된 감정**으로 sentiment 결정
    - "발색 아주 마음에 들어요. 다만 진한 편이라 쿨톤에만 적용 가능" → 결론은 만족 → **positive**
    - "발색이 연하고 묽어요 여러번 칠해야해요" → 불만 → **negative**
    - "웜톤립이라고는 그닥..." → "그닥"은 부정 → **negative**
  - **시간 흐름 주의**: "처음엔 예쁜데 점점 연해져서" → 최종 결과가 부정 → **negative**
  - **"생각보다 진해요"는 모호**: 맥락에 따라 판단. 좋다는 표현 없으면 neutral, 불편하다면 negative

**2. 배송/포장 vs 품질/퀄리티 (배송 중 문제 구분)**
- **배송/포장**: "왔어요, 도착, 받았는데, 배송" + 파손/누락/오배송
  - O: "부러져서 왔어요" → 배송 중 파손 → 배송/포장
  - O: "빈 상자만 왔음" → 배송 누락 → 배송/포장
  - O: "매트로 주문했는데 촉촉으로 옴" → 오배송 → 배송/포장
  - O: "깨지지 않고 왔어요" → 배송 상태 만족 → 배송/포장
- **품질/퀄리티**: 제품 자체의 제조 결함 (배송과 무관)
  - O: "처음 열었는데 불량", "유통기한 임박", "정품 의심"

**3. 품질/퀄리티 vs 사용감/성능**
- **품질/퀄리티**: 물리적 결함 + "퀄리티" 키워드 포함
  - O: "퀄리티 좋아요", "퀄리티 별로", "퀄이 좋다" → 품질/퀄리티
  - O: "불량", "깨짐", "변질", "유통기한" → 품질/퀄리티
- **사용감/성능**: 사용 후 느끼는 경험과 효과
  - O: "좋아요", "별로에요", "만족", "효과 없음" (퀄리티 언급 없이) → 사용감/성능

**4. 재구매 (가장 엄격한 기준 적용 — "이 제품"의 명시적 재구매 의사/행동만)**
- **[O 재구매 추출]** 명시적 키워드가 있을 때만:
  - "재구매", "재주문", "또 살", "또 삼", "더 살", "쟁여둠", "쟁여놓" → 재구매 추출
  - "N번째 구매", "두개째", "세통째", "몇개째", "다 떨어져서 구매" → 반복 구매 사실
  - "정착템", "정착했다", "앞으로 이것만" → 명시적 정착
  - "1년째 사용", "꾸준히 구매" → 명시적 지속 구매
- **[X 재구매 추출 금지]** 아래는 절대 재구매로 분류하지 마세요:
  - 현재 만족: "좋아요", "맘에 들어요", "잘 쓰고 있어요", "최고" → 사용감/성능
  - 사용 의지: "써봐야겠어요", "써야겠어요", "애용할것같아요" → 사용감/성능
  - 첫 구매: "사봤어요", "구매했어요", "바로 구매", "대용량구매" → 다른 Aspect
  - 브랜드 칭찬: "다이소 없인 못살아", "항상 애용하는 다이소" → 가격/가성비
  - 추천: "추천합니다", "장바구니에 담으세요" → 사용감/성능 또는 미분류
  - 미사용: "사용못해봤지만 재구매하려" → 미분류
- **[다른 색상/종류]** "다른 컬러로 살 것", "다른 종류로 사보려구요" → 이 제품은 재구매 X
- **[조건부 재구매]** "효과 있으면 재구매", "써보고 재구매" → 재구매 **neutral** (positive 아님)
- **[부정 문맥]** "정착할랬는데 안 맞아", "쟁여둬야 할것같은데 고생" → 재구매 키워드가 있어도 전후 문맥이 부정이면 추출 금지

**[카테고리별 판단 규칙]**

[스킨케어]
- 리들샷/스피큘: "따갑다, 찌릿" → 사용감/성능 neutral (정상 반응)
- 기초케어: "순하다, 자극없다" → 사용감/성능 positive / "트러블, 좁쌀" → 사용감/성능 negative
- 보습크림: "끈적임" → 재질/냄새 neutral / "촉촉" → 사용감/성능 positive

[메이크업]
- 립제품: "볼에 발라요" (본래 용도 실패) → 사용감/성능 negative
- 베이스: "다크닝, 잿빛" → 색상/발색 negative / "톤업" → 색상/발색 positive
- 립틴트: "기승전핑크" → 색상/발색 negative

[다이소 특화]
- 듀프 비교: "올리브영 XX랑 똑같다" → 가격/가성비 positive
- 소용량: "여행용 딱" → 용량/휴대 positive / "금방 다 씀" → 용량/휴대 negative
- 용기 결함: "펌핑 고장, 샌다, 뚜껑 안 닫혀" → 디자인 negative

[CS/응대 판단 기준 — 엄격 적용]
- **[O CS/응대]** 직원/고객센터와의 **직접 상호작용**이 있을 때만:
  - 직원 친절/불친절: "직원분이 친절하게 설명해주셨어요", "직원이 불친절해요"
  - 교환/환불 **처리 경험**: "환불 받았어요", "교환해줬는데 친절했어요"
  - 고객센터 응대: "고객센터에 문의했는데 답변이 늦어요"
- **[X CS/응대 아님]** 아래는 CS/응대로 분류하지 마세요:
  - 품절/재입고 단순 언급: "품절이라 못 샀어요", "재입고 됐어요", "힘들게 구했어요" → **미분류** 또는 다른 Aspect
  - 매장 재고만 언급: "매장에 없어서 온라인으로 샀어요", "오프라인 품절" → **미분류**
  - 배송 파손/오배송: "깨져서 왔어요", "다른 제품이 왔어요" → **배송/포장**
  - 교환/환불 **요청만** (처리 경험 없음): "교환해주나요?", "환불 가능한가요?" → **배송/포장** negative
  - 불량 제보: "불량이네요 어떻게 해야하나" → **품질/퀄리티** negative

[재질/냄새 판단 기준 — reason에 제형+향 모두 포함 필수]
- **제형(텍스처)**: 묽다, 되다, 크리미, 젤타입, 오일리, 점성, 발림성, 유분감, 수분감, 쫀쫀, 뻑뻑, 두껍다, 얇다 → 재질/냄새
- **향/냄새**: 향이 좋다, 냄새가 이상하다, 무향 → 재질/냄새
- **끈적임**: 토너/에센스의 끈적임 → 재질/냄새 / 크림류는 사용감/성능과 함께 판단
- **⚠️ reason 작성 규칙**: 하나의 리뷰에 제형+향이 모두 언급되면, reason에 **둘 다** 포함하세요.
  - O: "제형이 무겁지않고 흡수 빠르고 향도 은은해서" → reason: "제형이 가볍고 향이 은은함" (제형+향 모두)
  - X: "향이 은은하다" (제형 누락 ❌)
- **[X 재질/냄새 아님]** 아래는 다른 Aspect로 분류:
  - 자극감: "화한 느낌", "따끔", "쏘는 느낌" → **사용감/성능** (피부 자극은 텍스처가 아님)
  - 내용물 불량: "내용물이 떡져있다", "굳어있다", "분리됨" → **품질/퀄리티** (제조 결함)
  - 구체성 부족: "좋아요내춰향에맞네요" 같은 지나치게 일반적 표현 → **사용감/성능** 또는 **미분류**
- **[재질/냄새 sentiment — 단순 설명 ≠ 감정]**
  - "복숭아향이 나요~", "허브향이에요" → **neutral** (향의 종류를 설명할 뿐 좋고 나쁨 판단 없음)
  - "무향입니다", "향 없어요" → **neutral** (정보 제공)
  - "향이 좋아요", "향이 달달해서 좋아요" → **positive** (명확한 긍정 감정)
  - "향이 별로", "냄새가 역해요" → **negative** (명확한 부정 감정)

[디자인 판단 기준 — 패키지/용기의 구체적 언급 필수]
- **[O 디자인]** 패키지/용기/구조물이 **명시적으로** 언급될 때만:
  - 패키지 외관: "케이스가 예뻐요", "패키지 디자인이 귀여워요", "용기가 고급스러워요"
  - 용기 구조: "펌프가 고장났어요", "뚜껑이 잘 안 닫혀요", "브러쉬가 부드러워요"
  - 어플리케이터: "팁이 뭉툭해요", "어플리케이터가 사용하기 편해요"
- **[X 디자인 아님]** 아래는 디자인으로 분류하지 마세요:
  - "예뻐요", "이쁘다", "귀엽다" **단독**: 패키지/용기 언급 없이 외형만 칭찬 → **사용감/성능** (제품 자체 만족) 또는 **색상/발색** (색이 예쁘다는 뜻일 때)
  - 내구성 결함: "부서졌어요", "깨졌어요", "터졌어요", "주사기가 부러졌어요" → **품질/퀄리티** (제조/내구성 결함)
  - 배송 파손: "배송 중 깨져서 왔어요", "없이 왔어요" → **배송/포장**
- **[디자인 sentiment]**
  - 용기 구조 불만 (펌프 고장, 뚜껑 안 닫힘, 샘) → **negative**
  - 패키지 외관 칭찬 → **positive**

**[Aspect 목록 (11개)]**
1. 배송/포장: 배송 속도, 택배 포장, 배송 중 파손
2. 품질/퀄리티: 제품 물리적 결함만 (깨짐, 부러짐, 변질, 유통기한, 불량), "퀄리티" 키워드
3. 가격/가성비: 가격 적정성, 가성비, 듀프 비교
4. 사용감/성능: 사용 경험, 효과, 만족도 (좋아요/별로/만족/최고 포함)
5. 용량/휴대: 용량 크기, 휴대 편의성
6. 디자인: 패키지/용기/구조물이 **구체적으로** 언급될 때만. "예뻐요" 단독은 디자인 아님. 내구성 결함(부서짐, 깨짐)은 품질/퀄리티
7. 재질/냄새: 텍스처(제형, 발림성, 점성, 유분감) + 향기/냄새. reason에 제형과 향 모두 언급 시 둘 다 포함 필수. 단순 향 설명("~향이에요")은 neutral
8. CS/응대: 직원/고객센터와의 **직접 상호작용만** (교환/환불 처리 경험, 직원 친절/불친절, 고객센터 응대). 품절/재고 단순 언급은 CS 아님
9. 재구매: **이 제품**의 명시적 재구매 의사/행동만 (재구매, 또 살, 쟁여둠, 정착, N번째 구매). 만족 표현은 재구매 아님
10. 색상/발색: 색상 구현, 발색, 피부톤 적합성 (발색, 색, 컬러 키워드)
11. 미분류: 위 10개에 해당하지 않거나 confidence < 0.7인 경우 → sentiment는 neutral

**[출력 형식]**
{{
  "sentiment": "positive|neutral|negative",
  "sentiment_score": -1.0 ~ 1.0,
  "aspect_labels": [
    {{"aspect": "Aspect명", "sentiment": "positive|neutral|negative", "confidence": 0.0~1.0, "reason": "근거 문장"}}
  ],
  "evidence": "핵심 근거 원문 인용",
  "summary": "30자 이내 요약"
}}

반드시 유효한 JSON만 반환하세요. 추가 설명 없이 순수 JSON만 출력하세요."""

        return prompt

    def create_batch_file(self, df: pd.DataFrame, batch_name: str = None) -> str:
        """Batch API용 JSONL 파일 생성"""
        if batch_name is None:
            batch_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        jsonl_path = self.batch_dir / f"batch_input_{batch_name}.jsonl"

        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for idx, row in df.iterrows():
                request = {
                    "custom_id": f"review_{idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "messages": [
                            {"role": "user", "content": self._build_prompt(row)}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 500,
                        "response_format": {"type": "json_object"}
                    }
                }
                f.write(json.dumps(request, ensure_ascii=False) + "\n")

        print(f"JSONL 파일 생성: {jsonl_path}")
        print(f"총 요청 수: {len(df)}개")
        return str(jsonl_path)

    def upload_batch_file(self, jsonl_path: str) -> str:
        """파일 업로드"""
        with open(jsonl_path, 'rb') as f:
            file = self.client.files.create(file=f, purpose="batch")
        print(f"파일 업로드 완료: {file.id}")
        return file.id

    def create_batch(self, file_id: str, description: str = None) -> str:
        """Batch 생성"""
        batch = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": description or "ABSA labeling"}
        )
        print(f"Batch 생성 완료: {batch.id}")
        print(f"상태: {batch.status}")
        return batch.id

    def check_batch_status(self, batch_id: str) -> dict:
        """Batch 상태 확인"""
        batch = self.client.batches.retrieve(batch_id)
        return {
            "id": batch.id,
            "status": batch.status,
            "created_at": batch.created_at,
            "completed_at": batch.completed_at,
            "request_counts": batch.request_counts,
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id
        }

    def wait_for_completion(self, batch_id: str, check_interval: int = 60) -> dict:
        """완료 대기"""
        print(f"Batch 완료 대기 중... (ID: {batch_id})")
        while True:
            status = self.check_batch_status(batch_id)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 상태: {status['status']}, "
                  f"완료: {status['request_counts'].completed}/{status['request_counts'].total}")

            if status['status'] in ['completed', 'failed', 'cancelled', 'expired']:
                return status

            time.sleep(check_interval)

    def download_results(self, batch_id: str) -> pd.DataFrame:
        """결과 다운로드 및 파싱"""
        status = self.check_batch_status(batch_id)

        if status['status'] != 'completed':
            raise ValueError(f"Batch가 완료되지 않음: {status['status']}")

        # 결과 파일 다운로드
        output_file_id = status['output_file_id']
        content = self.client.files.content(output_file_id)

        # 결과 파싱
        results = []
        for line in content.text.strip().split('\n'):
            result = json.loads(line)
            custom_id = result['custom_id']
            idx = int(custom_id.replace('review_', ''))

            if result['response']['status_code'] == 200:
                body = result['response']['body']
                content = body['choices'][0]['message']['content']
                try:
                    parsed = json.loads(content)
                    results.append({
                        'idx': idx,
                        'sentiment': parsed.get('sentiment'),
                        'sentiment_score': parsed.get('sentiment_score'),
                        'aspect_labels': parsed.get('aspect_labels'),
                        'evidence': parsed.get('evidence'),
                        'summary': parsed.get('summary'),
                        'success': True
                    })
                except json.JSONDecodeError:
                    results.append({'idx': idx, 'success': False, 'error': 'JSON parse error'})
            else:
                results.append({'idx': idx, 'success': False, 'error': result['response']['body']})

        # 결과 저장
        results_df = pd.DataFrame(results)
        results_path = self.batch_dir / f"batch_results_{batch_id}.csv"
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        print(f"결과 저장: {results_path}")

        return results_df

    def run_full_pipeline(self, df: pd.DataFrame, description: str = None) -> str:
        """전체 파이프라인 실행 (비동기)"""
        # 1. JSONL 파일 생성
        jsonl_path = self.create_batch_file(df)

        # 2. 파일 업로드
        file_id = self.upload_batch_file(jsonl_path)

        # 3. Batch 생성
        batch_id = self.create_batch(file_id, description)

        # Batch ID 저장
        batch_info = {
            "batch_id": batch_id,
            "file_id": file_id,
            "jsonl_path": jsonl_path,
            "created_at": datetime.now().isoformat(),
            "total_requests": len(df)
        }

        batch_info_path = self.batch_dir / f"batch_info_{batch_id}.json"
        with open(batch_info_path, 'w', encoding='utf-8') as f:
            json.dump(batch_info, f, ensure_ascii=False, indent=2)

        print(f"\n=== Batch 생성 완료 ===")
        print(f"Batch ID: {batch_id}")
        print(f"총 요청: {len(df)}개")
        print(f"예상 비용: ${len(df) * 0.00125:.2f} (50% 할인 적용)")
        print(f"\n상태 확인: python batch_labeling.py --check {batch_id}")
        print(f"결과 다운로드: python batch_labeling.py --download {batch_id}")

        return batch_id


def main():
    import argparse

    parser = argparse.ArgumentParser(description="OpenAI Batch API ABSA 라벨링")
    parser.add_argument("--input", type=str, help="입력 CSV 파일")
    parser.add_argument("--model", type=str, default="gpt-4o", help="모델 (기본: gpt-4o)")
    parser.add_argument("--check", type=str, help="Batch 상태 확인")
    parser.add_argument("--download", type=str, help="결과 다운로드")
    parser.add_argument("--wait", type=str, help="완료 대기")
    parser.add_argument("--list", action="store_true", help="진행 중인 Batch 목록")

    args = parser.parse_args()

    labeler = BatchLabeler(model=args.model)

    if args.check:
        status = labeler.check_batch_status(args.check)
        print(json.dumps(status, indent=2, default=str))

    elif args.download:
        results = labeler.download_results(args.download)
        print(f"성공: {results['success'].sum()}/{len(results)}")

    elif args.wait:
        status = labeler.wait_for_completion(args.wait)
        print(json.dumps(status, indent=2, default=str))

    elif args.list:
        batches = labeler.client.batches.list(limit=10)
        for batch in batches.data:
            print(f"{batch.id}: {batch.status} ({batch.request_counts.completed}/{batch.request_counts.total})")

    elif args.input:
        df = pd.read_csv(args.input)
        batch_id = labeler.run_full_pipeline(df, description=f"ABSA labeling: {args.input}")

    else:
        # 기본: 전체 데이터 라벨링
        df = pd.read_csv("data/raw/sampled_reviews_20k.csv")
        batch_id = labeler.run_full_pipeline(df, description="ABSA labeling: 20k reviews")


if __name__ == "__main__":
    main()
