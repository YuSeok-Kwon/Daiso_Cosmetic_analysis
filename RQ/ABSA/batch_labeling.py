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
2. **미분류 처리**: 확실한 Aspect가 없으면 "미분류" Aspect + neutral sentiment로 반환
3. **혼합 감성 분리**: "내용물은 좋은데 용기가 샌다"처럼 역접(근데, 하지만)이 있으면 각 Aspect를 분리 추출
4. **별점-텍스트 불일치**: "구하기 힘들어서 짜증 (5점)"은 품절 대란으로 인한 아쉬움 → 재구매 positive
5. **재구매 시그널**: "쟁여둠, 또 삼, 정착, 보이면 무조건" → 재구매 positive (confidence ≥ 0.9)
6. **미사용 리뷰**: 사용 전/기대감만 있으면 성능/품질/재질 Aspect 추출 금지

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
- **사용감/성능**: 발림성, 지속력, 커버력, 보습력, 효과 등 '사용 경험'
  - O: "발림 좋아요", "커버력 별로", "보습력 최고", "효과 있어요"
- **핵심**: "발색" ≠ "발림". 발색=색상, 발림=사용감

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

**4. 재구매 (다른 이슈와 함께 나올 때)**
- **재구매 의사가 명확하면 반드시 재구매 Aspect 포함**
  - O: "또 살거예요", "재구매할게요", "쟁여둠", "정착", "계속 쓸" → 재구매 추출 필수
  - O: "전에 써보고 좋아서 재구매" → 재구매 positive (불량 이슈는 별도 Aspect로)
  - O: "계속 쓰던 제품" → 재구매 neutral/positive

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

[CS/응대 판단 기준]
- 직원 친절/불친절 → CS/응대
- 교환/환불 경험 → CS/응대
- 매장 재고 부족/품절 → CS/응대 (구하기 힘들다, 없어서 못 샀다)

[재질/냄새 판단 기준]
- 제품의 물리적 텍스처: 묽다, 되다, 크리미, 젤타입 → 재질/냄새
- 향/냄새 관련: 향이 좋다, 냄새가 이상하다 → 재질/냄새
- 끈적임: 토너/에센스의 끈적임 → 재질/냄새 / 크림류는 사용감/성능과 함께 판단

**[Aspect 목록 (11개)]**
1. 배송/포장: 배송 속도, 택배 포장, 배송 중 파손
2. 품질/퀄리티: 제품 물리적 결함만 (깨짐, 부러짐, 변질, 유통기한, 불량), "퀄리티" 키워드
3. 가격/가성비: 가격 적정성, 가성비, 듀프 비교
4. 사용감/성능: 사용 경험, 효과, 만족도 (좋아요/별로/만족/최고 포함)
5. 용량/휴대: 용량 크기, 휴대 편의성
6. 디자인: 패키징 외관, 용기 구조/결함 (펌프, 뚜껑)
7. 재질/냄새: 텍스처, 질감, 향기/냄새
8. CS/응대: 교환/환불, 직원 응대, 매장 재고/품절
9. 재구매: 재구매 의사, 추천 (쟁여둠, 정착, 또 살)
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
