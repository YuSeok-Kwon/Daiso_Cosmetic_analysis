"""
GPT-4o Batch API 전체 파이프라인
- 토큰 한도 (90,000) 때문에 50개씩 순차 처리
- 각 배치 완료 후 다음 배치 제출
"""
import json
import time
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")


class FullBatchPipeline:
    """전체 배치 파이프라인"""

    BATCH_SIZE = 50  # 토큰 한도 때문에 50개씩

    def __init__(self, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.data_dir = Path(__file__).parent / "data"
        self.batch_dir = self.data_dir / "batch"
        self.batch_dir.mkdir(parents=True, exist_ok=True)

        # 진행 상태 파일
        self.progress_file = self.batch_dir / "pipeline_progress.json"

    def _build_prompt(self, row: pd.Series) -> str:
        """프롬프트 생성"""
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

    def load_progress(self) -> dict:
        """진행 상태 로드"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {"completed_batches": [], "total_processed": 0, "current_batch": 0}

    def save_progress(self, progress: dict):
        """진행 상태 저장"""
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

    def create_single_batch(self, df_batch: pd.DataFrame, batch_num: int) -> str:
        """단일 배치 생성"""
        batch_name = f"batch_{batch_num:04d}"
        jsonl_path = self.batch_dir / f"batch_input_{batch_name}.jsonl"

        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for idx, row in df_batch.iterrows():
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

        # 파일 업로드
        with open(jsonl_path, 'rb') as f:
            file = self.client.files.create(file=f, purpose="batch")

        # 배치 생성
        batch = self.client.batches.create(
            input_file_id=file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"ABSA batch {batch_num}"}
        )

        return batch.id

    def wait_for_batch(self, batch_id: str) -> dict:
        """배치 완료 대기"""
        while True:
            batch = self.client.batches.retrieve(batch_id)
            status = batch.status

            if status == "completed":
                return {"status": "completed", "output_file_id": batch.output_file_id}
            elif status in ["failed", "cancelled", "expired"]:
                return {"status": status, "error": str(getattr(batch, "errors", None))}

            # 진행 상황 출력
            completed = batch.request_counts.completed
            total = batch.request_counts.total
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] {completed}/{total} 완료", end="\r")

            time.sleep(10)

    def download_batch_results(self, batch_id: str, output_file_id: str) -> pd.DataFrame:
        """배치 결과 다운로드"""
        content = self.client.files.content(output_file_id)

        results = []
        for line in content.text.strip().split('\n'):
            result = json.loads(line)
            custom_id = result['custom_id']
            idx = int(custom_id.replace('review_', ''))

            if result['response']['status_code'] == 200:
                body = result['response']['body']
                content_text = body['choices'][0]['message']['content']
                try:
                    parsed = json.loads(content_text)
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
                results.append({'idx': idx, 'success': False, 'error': str(result['response']['body'])})

        return pd.DataFrame(results)

    def run_pipeline(self, input_csv: str = None, start_from: int = None):
        """전체 파이프라인 실행"""
        # 데이터 로드
        if input_csv:
            df = pd.read_csv(input_csv)
        else:
            df = pd.read_csv(self.data_dir / "raw/sampled_reviews_20k.csv")

        total_reviews = len(df)
        total_batches = (total_reviews + self.BATCH_SIZE - 1) // self.BATCH_SIZE

        print(f"=== GPT-4o Batch Pipeline ===")
        print(f"총 리뷰: {total_reviews}개")
        print(f"배치 크기: {self.BATCH_SIZE}개")
        print(f"총 배치 수: {total_batches}개")
        print(f"예상 비용: ${total_reviews * 0.00125:.2f} (50% 할인)")
        print()

        # 진행 상태 로드
        progress = self.load_progress()
        start_batch = start_from if start_from else progress.get("current_batch", 0)

        # 결과 저장용
        all_results = []

        for batch_num in range(start_batch, total_batches):
            start_idx = batch_num * self.BATCH_SIZE
            end_idx = min(start_idx + self.BATCH_SIZE, total_reviews)
            df_batch = df.iloc[start_idx:end_idx]

            print(f"\n[배치 {batch_num + 1}/{total_batches}] 리뷰 {start_idx}-{end_idx}")

            # 배치 생성
            print(f"  배치 생성 중...")
            try:
                batch_id = self.create_single_batch(df_batch, batch_num)
                print(f"  배치 ID: {batch_id}")
            except Exception as e:
                print(f"  오류: {e}")
                print(f"  60초 후 재시도...")
                time.sleep(60)
                continue

            # 완료 대기
            print(f"  처리 중...")
            result = self.wait_for_batch(batch_id)

            if result["status"] == "completed":
                print(f"\n  완료!")

                # 결과 다운로드
                batch_results = self.download_batch_results(batch_id, result["output_file_id"])
                all_results.append(batch_results)

                # 중간 저장
                results_df = pd.concat(all_results, ignore_index=True)
                results_path = self.batch_dir / "pipeline_results_partial.csv"
                results_df.to_csv(results_path, index=False, encoding='utf-8-sig')

                # 진행 상태 업데이트
                progress["completed_batches"].append(batch_id)
                progress["total_processed"] = end_idx
                progress["current_batch"] = batch_num + 1
                self.save_progress(progress)

                success_count = batch_results['success'].sum()
                print(f"  성공: {success_count}/{len(batch_results)}")

            else:
                print(f"\n  실패: {result}")
                print(f"  30초 후 재시도...")
                time.sleep(30)
                continue

            # API 한도 대기 (다음 배치 전)
            if batch_num < total_batches - 1:
                print(f"  다음 배치 대기 (5초)...")
                time.sleep(5)

        # 최종 결과 저장
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            final_path = self.batch_dir / f"pipeline_results_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            final_df.to_csv(final_path, index=False, encoding='utf-8-sig')

            print(f"\n=== 파이프라인 완료 ===")
            print(f"총 처리: {len(final_df)}개")
            print(f"성공: {final_df['success'].sum()}/{len(final_df)}")
            print(f"결과 파일: {final_path}")

            return final_df

        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="GPT-4o Batch Pipeline")
    parser.add_argument("--input", type=str, help="입력 CSV 파일")
    parser.add_argument("--start-from", type=int, help="시작 배치 번호")
    parser.add_argument("--status", action="store_true", help="진행 상태 확인")

    args = parser.parse_args()

    pipeline = FullBatchPipeline(model="gpt-4o")

    if args.status:
        progress = pipeline.load_progress()
        print(json.dumps(progress, indent=2))
    else:
        pipeline.run_pipeline(
            input_csv=args.input,
            start_from=args.start_from
        )


if __name__ == "__main__":
    main()
