"""
두 개의 API 키로 병렬 배치 처리
"""
import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
from openai import OpenAI

# 프롬프트 빌더 (기존과 동일)
def build_prompt(row):
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
1. **확실한 Aspect만 추출**: confidence ≥ 0.7인 Aspect만 추출하세요.
2. **미분류 처리**: 확실한 Aspect가 없으면 "미분류" Aspect + neutral sentiment로 반환
3. **혼합 감성 분리**: 역접이 있으면 각 Aspect를 분리 추출
4. **재구매 시그널**: "쟁여둠, 또 삼, 정착" → 재구매 positive

**[Aspect 혼동 방지]**
- **색상/발색**: "발색, 색, 컬러, 톤" 키워드 → 색상/발색
- **사용감/성능**: 발림성, 지속력, 커버력, 효과 → 사용감/성능
- **배송/포장**: "왔어요, 도착, 배송" + 파손/누락 → 배송/포장
- **품질/퀄리티**: 제조 결함, "퀄리티" 키워드 → 품질/퀄리티

**[Aspect 목록 (11개)]**
배송/포장, 품질/퀄리티, 가격/가성비, 사용감/성능, 용량/휴대, 디자인, 재질/냄새, CS/응대, 재구매, 색상/발색, 미분류

**[출력 형식]**
{{
  "sentiment": "positive|neutral|negative",
  "sentiment_score": -1.0 ~ 1.0,
  "aspect_labels": [
    {{"aspect": "Aspect명", "sentiment": "positive|neutral|negative", "confidence": 0.0~1.0, "reason": "근거"}}
  ],
  "evidence": "핵심 근거 원문",
  "summary": "30자 이내 요약"
}}

반드시 유효한 JSON만 반환하세요."""
    return prompt


def run_worker(api_key, start_batch, end_batch, worker_id):
    """단일 워커 실행"""
    client = OpenAI(api_key=api_key)
    data_dir = Path(__file__).parent / "data"
    batch_dir = data_dir / "batch"

    # 데이터 로드
    df = pd.read_csv(data_dir / "raw/sampled_reviews_20k.csv")

    BATCH_SIZE = 50
    all_results = []

    print(f"[Worker {worker_id}] 배치 {start_batch}~{end_batch} 시작")

    for batch_num in range(start_batch, end_batch + 1):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(df))
        df_batch = df.iloc[start_idx:end_idx]

        if len(df_batch) == 0:
            continue

        print(f"[Worker {worker_id}] 배치 {batch_num} (리뷰 {start_idx}-{end_idx})")

        # JSONL 파일 생성
        jsonl_path = batch_dir / f"worker{worker_id}_batch_{batch_num:04d}.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for idx, row in df_batch.iterrows():
                request = {
                    "custom_id": f"review_{idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": build_prompt(row)}],
                        "temperature": 0.3,
                        "max_tokens": 500,
                        "response_format": {"type": "json_object"}
                    }
                }
                f.write(json.dumps(request, ensure_ascii=False) + "\n")

        # 업로드 및 배치 생성
        try:
            with open(jsonl_path, 'rb') as f:
                file = client.files.create(file=f, purpose="batch")

            batch = client.batches.create(
                input_file_id=file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"description": f"Worker {worker_id} batch {batch_num}"}
            )

            # 완료 대기
            while True:
                status = client.batches.retrieve(batch.id)
                if status.status == "completed":
                    break
                elif status.status in ["failed", "cancelled", "expired"]:
                    print(f"[Worker {worker_id}] 배치 {batch_num} 실패: {status.status}")
                    break
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] {status.request_counts.completed}/{status.request_counts.total}", end="\r")
                time.sleep(10)

            # 결과 다운로드
            if status.status == "completed":
                content = client.files.content(status.output_file_id)
                for line in content.text.strip().split('\n'):
                    result = json.loads(line)
                    idx = int(result['custom_id'].replace('review_', ''))
                    if result['response']['status_code'] == 200:
                        body = result['response']['body']
                        content_text = body['choices'][0]['message']['content']
                        try:
                            parsed = json.loads(content_text)
                            all_results.append({
                                'idx': idx,
                                'sentiment': parsed.get('sentiment'),
                                'sentiment_score': parsed.get('sentiment_score'),
                                'aspect_labels': parsed.get('aspect_labels'),
                                'evidence': parsed.get('evidence'),
                                'summary': parsed.get('summary'),
                                'success': True
                            })
                        except:
                            all_results.append({'idx': idx, 'success': False})
                    else:
                        all_results.append({'idx': idx, 'success': False})

                # 중간 저장
                results_df = pd.DataFrame(all_results)
                results_df.to_csv(batch_dir / f"worker{worker_id}_results.csv", index=False, encoding='utf-8-sig')
                print(f"\n[Worker {worker_id}] 배치 {batch_num} 완료 ({len(all_results)}개 누적)")

            time.sleep(3)

        except Exception as e:
            print(f"[Worker {worker_id}] 배치 {batch_num} 오류: {e}")
            time.sleep(30)
            continue

    print(f"[Worker {worker_id}] 완료! 총 {len(all_results)}개 처리")
    return all_results


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python run_parallel_batch.py <api_key> <start_batch> <end_batch> <worker_id>")
        sys.exit(1)

    api_key = sys.argv[1]
    start_batch = int(sys.argv[2])
    end_batch = int(sys.argv[3])
    worker_id = sys.argv[4]

    run_worker(api_key, start_batch, end_batch, worker_id)
