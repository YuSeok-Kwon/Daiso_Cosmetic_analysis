"""
ABSA 패턴 기반 재분석
1단계: 샘플 100개 GPT-5 분석 → aspect별 sentiment 패턴 추출
2단계: 패턴을 전체 데이터에 적용
"""

import sys
import os
import json
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from openai import OpenAI

project_root = Path(__file__).parent.parent
aspect_dir = project_root / "data" / "raw" / "by_aspect"
output_dir = project_root / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

ASPECTS = [
    "배송/포장", "품질/불량", "가격/가성비", "사용감/성능",
    "디자인", "재질/냄새", "CS/응대", "재구매", "색상/발색", "용량/휴대"
]

FILE_TO_ASPECT = {
    "배송_포장": "배송/포장",
    "품질": "품질/불량",
    "가격_가성비": "가격/가성비",
    "사용감_성능": "사용감/성능",
    "디자인": "디자인",
    "재질_냄새": "재질/냄새",
    "CS_응대": "CS/응대",
    "재구매": "재구매",
    "색상_발색": "색상/발색",
    "용량_휴대": "용량/휴대"
}


def extract_json(text: str) -> dict:
    """응답에서 JSON 추출"""
    if not text:
        return None
    # JSON 블록 찾기
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            pass
    try:
        return json.loads(text)
    except:
        return None


def analyze_with_gpt5(client: OpenAI, text: str, rating: int) -> dict:
    """GPT-5로 리뷰의 aspect별 sentiment 분석"""
    prompt = f"""당신은 한국어 쇼핑몰 리뷰의 ABSA(Aspect-Based Sentiment Analysis) 전문가입니다.

[리뷰]
"{text}"

[별점]
{rating}점 (5점 만점)

[가능한 Aspect]
배송/포장, 품질/불량, 가격/가성비, 사용감/성능, 디자인, 재질/냄새, CS/응대, 재구매, 색상/발색, 용량/휴대

[작업]
이 리뷰에서 언급된 aspect와 각각의 sentiment를 추출하세요.
- 실제로 언급된 aspect만 추출
- 각 aspect별로 독립적인 sentiment 판단
- sentiment: positive/neutral/negative
- score: -1.0 ~ 1.0

[출력 - JSON]
{{
  "aspects": [
    {{
      "aspect": "aspect명",
      "sentiment": "positive/neutral/negative",
      "score": 0.8,
      "keywords": ["판단 근거 키워드"],
      "opinion": "해당 aspect에 대한 의견 표현"
    }}
  ]
}}

반드시 리뷰에서 실제로 언급된 aspect만 포함하세요."""

    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=800,
            response_format={"type": "json_object"}
        )
        result = extract_json(response.choices[0].message.content)
        if result:
            tokens = response.usage.prompt_tokens + response.usage.completion_tokens
            result["cost"] = tokens * 0.01 / 1000
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None


def extract_patterns_from_samples(client: OpenAI, all_dfs: dict, sample_ratio: float = 0.1, max_samples: int = 300):
    """샘플에서 aspect-sentiment 패턴 추출 (각 파일의 10%, 최대 300개)"""
    # {aspect: {sentiment: [keywords]}}
    patterns = defaultdict(lambda: defaultdict(list))
    all_samples = []
    total_cost = 0

    for file_name, df in all_dfs.items():
        sample_size = max(10, min(int(len(df) * sample_ratio), max_samples))  # 최소 10개, 최대 300개
        print(f"\n📁 {file_name} ({len(df)}건) → {sample_size}개 샘플 분석...")

        if len(df) > sample_size:
            sample_df = df.sample(n=sample_size, random_state=42)
        else:
            sample_df = df

        for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="샘플 분석", leave=False):
            result = analyze_with_gpt5(client, row['text'], row.get('rating', 5))

            if result is None:
                continue

            total_cost += result.get("cost", 0)

            aspects_data = result.get("aspects", [])
            for asp_data in aspects_data:
                aspect = asp_data.get("aspect")
                sentiment = asp_data.get("sentiment")
                keywords = asp_data.get("keywords", [])

                if aspect in ASPECTS and sentiment in ["positive", "neutral", "negative"]:
                    patterns[aspect][sentiment].extend(keywords)
                    all_samples.append({
                        'file': file_name,
                        'text': row['text'][:50],
                        'aspect': aspect,
                        'sentiment': sentiment,
                        'score': asp_data.get('score', 0),
                        'keywords': keywords
                    })

    # 패턴 정리: 빈도 2회 이상 키워드만
    refined_patterns = {}
    for aspect, sent_dict in patterns.items():
        refined_patterns[aspect] = {}
        for sentiment, kws in sent_dict.items():
            kw_counts = defaultdict(int)
            for kw in kws:
                if kw and len(kw) >= 2:
                    kw_counts[kw.lower()] += 1
            refined_patterns[aspect][sentiment] = [kw for kw, cnt in kw_counts.items() if cnt >= 2]

    return refined_patterns, all_samples, total_cost


def apply_patterns_to_all(all_dfs: dict, patterns: dict):
    """패턴을 전체 데이터에 적용"""
    results = []

    for file_name, df in all_dfs.items():
        print(f"\n📁 {file_name} 패턴 적용 중...")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="적용", leave=False):
            text = str(row['text']).lower()
            rating = row.get('rating', 5)

            detected_aspects = []

            for aspect, sent_patterns in patterns.items():
                best_match = None
                best_score = 0

                for sentiment, keywords in sent_patterns.items():
                    matched = [kw for kw in keywords if kw in text]
                    if len(matched) > best_score:
                        best_score = len(matched)
                        best_match = {
                            'aspect': aspect,
                            'sentiment': sentiment,
                            'matched_keywords': matched,
                            'score': 0.8 if sentiment == 'positive' else (-0.8 if sentiment == 'negative' else 0.0)
                        }

                if best_match and best_score >= 1:
                    detected_aspects.append(best_match)

            # 기존 aspect_labels도 확인
            old_labels = str(row.get('aspect_labels', ''))
            old_aspects = [l.strip() for l in old_labels.split(',') if l.strip()]

            results.append({
                'file': file_name,
                'idx': idx,
                'text': row['text'],
                'rating': rating,
                'old_sentiment': row.get('sentiment', ''),
                'old_score': row.get('sentiment_score', 0),
                'old_aspects': old_aspects,
                'new_aspects': detected_aspects
            })

    return results


def save_new_format(results: list, output_path: Path):
    """새 형식으로 저장"""
    rows = []
    for r in results:
        if r['new_aspects']:
            for asp in r['new_aspects']:
                rows.append({
                    'text': r['text'],
                    'rating': r['rating'],
                    'aspect': asp['aspect'],
                    'sentiment': asp['sentiment'],
                    'score': asp['score'],
                    'matched_keywords': ', '.join(asp.get('matched_keywords', []))
                })
        else:
            # 패턴 매칭 안 된 경우 기존 데이터 유지
            for old_asp in r['old_aspects']:
                rows.append({
                    'text': r['text'],
                    'rating': r['rating'],
                    'aspect': old_asp,
                    'sentiment': r['old_sentiment'],
                    'score': r['old_score'],
                    'matched_keywords': ''
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    return df


def main():
    print("="*70)
    print("ABSA 패턴 기반 재분석")
    print("1단계: GPT-5 샘플 분석 → 패턴 추출")
    print("2단계: 패턴으로 전체 데이터 적용")
    print("="*70)

    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY 필요")
        return

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # 모든 파일 로드
    all_dfs = {}
    csv_files = sorted(aspect_dir.glob("*.csv"))
    csv_files = [f for f in csv_files if "correction" not in f.name and "all_" not in f.name]

    total_reviews = 0
    for filepath in csv_files:
        df = pd.read_csv(filepath)
        all_dfs[filepath.stem] = df
        total_reviews += len(df)

    print(f"\n총 {len(csv_files)}개 파일, {total_reviews}건 리뷰")

    # 1단계: 패턴 추출
    print("\n" + "="*70)
    print("1단계: 샘플 GPT-5 분석")
    print("="*70)

    patterns, samples, cost = extract_patterns_from_samples(client, all_dfs, sample_ratio=0.1, max_samples=300)

    print(f"\n추출된 패턴:")
    for aspect, sent_dict in patterns.items():
        print(f"\n  {aspect}:")
        for sentiment, keywords in sent_dict.items():
            if keywords:
                print(f"    {sentiment}: {keywords[:5]}...")

    print(f"\n1단계 비용: ${cost:.4f}")

    # 패턴 저장
    pattern_path = output_dir / "absa_patterns.json"
    with open(pattern_path, 'w', encoding='utf-8') as f:
        json.dump({
            'patterns': patterns,
            'samples': samples[:200],
            'cost': cost
        }, f, ensure_ascii=False, indent=2)
    print(f"패턴 저장: {pattern_path}")

    # 2단계: 패턴 적용
    print("\n" + "="*70)
    print("2단계: 전체 데이터에 패턴 적용")
    print("="*70)

    results = apply_patterns_to_all(all_dfs, patterns)

    # 결과 저장
    output_path = output_dir / "absa_results.csv"
    df_result = save_new_format(results, output_path)

    print(f"\n결과 저장: {output_path}")
    print(f"총 {len(df_result)}건 (aspect-sentiment 쌍)")

    # 통계
    print("\n[Aspect별 Sentiment 분포]")
    if 'aspect' in df_result.columns and 'sentiment' in df_result.columns:
        pivot = df_result.groupby(['aspect', 'sentiment']).size().unstack(fill_value=0)
        print(pivot)

    # 의심 케이스 추출 (2단계용)
    suspicious = []
    for r in results:
        rating = r['rating']
        for asp in r['new_aspects']:
            sentiment = asp['sentiment']
            # 별점 불일치
            if rating >= 4 and sentiment == 'negative':
                suspicious.append({**r, 'reason': f'별점 {rating} but negative'})
            elif rating <= 2 and sentiment == 'positive':
                suspicious.append({**r, 'reason': f'별점 {rating} but positive'})

    suspicious_path = output_dir / "suspicious_cases.json"
    with open(suspicious_path, 'w', encoding='utf-8') as f:
        json.dump(suspicious[:500], f, ensure_ascii=False, indent=2, default=str)

    print(f"\n의심 케이스: {len(suspicious)}건 → {suspicious_path}")
    print(f"(2단계에서 GPT-4o로 검증 예정)")

    print(f"\n총 비용: ${cost:.4f}")


if __name__ == "__main__":
    main()
