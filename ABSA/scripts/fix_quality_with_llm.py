"""
품질.csv 감성 레이블 LLM 재분석 스크립트
- 부정확해 보이는 리뷰만 선별해서 ChatGPT로 재분석
- 더 정확한 프롬프트로 품질/불량 관련 감성 분석
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from openai_client import OpenAIClient


def identify_suspicious_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """부정확해 보이는 리뷰 식별"""
    suspicious_indices = []
    reasons = []

    for idx, row in df.iterrows():
        rating = row['rating']
        sentiment = row['sentiment']
        text = str(row['text']).lower()
        score = row['sentiment_score']

        # 불량 관련 키워드
        defect_keywords = ['불량', '안열', '안나와', '안되', '안돌', '터져', '깨져',
                          '찢어', '박살', '하자', '고장', '파손', '망가']
        has_defect = any(kw in text for kw in defect_keywords)

        # 의심 케이스 1: rating 1점 + neutral
        if rating == 1 and sentiment == 'neutral':
            suspicious_indices.append(idx)
            reasons.append(f"rating 1 + neutral (score: {score})")
            continue

        # 의심 케이스 2: rating 2점 + neutral + 불량 키워드
        if rating == 2 and sentiment == 'neutral' and has_defect:
            suspicious_indices.append(idx)
            reasons.append(f"rating 2 + neutral + 불량키워드")
            continue

        # 의심 케이스 3: 불량 키워드 + neutral
        if has_defect and sentiment == 'neutral':
            suspicious_indices.append(idx)
            reasons.append(f"불량키워드 + neutral")
            continue

        # 의심 케이스 4: score와 sentiment 불일치
        if score <= -0.5 and sentiment == 'neutral':
            suspicious_indices.append(idx)
            reasons.append(f"score {score} + neutral")
            continue

        if score >= 0.5 and sentiment == 'neutral':
            suspicious_indices.append(idx)
            reasons.append(f"score {score} + neutral")
            continue

        # 의심 케이스 5: rating 4-5점 + neutral + 긍정 키워드
        positive_keywords = ['좋', '만족', '촉촉', '넉넉', '괜찮', '예쁘', '길고', '기네']
        has_positive = any(kw in text for kw in positive_keywords)
        if rating >= 4 and sentiment == 'neutral' and has_positive:
            suspicious_indices.append(idx)
            reasons.append(f"rating {rating} + neutral + 긍정키워드")
            continue

    suspicious_df = df.iloc[suspicious_indices].copy()
    suspicious_df['reason'] = reasons

    return suspicious_df


def build_quality_prompt(review_text: str, rating: int) -> str:
    """품질/불량 전용 감성 분석 프롬프트"""
    return f"""당신은 한국어 쇼핑몰 리뷰에서 **품질/불량** 관련 감성을 분석하는 전문가입니다.

[분석 대상 리뷰]
- 평점: {rating}/5
- 내용: "{review_text}"

[분석 규칙]
1. **품질/불량 관련 표현에 집중**하여 감성 판단
2. 평점이 아닌 **리뷰 텍스트의 실제 표현**을 기준으로 판단
3. 불량, 하자, 파손, 고장, 안열림, 안나옴 등은 **명확한 부정**
4. 유통기한 넉넉, 품질 좋음 등은 **명확한 긍정**
5. 단순 사실 나열은 중립

[감성 기준]
- **negative**: 불량, 하자, 파손, 고장 등 품질 문제 언급
- **positive**: 품질 만족, 유통기한 충분 등 긍정적 언급
- **neutral**: 품질에 대한 평가 없이 사실만 나열

[출력 형식 - JSON만 반환]
{{
  "sentiment": "positive" | "neutral" | "negative",
  "sentiment_score": -1.0 ~ 1.0,
  "evidence": "판단 근거 문장 (원문 인용)",
  "reason": "왜 이렇게 판단했는지 간략 설명"
}}

반드시 유효한 JSON만 반환하세요."""


def relabel_with_llm(client: OpenAIClient, row: pd.Series, model: str = "gpt-4o-mini") -> dict:
    """LLM으로 재분석"""
    prompt = build_quality_prompt(row['text'], row['rating'])

    try:
        response = client.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        # Validate sentiment
        if result.get("sentiment") not in ["positive", "neutral", "negative"]:
            score = result.get("sentiment_score", 0)
            if score > 0.2:
                result["sentiment"] = "positive"
            elif score < -0.2:
                result["sentiment"] = "negative"
            else:
                result["sentiment"] = "neutral"

        # Validate score
        result["sentiment_score"] = max(-1.0, min(1.0, float(result.get("sentiment_score", 0))))

        # Calculate cost
        tokens_in = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens
        cost = tokens_in * 0.00015 / 1000 + tokens_out * 0.0006 / 1000

        result["tokens"] = tokens_in + tokens_out
        result["cost"] = cost

        return result

    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    print("="*60)
    print("품질.csv LLM 재분석")
    print("="*60)

    # Check API key
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        return

    # Load data
    input_path = project_root / "data" / "raw" / "by_aspect" / "품질.csv"
    df = pd.read_csv(input_path)
    print(f"총 {len(df)}개 리뷰 로드")

    # Identify suspicious reviews
    suspicious_df = identify_suspicious_reviews(df)
    print(f"\n의심스러운 리뷰: {len(suspicious_df)}개")

    if len(suspicious_df) == 0:
        print("재분석할 리뷰가 없습니다.")
        return

    print("\n[의심 리뷰 목록]")
    for idx, row in suspicious_df.iterrows():
        print(f"  [{row['index']}] {row['reason']}")
        print(f"       {row['text'][:50]}...")
        print(f"       현재: {row['sentiment']} ({row['sentiment_score']})")
        print()

    # Confirm (auto-yes for batch mode)
    print(f"\n예상 비용: ${len(suspicious_df) * 0.0003:.4f}")
    print("재분석을 진행합니다...")

    # Initialize client
    client = OpenAIClient()

    # Relabel
    corrections = []
    total_cost = 0

    print("\n[재분석 시작]")
    for idx, row in tqdm(suspicious_df.iterrows(), total=len(suspicious_df)):
        result = relabel_with_llm(client, row)

        if result is None:
            continue

        total_cost += result.get("cost", 0)

        # Check if changed
        old_sentiment = row['sentiment']
        new_sentiment = result['sentiment']

        if old_sentiment != new_sentiment:
            corrections.append({
                'df_idx': idx,
                'original_idx': row['index'],
                'text': row['text'][:50],
                'rating': row['rating'],
                'before': old_sentiment,
                'after': new_sentiment,
                'new_score': result['sentiment_score'],
                'evidence': result.get('evidence', ''),
                'reason': result.get('reason', '')
            })

            # Update dataframe
            df.at[idx, 'sentiment'] = new_sentiment
            df.at[idx, 'sentiment_score'] = result['sentiment_score']

            # Update summary
            old_summary = df.at[idx, 'summary']
            if '중립적' in old_summary and new_sentiment == 'negative':
                df.at[idx, 'summary'] = old_summary.replace('중립적', '부정적')
            elif '중립적' in old_summary and new_sentiment == 'positive':
                df.at[idx, 'summary'] = old_summary.replace('중립적', '긍정적')

    # Results
    print("\n" + "="*60)
    print(f"재분석 완료")
    print("="*60)
    print(f"검토: {len(suspicious_df)}개")
    print(f"수정: {len(corrections)}개")
    print(f"비용: ${total_cost:.4f}")

    if corrections:
        print("\n[수정 내역]")
        for c in corrections:
            print(f"\n[{c['original_idx']}] {c['before']} → {c['after']} (score: {c['new_score']})")
            print(f"    rating: {c['rating']}")
            print(f"    text: {c['text']}...")
            print(f"    근거: {c['evidence'][:50]}...")
            print(f"    이유: {c['reason']}")

    # Save
    print("\n[저장 전 분포]")
    print(df['sentiment'].value_counts())

    # Auto-save
    df.to_csv(input_path, index=False)
    print(f"\n품질.csv 저장 완료")

    # Save correction log
    log_path = project_root / "data" / "raw" / "by_aspect" / "품질_correction_log.json"
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(corrections, f, ensure_ascii=False, indent=2)
    print(f"수정 로그 저장: {log_path}")


if __name__ == "__main__":
    main()
