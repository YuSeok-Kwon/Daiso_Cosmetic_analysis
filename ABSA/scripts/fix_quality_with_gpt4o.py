"""
품질.csv 감성 레이블 GPT-5 재분석 스크립트
- 이전에 수정된 20개 리뷰를 GPT-5로 재분석
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

from openai import OpenAI


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


import re

def extract_json(text: str) -> dict:
    """응답에서 JSON 추출"""
    if not text:
        return None
    # JSON 블록 찾기
    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            pass
    # 전체가 JSON인 경우
    try:
        return json.loads(text)
    except:
        return None


def relabel_with_gpt5(client: OpenAI, text: str, rating: int) -> dict:
    """GPT-5로 재분석"""
    prompt = build_quality_prompt(text, rating)

    try:
        response = client.chat.completions.create(
            model="gpt-5",  # GPT-5 사용
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=1500,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        result = extract_json(content)

        if result is None:
            print(f"JSON 파싱 실패: {content[:100]}...")
            return None

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

        # Calculate cost (gpt-5 pricing - estimate)
        tokens_in = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens
        cost = tokens_in * 0.005 / 1000 + tokens_out * 0.015 / 1000

        result["tokens"] = tokens_in + tokens_out
        result["cost"] = cost

        return result

    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    print("="*60)
    print("품질.csv GPT-5 재분석")
    print("="*60)

    # Check API key
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        return

    # Load correction log
    log_path = project_root / "data" / "raw" / "by_aspect" / "품질_correction_log.json"
    with open(log_path, 'r', encoding='utf-8') as f:
        prev_corrections = json.load(f)

    print(f"이전 수정 건수: {len(prev_corrections)}개")

    # Load current data
    input_path = project_root / "data" / "raw" / "by_aspect" / "품질.csv"
    df = pd.read_csv(input_path)

    # Initialize client
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Re-analyze with GPT-5
    corrections = []
    total_cost = 0

    print("\n[GPT-5 재분석 시작]")
    for prev in tqdm(prev_corrections):
        df_idx = prev['df_idx']
        row = df.iloc[df_idx]

        # Get full text from dataframe
        full_text = row['text']
        rating = row['rating']

        result = relabel_with_gpt5(client, full_text, rating)

        if result is None:
            continue

        total_cost += result.get("cost", 0)

        # Get previous result from mini
        prev_sentiment = prev['after']
        prev_score = prev['new_score']

        # New result from gpt-4o
        new_sentiment = result['sentiment']
        new_score = result['sentiment_score']

        corrections.append({
            'df_idx': int(df_idx),
            'original_idx': int(prev['original_idx']),
            'text': full_text[:60],
            'rating': int(rating),
            'mini_sentiment': prev_sentiment,
            'mini_score': float(prev_score),
            'gpt5_sentiment': new_sentiment,
            'gpt5_score': float(new_score),
            'evidence': result.get('evidence', ''),
            'reason': result.get('reason', ''),
            'changed': prev_sentiment != new_sentiment
        })

        # Update dataframe
        df.at[df_idx, 'sentiment'] = new_sentiment
        df.at[df_idx, 'sentiment_score'] = new_score

        # Update summary
        old_summary = df.at[df_idx, 'summary']
        if '부정적' in old_summary and new_sentiment == 'neutral':
            df.at[df_idx, 'summary'] = old_summary.replace('부정적', '중립적')
        elif '부정적' in old_summary and new_sentiment == 'positive':
            df.at[df_idx, 'summary'] = old_summary.replace('부정적', '긍정적')
        elif '긍정적' in old_summary and new_sentiment == 'negative':
            df.at[df_idx, 'summary'] = old_summary.replace('긍정적', '부정적')
        elif '긍정적' in old_summary and new_sentiment == 'neutral':
            df.at[df_idx, 'summary'] = old_summary.replace('긍정적', '중립적')
        elif '중립적' in old_summary and new_sentiment == 'negative':
            df.at[df_idx, 'summary'] = old_summary.replace('중립적', '부정적')
        elif '중립적' in old_summary and new_sentiment == 'positive':
            df.at[df_idx, 'summary'] = old_summary.replace('중립적', '긍정적')

    # Results
    print("\n" + "="*60)
    print(f"GPT-5 재분석 완료")
    print("="*60)
    print(f"분석: {len(corrections)}개")
    print(f"비용: ${total_cost:.4f}")

    # Show differences
    changed = [c for c in corrections if c['changed']]
    print(f"\n[mini vs gpt-4o 차이]: {len(changed)}개")

    for c in corrections:
        status = "⚡ 변경" if c['changed'] else "✓ 동일"
        print(f"\n[{c['original_idx']}] {status}")
        print(f"    rating: {c['rating']}")
        print(f"    text: {c['text']}...")
        print(f"    mini:  {c['mini_sentiment']} ({c['mini_score']})")
        print(f"    gpt5: {c['gpt5_sentiment']} ({c['gpt5_score']})")
        if c['changed']:
            print(f"    근거: {c['evidence'][:60]}...")
            print(f"    이유: {c['reason'][:60]}...")

    # Save
    print("\n[저장 후 분포]")
    print(df['sentiment'].value_counts())

    df.to_csv(input_path, index=False)
    print(f"\n품질.csv 저장 완료")

    # Save new correction log
    log_path_new = project_root / "data" / "raw" / "by_aspect" / "품질_correction_gpt5.json"
    with open(log_path_new, 'w', encoding='utf-8') as f:
        json.dump(corrections, f, ensure_ascii=False, indent=2)
    print(f"GPT-5 수정 로그 저장: {log_path_new}")


if __name__ == "__main__":
    main()
