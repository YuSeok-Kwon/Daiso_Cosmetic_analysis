"""
모든 aspect 파일의 의심 리뷰를 GPT-5로 재분석
"""

import sys
import os
import json
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

project_root = Path(__file__).parent.parent
aspect_dir = project_root / "data" / "raw" / "by_aspect"

# 각 aspect별 부정 키워드
ASPECT_KEYWORDS = {
    "품질": ['불량', '하자', '파손', '고장', '안열', '안나와', '터져', '깨져', '찢어', '박살'],
    "배송_포장": ['늦', '파손', '훼손', '찌그러', '터져', '깨져', '안옴', '분실'],
    "가격_가성비": ['비싸', '비쌈', '아까', '손해', '별로'],
    "사용감_성능": ['안좋', '별로', '안되', '안맞', '불편', '따가', '쓰림', '건조'],
    "디자인": ['별로', '촌스러', '안예쁘', '구리', '싸구려'],
    "재질_냄새": ['냄새', '악취', '비릿', '퀴퀴', '별로'],
    "용량_휴대": ['작', '적', '모자라', '부족'],
    "색상_발색": ['안나', '별로', '안예쁘', '다르', '이상'],
    "CS_응대": ['불친절', '무시', '답변없', '환불안', '교환안'],
    "재구매": ['안삼', '안살', '안함', '비추', '별로']
}


def extract_json(text: str) -> dict:
    """응답에서 JSON 추출"""
    if not text:
        return None
    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            pass
    try:
        return json.loads(text)
    except:
        return None


def build_aspect_prompt(review_text: str, rating: int, aspect_name: str) -> str:
    """aspect별 감성 분석 프롬프트"""
    aspect_display = aspect_name.replace("_", "/")

    return f"""당신은 한국어 쇼핑몰 리뷰에서 **{aspect_display}** 관련 감성을 분석하는 전문가입니다.

[분석 대상 리뷰]
- 평점: {rating}/5
- 내용: "{review_text}"

[분석 규칙]
1. **{aspect_display} 관련 표현에 집중**하여 감성 판단
2. 평점이 아닌 **리뷰 텍스트의 실제 표현**을 기준으로 판단
3. 불만, 문제, 부정적 표현은 **negative**
4. 만족, 좋음, 긍정적 표현은 **positive**
5. 단순 사실 나열은 **neutral**

[출력 형식 - JSON만 반환]
{{
  "sentiment": "positive" | "neutral" | "negative",
  "sentiment_score": -1.0 ~ 1.0,
  "evidence": "판단 근거 문장",
  "reason": "간략 설명"
}}

반드시 유효한 JSON만 반환하세요."""


def identify_suspicious(df: pd.DataFrame, aspect_name: str) -> list:
    """의심 리뷰 식별"""
    keywords = ASPECT_KEYWORDS.get(aspect_name, [])
    suspicious = []

    for idx, row in df.iterrows():
        rating = row['rating']
        sentiment = row['sentiment']
        text = str(row['text']).lower()
        score = row.get('sentiment_score', 0)

        issues = []

        # rating 1점 + neutral
        if rating == 1 and sentiment == 'neutral':
            issues.append("rating 1 + neutral")

        # rating 2점 + neutral + 부정 키워드
        if rating == 2 and sentiment == 'neutral':
            if any(kw in text for kw in keywords):
                issues.append("rating 2 + neutral + 키워드")

        # 부정 키워드 + neutral (rating 1-3)
        if rating <= 3 and sentiment == 'neutral' and not issues:
            if any(kw in text for kw in keywords):
                issues.append("부정키워드 + neutral")

        # score 불일치
        if score <= -0.5 and sentiment == 'neutral':
            issues.append(f"score {score} + neutral")
        if score >= 0.5 and sentiment == 'neutral':
            issues.append(f"score {score} + neutral")

        # rating 4-5점 + negative
        if rating >= 4 and sentiment == 'negative':
            issues.append(f"rating {rating} + negative")

        if issues:
            suspicious.append({
                'df_idx': idx,
                'original_idx': row.get('index', idx),
                'text': row['text'],
                'rating': rating,
                'sentiment': sentiment,
                'score': score,
                'issues': issues
            })

    return suspicious


def relabel_with_gpt5(client: OpenAI, text: str, rating: int, aspect_name: str) -> dict:
    """GPT-5로 재분석"""
    prompt = build_aspect_prompt(text, rating, aspect_name)

    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=1500,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        result = extract_json(content)

        if result is None:
            return None

        # Validate
        if result.get("sentiment") not in ["positive", "neutral", "negative"]:
            score = result.get("sentiment_score", 0)
            if score > 0.2:
                result["sentiment"] = "positive"
            elif score < -0.2:
                result["sentiment"] = "negative"
            else:
                result["sentiment"] = "neutral"

        result["sentiment_score"] = max(-1.0, min(1.0, float(result.get("sentiment_score", 0))))

        # Cost
        tokens_in = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens
        result["cost"] = tokens_in * 0.005 / 1000 + tokens_out * 0.015 / 1000

        return result

    except Exception as e:
        print(f"Error: {e}")
        return None


def process_aspect_file(client: OpenAI, filepath: Path) -> dict:
    """하나의 aspect 파일 처리"""
    aspect_name = filepath.stem
    df = pd.read_csv(filepath)

    # 의심 리뷰 찾기
    suspicious = identify_suspicious(df, aspect_name)

    if not suspicious:
        return {'name': aspect_name, 'total': len(df), 'fixed': 0, 'cost': 0}

    corrections = []
    total_cost = 0

    for item in tqdm(suspicious, desc=f"{aspect_name}", leave=False):
        result = relabel_with_gpt5(client, item['text'], item['rating'], aspect_name)

        if result is None:
            continue

        total_cost += result.get("cost", 0)

        old_sentiment = item['sentiment']
        new_sentiment = result['sentiment']
        new_score = result['sentiment_score']

        df_idx = item['df_idx']

        # Update dataframe
        df.at[df_idx, 'sentiment'] = new_sentiment
        df.at[df_idx, 'sentiment_score'] = new_score

        # Update summary
        old_summary = str(df.at[df_idx, 'summary'])
        if '중립적' in old_summary and new_sentiment == 'negative':
            df.at[df_idx, 'summary'] = old_summary.replace('중립적', '부정적')
        elif '중립적' in old_summary and new_sentiment == 'positive':
            df.at[df_idx, 'summary'] = old_summary.replace('중립적', '긍정적')
        elif '부정적' in old_summary and new_sentiment == 'positive':
            df.at[df_idx, 'summary'] = old_summary.replace('부정적', '긍정적')
        elif '부정적' in old_summary and new_sentiment == 'neutral':
            df.at[df_idx, 'summary'] = old_summary.replace('부정적', '중립적')
        elif '긍정적' in old_summary and new_sentiment == 'negative':
            df.at[df_idx, 'summary'] = old_summary.replace('긍정적', '부정적')
        elif '긍정적' in old_summary and new_sentiment == 'neutral':
            df.at[df_idx, 'summary'] = old_summary.replace('긍정적', '중립적')

        if old_sentiment != new_sentiment:
            corrections.append({
                'original_idx': int(item['original_idx']),
                'before': old_sentiment,
                'after': new_sentiment,
                'score': float(new_score)
            })

    # Save
    df.to_csv(filepath, index=False)

    return {
        'name': aspect_name,
        'total': len(df),
        'checked': len(suspicious),
        'fixed': len(corrections),
        'cost': total_cost,
        'corrections': corrections
    }


def main():
    print("="*70)
    print("모든 Aspect 파일 GPT-5 재분석")
    print("="*70)

    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        return

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # 모든 CSV 파일 처리
    results = []
    total_cost = 0
    total_fixed = 0

    csv_files = sorted(aspect_dir.glob("*.csv"))
    csv_files = [f for f in csv_files if not f.name.startswith("품질_correction")]

    print(f"\n총 {len(csv_files)}개 파일 처리 예정\n")

    for filepath in tqdm(csv_files, desc="전체 진행"):
        result = process_aspect_file(client, filepath)
        results.append(result)
        total_cost += result.get('cost', 0)
        total_fixed += result.get('fixed', 0)

    # 결과 출력
    print("\n" + "="*70)
    print("처리 완료")
    print("="*70)

    print(f"\n{'Aspect':<15} {'검토':>8} {'수정':>8} {'비용':>10}")
    print("-"*45)

    for r in results:
        checked = r.get('checked', 0)
        fixed = r.get('fixed', 0)
        cost = r.get('cost', 0)
        print(f"{r['name']:<15} {checked:>8} {fixed:>8} ${cost:>8.4f}")

    print("-"*45)
    print(f"{'총계':<15} {'':>8} {total_fixed:>8} ${total_cost:>8.4f}")

    # 로그 저장
    log_path = aspect_dir / "all_corrections_gpt5.json"
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n수정 로그 저장: {log_path}")


if __name__ == "__main__":
    main()
