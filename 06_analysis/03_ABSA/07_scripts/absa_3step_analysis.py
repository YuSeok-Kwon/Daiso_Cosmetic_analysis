"""
ABSA 3단계 분석 전략
1단계: GPT-4o-mini로 전체 대량 라벨링
2단계: 불확실/충돌 케이스 GPT-4o 재판정
3단계: 골드셋 생성 (GPT-4o)

사용법:
    python absa_3step_analysis.py --team 1
    python absa_3step_analysis.py --team 2
    ...
"""

import os
import sys
import json
import re
import random
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from openai import OpenAI

# 경로 설정 (Windows/Mac 호환 - pathlib 사용)
project_root = Path(__file__).parent.parent.resolve()
split_dir = project_root / "data" / "raw" / "split"
output_dir = project_root / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

ASPECTS = [
    "배송/포장", "품질/불량", "가격/가성비", "사용감/성능",
    "디자인", "재질/냄새", "CS/응대", "재구매", "색상/발색", "용량/휴대"
]


def extract_json(text: str) -> dict:
    """응답에서 JSON 추출"""
    if not text:
        return None
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


def analyze_review(client: OpenAI, text: str, rating: int, model: str = "gpt-4o-mini") -> dict:
    """리뷰의 aspect별 sentiment 분석"""
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
- 각 aspect별로 독립적인 sentiment 판단 (별점과 무관하게 텍스트 기반)
- sentiment: positive/neutral/negative
- confidence: 0.0 ~ 1.0 (판단 확신도)

[출력 - JSON]
{{
  "aspects": [
    {{
      "aspect": "aspect명",
      "sentiment": "positive/neutral/negative",
      "confidence": 0.9,
      "reason": "판단 근거 (간단히)"
    }}
  ]
}}

반드시 리뷰에서 실제로 언급된 aspect만 포함하세요. 언급되지 않은 aspect는 절대 포함하지 마세요."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        result = extract_json(response.choices[0].message.content)
        if result:
            tokens = response.usage.prompt_tokens + response.usage.completion_tokens
            # 비용 계산 (GPT-4o-mini vs GPT-4o)
            if "mini" in model:
                cost = (response.usage.prompt_tokens * 0.00015 + response.usage.completion_tokens * 0.0006) / 1000
            else:
                cost = (response.usage.prompt_tokens * 0.0025 + response.usage.completion_tokens * 0.01) / 1000
            result["cost"] = cost
            result["model"] = model
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None


def step1_bulk_labeling(client: OpenAI, df: pd.DataFrame, team_num: int):
    """1단계: GPT-4o-mini로 전체 대량 라벨링"""
    print("\n" + "="*70)
    print(f"1단계: GPT-4o-mini 대량 라벨링 (팀원{team_num})")
    print("="*70)

    results = []
    total_cost = 0

    print(f"\n총 {len(df)}건 분석 중...")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="분석"):
        result = analyze_review(client, row['text'], row.get('rating', 5), model="gpt-4o-mini")

        if result is None:
            continue

        total_cost += result.get("cost", 0)

        for asp_data in result.get("aspects", []):
            if asp_data.get("aspect") in ASPECTS:
                results.append({
                    'original_index': row.get('original_index', idx),
                    'text': row['text'],
                    'rating': row.get('rating', 5),
                    'aspect': asp_data['aspect'],
                    'sentiment': asp_data['sentiment'],
                    'confidence': asp_data.get('confidence', 0.8),
                    'reason': asp_data.get('reason', ''),
                    'model': 'gpt-4o-mini'
                })

    # 저장
    df_results = pd.DataFrame(results)
    step1_path = output_dir / f"step1_team{team_num}_bulk_labels.csv"
    df_results.to_csv(step1_path, index=False, encoding='utf-8-sig')

    print(f"\n1단계 완료!")
    print(f"총 {len(df_results)}건 라벨링")
    print(f"비용: ${total_cost:.4f}")
    print(f"저장: {step1_path}")

    return df_results, total_cost


def step2_uncertain_review(client: OpenAI, df_step1: pd.DataFrame, team_num: int):
    """2단계: 불확실/충돌 케이스 GPT-4o 재판정"""
    print("\n" + "="*70)
    print(f"2단계: 불확실/충돌 케이스 GPT-4o 재판정 (팀원{team_num})")
    print("="*70)

    # 불확실 케이스 추출
    uncertain = df_step1[df_step1['confidence'] < 0.7].copy()
    print(f"낮은 confidence (<0.7): {len(uncertain)}건")

    # 별점-sentiment 충돌 케이스
    conflicts = df_step1[
        ((df_step1['rating'] >= 4) & (df_step1['sentiment'] == 'negative')) |
        ((df_step1['rating'] <= 2) & (df_step1['sentiment'] == 'positive'))
    ].copy()
    print(f"별점-sentiment 충돌: {len(conflicts)}건")

    # 합치기 (중복 제거)
    review_keys = set()
    cases_to_review = []

    for _, row in pd.concat([uncertain, conflicts]).iterrows():
        key = (row['text'][:100], row['aspect'])
        if key not in review_keys:
            review_keys.add(key)
            cases_to_review.append(row.to_dict())

    print(f"중복 제거 후 재판정 대상: {len(cases_to_review)}건")

    if len(cases_to_review) == 0:
        print("재판정 대상 없음")
        return df_step1, 0

    # GPT-4o로 재판정
    total_cost = 0
    reviewed_results = []

    for case in tqdm(cases_to_review, desc="GPT-4o 재판정"):
        result = analyze_review(client, case['text'], case['rating'], model="gpt-4o")

        if result is None:
            reviewed_results.append(case)  # 실패 시 기존 유지
            continue

        total_cost += result.get("cost", 0)

        # 해당 aspect 찾기
        found = False
        for asp_data in result.get("aspects", []):
            if asp_data.get("aspect") == case['aspect']:
                reviewed_results.append({
                    **case,
                    'sentiment': asp_data['sentiment'],
                    'confidence': asp_data.get('confidence', 0.9),
                    'reason': asp_data.get('reason', ''),
                    'model': 'gpt-4o'
                })
                found = True
                break

        if not found:
            # GPT-4o가 해당 aspect를 언급하지 않음 → 삭제 대상
            reviewed_results.append({**case, 'sentiment': 'REMOVE', 'model': 'gpt-4o'})

    # 결과 병합
    df_reviewed = pd.DataFrame(reviewed_results)

    # 기존 결과에서 재판정된 케이스 교체
    df_final = df_step1.copy()
    for _, reviewed_row in df_reviewed.iterrows():
        mask = (df_final['text'].str[:100] == reviewed_row['text'][:100]) & \
               (df_final['aspect'] == reviewed_row['aspect'])
        if reviewed_row['sentiment'] == 'REMOVE':
            df_final = df_final[~mask]
        else:
            df_final.loc[mask, 'sentiment'] = reviewed_row['sentiment']
            df_final.loc[mask, 'confidence'] = reviewed_row['confidence']
            df_final.loc[mask, 'model'] = reviewed_row['model']

    # 저장
    step2_path = output_dir / f"step2_team{team_num}_reviewed_labels.csv"
    df_final.to_csv(step2_path, index=False, encoding='utf-8-sig')

    print(f"\n2단계 완료!")
    print(f"재판정: {len(cases_to_review)}건")
    print(f"비용: ${total_cost:.4f}")
    print(f"저장: {step2_path}")

    return df_final, total_cost


def step3_gold_set(client: OpenAI, df_step2: pd.DataFrame, team_num: int, gold_size: int = 100):
    """3단계: 골드셋 생성 (팀당 100건)"""
    print("\n" + "="*70)
    print(f"3단계: 골드셋 생성 (팀원{team_num}, GPT-4o)")
    print("="*70)

    # 다양한 케이스 샘플링
    random.seed(42 + team_num)  # 팀별 다른 시드

    # aspect별, sentiment별 균등 샘플링
    gold_samples = []
    per_category = gold_size // (len(ASPECTS) * 3)  # aspect × sentiment 조합

    for aspect in ASPECTS:
        for sentiment in ['positive', 'neutral', 'negative']:
            subset = df_step2[(df_step2['aspect'] == aspect) &
                             (df_step2['sentiment'] == sentiment)]
            if len(subset) > 0:
                n = min(per_category, len(subset))
                gold_samples.extend(subset.sample(n=n, random_state=42).to_dict('records'))

    # 부족분 랜덤 채우기
    if len(gold_samples) < gold_size:
        remaining = df_step2[~df_step2.index.isin([s.get('original_index', -1) for s in gold_samples])]
        n_more = min(gold_size - len(gold_samples), len(remaining))
        if n_more > 0:
            gold_samples.extend(remaining.sample(n=n_more, random_state=42).to_dict('records'))

    print(f"골드셋 대상: {len(gold_samples)}건")

    # GPT-4o로 정밀 라벨링
    total_cost = 0
    gold_results = []

    for sample in tqdm(gold_samples, desc="GPT-4o 골드셋"):
        result = analyze_review(client, sample['text'], sample['rating'], model="gpt-4o")

        if result is None:
            continue

        total_cost += result.get("cost", 0)

        for asp_data in result.get("aspects", []):
            if asp_data.get("aspect") in ASPECTS:
                gold_results.append({
                    'text': sample['text'],
                    'rating': sample['rating'],
                    'aspect': asp_data['aspect'],
                    'sentiment': asp_data['sentiment'],
                    'confidence': asp_data.get('confidence', 0.95),
                    'reason': asp_data.get('reason', ''),
                    'model': 'gpt-4o',
                    'is_gold': True
                })

    # 저장
    df_gold = pd.DataFrame(gold_results)
    gold_path = output_dir / f"step3_team{team_num}_gold_set.csv"
    df_gold.to_csv(gold_path, index=False, encoding='utf-8-sig')

    print(f"\n3단계 완료!")
    print(f"골드셋: {len(df_gold)}건")
    print(f"비용: ${total_cost:.4f}")
    print(f"저장: {gold_path}")

    return df_gold, total_cost


def main():
    parser = argparse.ArgumentParser(description='ABSA 3단계 분석')
    parser.add_argument('--team', type=int, required=True, help='팀원 번호 (1-6)')
    parser.add_argument('--step', type=int, default=0, help='특정 단계만 실행 (1, 2, 3). 0=전체')
    parser.add_argument('--gold-size', type=int, default=100, help='팀당 골드셋 크기')
    args = parser.parse_args()

    if args.team < 1 or args.team > 6:
        print("Error: --team은 1~6 사이 값이어야 합니다")
        return

    print("="*70)
    print(f"ABSA 3단계 분석 - 팀원{args.team}")
    print("1단계: GPT-4o-mini 대량 라벨링")
    print("2단계: 불확실/충돌 케이스 GPT-4o 재판정")
    print("3단계: 골드셋 생성")
    print("="*70)

    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY 환경변수가 필요합니다")
        print("\n[설정 방법]")
        print("Windows (CMD):   set OPENAI_API_KEY=sk-xxxx")
        print("Windows (PS):    $env:OPENAI_API_KEY=\"sk-xxxx\"")
        print("Mac/Linux:       export OPENAI_API_KEY=sk-xxxx")
        return

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # 팀 데이터 로드
    team_file = split_dir / f"team_{args.team}.csv"
    if not team_file.exists():
        print(f"Error: {team_file} 파일이 없습니다")
        print("먼저 split_data.py를 실행하세요")
        return

    df = pd.read_csv(team_file)
    print(f"\n팀원{args.team} 데이터: {len(df)}건")

    total_cost = 0

    # 단계별 실행
    if args.step == 0 or args.step == 1:
        df_step1, cost1 = step1_bulk_labeling(client, df, args.team)
        total_cost += cost1
    else:
        # 기존 1단계 결과 로드
        step1_file = output_dir / f"step1_team{args.team}_bulk_labels.csv"
        if step1_file.exists():
            df_step1 = pd.read_csv(step1_file)
        else:
            print(f"Error: {step1_file} 파일이 없습니다. --step 1부터 실행하세요")
            return
        cost1 = 0

    if args.step == 0 or args.step == 2:
        df_step2, cost2 = step2_uncertain_review(client, df_step1, args.team)
        total_cost += cost2
    else:
        if args.step > 2:
            step2_file = output_dir / f"step2_team{args.team}_reviewed_labels.csv"
            if step2_file.exists():
                df_step2 = pd.read_csv(step2_file)
            else:
                df_step2 = df_step1
        else:
            df_step2 = df_step1
        cost2 = 0

    if args.step == 0 or args.step == 3:
        df_gold, cost3 = step3_gold_set(client, df_step2, args.team, args.gold_size)
        total_cost += cost3
    else:
        cost3 = 0

    # 최종 통계
    print("\n" + "="*70)
    print(f"팀원{args.team} 최종 결과")
    print("="*70)
    if args.step == 0 or args.step >= 2:
        print(f"총 라벨: {len(df_step2)}건")
    if args.step == 0 or args.step == 3:
        print(f"골드셋: {len(df_gold)}건")
    print(f"\n[비용]")
    if cost1 > 0:
        print(f"1단계 (GPT-4o-mini): ${cost1:.4f}")
    if cost2 > 0:
        print(f"2단계 (GPT-4o): ${cost2:.4f}")
    if cost3 > 0:
        print(f"3단계 (GPT-4o): ${cost3:.4f}")
    print(f"총 비용: ${total_cost:.4f}")

    # aspect별 분포
    if (args.step == 0 or args.step >= 2) and 'aspect' in df_step2.columns:
        print("\n[Aspect별 Sentiment 분포]")
        pivot = df_step2.groupby(['aspect', 'sentiment']).size().unstack(fill_value=0)
        print(pivot)


if __name__ == "__main__":
    main()
