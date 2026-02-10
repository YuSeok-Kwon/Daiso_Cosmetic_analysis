"""
Aspect 분류 오류를 패턴 기반으로 전체 수정
1. 100개 샘플 GPT-5 검사 → 오분류 패턴 추출
2. 패턴 기반으로 전체 리뷰 수정
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

ASPECT_TO_FILE = {v: k for k, v in FILE_TO_ASPECT.items()}

# 각 aspect별 핵심 키워드 (기본)
ASPECT_KEYWORDS = {
    "배송/포장": ['배송', '택배', '포장', '도착', '배달', '늦', '빠르'],
    "품질/불량": ['불량', '하자', '고장', '파손', '유통기한', '품질'],
    "가격/가성비": ['가격', '가성비', '비싸', '싸', '저렴', '비용', '값'],
    "사용감/성능": ['발림', '흡수', '촉촉', '건조', '지속', '효과', '사용감'],
    "디자인": ['디자인', '예쁘', '이쁘', '못생', '외관', '모양'],
    "재질/냄새": ['냄새', '향', '재질', '질감', '텍스처'],
    "CS/응대": ['고객센터', '응대', '환불', '교환', '상담', 'cs'],
    "재구매": ['재구매', '또 사', '다시 사', '추천', '비추'],
    "색상/발색": ['색상', '발색', '컬러', '색깔', '색'],
    "용량/휴대": ['용량', '크기', '사이즈', '휴대', '작', '적']
}


def extract_json(text: str) -> dict:
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


def check_aspect_with_gpt5(client: OpenAI, text: str, current_aspect: str) -> dict:
    """GPT-5로 aspect 검증 및 키워드 추출"""
    prompt = f"""당신은 한국어 쇼핑몰 리뷰의 aspect 분류 전문가입니다.

[리뷰]
"{text}"

[현재 분류]
{current_aspect}

[가능한 aspect]
배송/포장, 품질/불량, 가격/가성비, 사용감/성능, 디자인, 재질/냄새, CS/응대, 재구매, 색상/발색, 용량/휴대

[작업]
1. 현재 분류가 올바른지 판단
2. 잘못됐다면 올바른 aspect와 판단 근거 키워드 제시

[출력 - JSON]
{{
  "is_correct": true/false,
  "correct_aspect": "올바른 aspect",
  "keywords": ["판단에 사용된", "키워드들"],
  "reason": "간단한 이유"
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=300,
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


def extract_patterns_from_samples(client: OpenAI, df: pd.DataFrame, current_aspect: str, sample_size: int = 100):
    """샘플에서 오분류 패턴 추출"""
    if len(df) > sample_size:
        sample_df = df.sample(n=sample_size, random_state=42)
    else:
        sample_df = df

    patterns = defaultdict(list)  # {correct_aspect: [keywords]}
    misclassified = []
    total_cost = 0

    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="샘플 분석", leave=False):
        result = check_aspect_with_gpt5(client, row['text'], current_aspect)
        if result is None:
            continue

        total_cost += result.get("cost", 0)

        if not result.get("is_correct", True):
            correct_aspect = result.get("correct_aspect")
            keywords = result.get("keywords", [])

            if correct_aspect and correct_aspect in ASPECTS:
                patterns[correct_aspect].extend(keywords)
                misclassified.append({
                    'idx': idx,
                    'text': row['text'][:50],
                    'correct_aspect': correct_aspect,
                    'keywords': keywords
                })

    # 키워드 빈도 정리
    pattern_summary = {}
    for aspect, kws in patterns.items():
        # 중복 제거 및 빈도 계산
        kw_counts = defaultdict(int)
        for kw in kws:
            if kw and len(kw) >= 2:  # 2글자 이상만
                kw_counts[kw.lower()] += 1
        # 2회 이상 등장한 키워드만
        pattern_summary[aspect] = [kw for kw, cnt in kw_counts.items() if cnt >= 2]

    return pattern_summary, misclassified, total_cost


def apply_patterns_to_all(df: pd.DataFrame, current_aspect: str, patterns: dict):
    """패턴을 전체 데이터에 적용"""
    moves = []

    for idx, row in df.iterrows():
        text = str(row['text']).lower()

        # 각 패턴 체크
        for target_aspect, keywords in patterns.items():
            if target_aspect == current_aspect:
                continue

            # 키워드 매칭
            matched_keywords = [kw for kw in keywords if kw in text]

            if len(matched_keywords) >= 1:  # 1개 이상 키워드 매칭
                moves.append({
                    'idx': idx,
                    'from': current_aspect,
                    'to': target_aspect,
                    'matched': matched_keywords,
                    'text': row['text'][:50]
                })
                break  # 첫 번째 매칭에서 중단

    return moves


def execute_moves(moves: list):
    """실제로 리뷰 이동 실행"""
    # 파일별로 그룹화
    dataframes = {}
    for filepath in aspect_dir.glob("*.csv"):
        if "correction" in filepath.name or "all_" in filepath.name:
            continue
        dataframes[filepath.stem] = pd.read_csv(filepath)

    # 이동 실행
    move_log = []
    for move in moves:
        from_file = ASPECT_TO_FILE.get(move['from'])
        to_file = ASPECT_TO_FILE.get(move['to'])

        if not from_file or not to_file:
            continue
        if from_file not in dataframes or to_file not in dataframes:
            continue

        idx = move['idx']
        from_df = dataframes[from_file]

        if idx not in from_df.index:
            continue

        # 행 복사
        row = from_df.loc[idx].copy()

        # aspect_labels 업데이트
        old_labels = str(row.get('aspect_labels', ''))
        labels = [l.strip() for l in old_labels.split(',') if l.strip()]

        if move['from'] in labels:
            labels.remove(move['from'])
        if move['to'] not in labels:
            labels.append(move['to'])
        row['aspect_labels'] = ', '.join(labels)

        # 타겟에 추가
        dataframes[to_file] = pd.concat([dataframes[to_file], row.to_frame().T], ignore_index=True)

        # 소스에서 삭제 마킹
        dataframes[from_file] = dataframes[from_file].drop(idx)

        move_log.append(move)

    # 저장
    for name, df in dataframes.items():
        df = df.reset_index(drop=True)
        df.to_csv(aspect_dir / f"{name}.csv", index=False)

    return move_log


def main():
    print("="*70)
    print("Aspect 분류 오류 - 패턴 기반 전체 수정")
    print("="*70)

    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY 필요")
        return

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    csv_files = sorted(aspect_dir.glob("*.csv"))
    csv_files = [f for f in csv_files if "correction" not in f.name and "all_" not in f.name]

    all_patterns = {}
    all_moves = []
    total_cost = 0

    print(f"\n총 {len(csv_files)}개 파일 처리\n")

    for filepath in tqdm(csv_files, desc="전체 진행"):
        file_name = filepath.stem
        current_aspect = FILE_TO_ASPECT.get(file_name)

        if not current_aspect:
            continue

        df = pd.read_csv(filepath)
        print(f"\n{'='*50}")
        print(f"📁 {file_name} ({len(df)}건)")
        print(f"{'='*50}")

        # 1. 샘플에서 패턴 추출
        print("1단계: 샘플 100개 GPT-5 분석...")
        patterns, misclassified, cost = extract_patterns_from_samples(
            client, df, current_aspect, sample_size=100
        )
        total_cost += cost

        if not patterns:
            print(f"   오분류 패턴 없음")
            continue

        print(f"   발견된 패턴:")
        for aspect, keywords in patterns.items():
            print(f"   → {aspect}: {keywords[:5]}...")

        all_patterns[file_name] = patterns

        # 2. 패턴을 전체에 적용
        print(f"\n2단계: 전체 {len(df)}건에 패턴 적용...")
        moves = apply_patterns_to_all(df, current_aspect, patterns)
        print(f"   이동 대상: {len(moves)}건")

        all_moves.extend(moves)

    # 3. 실제 이동 실행
    print(f"\n{'='*70}")
    print(f"3단계: 총 {len(all_moves)}건 이동 실행")
    print(f"{'='*70}")

    if all_moves:
        move_log = execute_moves(all_moves)
        print(f"✅ {len(move_log)}건 이동 완료")

        # 이동 요약
        summary = defaultdict(lambda: defaultdict(int))
        for m in move_log:
            summary[m['from']][m['to']] += 1

        print("\n[이동 요약]")
        for from_asp, to_dict in summary.items():
            for to_asp, cnt in to_dict.items():
                print(f"   {from_asp} → {to_asp}: {cnt}건")

    print(f"\n총 비용: ${total_cost:.4f}")

    # 로그 저장
    log_path = aspect_dir / "aspect_pattern_corrections.json"
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump({
            'patterns': {k: {a: list(kws) for a, kws in v.items()} for k, v in all_patterns.items()},
            'moves': all_moves[:100],  # 상위 100개만
            'total_moves': len(all_moves),
            'cost': total_cost
        }, f, ensure_ascii=False, indent=2)
    print(f"로그 저장: {log_path}")


if __name__ == "__main__":
    main()
