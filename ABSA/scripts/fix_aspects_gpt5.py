"""
모든 리뷰의 aspect 분류 오류를 GPT-5로 수정
- 각 파일의 리뷰가 해당 aspect에 맞는지 검증
- 잘못된 aspect는 올바른 파일로 이동
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

ASPECTS = [
    "배송/포장",
    "품질/불량",
    "가격/가성비",
    "사용감/성능",
    "사이즈/호환",
    "디자인",
    "재질/냄새",
    "CS/응대",
    "재구매",
    "색상/발색",
    "용량/휴대"
]

# 파일명과 aspect 매핑
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


def build_aspect_check_prompt(review_text: str, current_aspect: str) -> str:
    """aspect 검증 프롬프트"""
    aspect_list = "\n".join([f"- {a}" for a in ASPECTS])

    return f"""당신은 한국어 쇼핑몰 리뷰의 aspect(측면) 분류 전문가입니다.

[리뷰]
"{review_text}"

[현재 분류된 aspect]
{current_aspect}

[가능한 aspect 목록]
{aspect_list}

[작업]
이 리뷰가 현재 aspect에 올바르게 분류되었는지 확인하세요.

[판단 기준]
- 배송/포장: 배송 속도, 포장 상태, 택배 관련
- 품질/불량: 제품 불량, 하자, 고장, 유통기한
- 가격/가성비: 가격, 가성비, 비싸다/싸다
- 사용감/성능: 사용감, 효과, 성능, 발림성, 지속력
- 디자인: 디자인, 외관, 예쁘다/못생겼다
- 재질/냄새: 재질, 향, 냄새
- CS/응대: 고객센터, 응대, 환불, 교환 서비스
- 재구매: 재구매 의향, 추천 여부
- 색상/발색: 색상, 발색, 컬러
- 용량/휴대: 용량, 크기, 휴대성

[출력 형식 - JSON만]
{{
  "is_correct": true/false,
  "correct_aspect": "올바른 aspect (현재와 같으면 현재 aspect)",
  "reason": "판단 이유"
}}

반드시 유효한 JSON만 반환하세요."""


def check_aspect_with_gpt5(client: OpenAI, text: str, current_aspect: str) -> dict:
    """GPT-5로 aspect 검증"""
    prompt = build_aspect_check_prompt(text, current_aspect)

    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=500,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        result = extract_json(content)

        if result is None:
            return None

        # Cost
        tokens_in = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens
        result["cost"] = tokens_in * 0.005 / 1000 + tokens_out * 0.015 / 1000

        return result

    except Exception as e:
        print(f"Error: {e}")
        return None


def process_file_for_aspects(client: OpenAI, filepath: Path, sample_size: int = None) -> dict:
    """파일의 aspect 분류 검증"""
    file_name = filepath.stem
    current_aspect = FILE_TO_ASPECT.get(file_name, file_name)

    df = pd.read_csv(filepath)

    # 샘플링 (전체가 너무 많으면)
    if sample_size and len(df) > sample_size:
        check_df = df.sample(n=sample_size, random_state=42)
    else:
        check_df = df

    misclassified = []
    total_cost = 0

    for idx, row in tqdm(check_df.iterrows(), total=len(check_df), desc=file_name, leave=False):
        result = check_aspect_with_gpt5(client, row['text'], current_aspect)

        if result is None:
            continue

        total_cost += result.get("cost", 0)

        if not result.get("is_correct", True):
            correct_aspect = result.get("correct_aspect", current_aspect)
            if correct_aspect != current_aspect and correct_aspect in ASPECTS:
                misclassified.append({
                    'df_idx': idx,
                    'original_idx': row.get('index', idx),
                    'text': row['text'][:60],
                    'current_aspect': current_aspect,
                    'correct_aspect': correct_aspect,
                    'reason': result.get('reason', '')
                })

    return {
        'file': file_name,
        'total': len(df),
        'checked': len(check_df),
        'misclassified': len(misclassified),
        'details': misclassified,
        'cost': total_cost
    }


def move_misclassified_reviews(results: list):
    """잘못 분류된 리뷰를 올바른 파일로 이동"""
    # 각 파일 로드
    dataframes = {}
    for filepath in aspect_dir.glob("*.csv"):
        if filepath.name.startswith("품질_correction") or filepath.name.startswith("all_"):
            continue
        dataframes[filepath.stem] = pd.read_csv(filepath)

    moves = []

    for result in results:
        source_file = result['file']
        source_df = dataframes.get(source_file)

        if source_df is None:
            continue

        for item in result['details']:
            correct_aspect = item['correct_aspect']
            target_file = ASPECT_TO_FILE.get(correct_aspect)

            if target_file is None or target_file not in dataframes:
                continue

            df_idx = item['df_idx']
            if df_idx >= len(source_df):
                continue

            # 리뷰 데이터 복사
            row = source_df.loc[df_idx].copy()

            # aspect_labels 업데이트
            old_labels = row.get('aspect_labels', '')
            if isinstance(old_labels, str):
                labels = [l.strip() for l in old_labels.split(',')]
            else:
                labels = []

            # 현재 aspect 제거, 새 aspect 추가
            current_aspect = FILE_TO_ASPECT.get(source_file)
            if current_aspect in labels:
                labels.remove(current_aspect)
            if correct_aspect not in labels:
                labels.append(correct_aspect)

            row['aspect_labels'] = ', '.join(labels)

            # 타겟 파일에 추가
            dataframes[target_file] = pd.concat([dataframes[target_file], row.to_frame().T], ignore_index=True)

            # 소스에서 제거 표시
            moves.append({
                'from': source_file,
                'to': target_file,
                'idx': df_idx,
                'text': item['text']
            })

    # 이동된 리뷰 소스에서 삭제
    for move in moves:
        source_file = move['from']
        idx = move['idx']
        if idx in dataframes[source_file].index:
            dataframes[source_file] = dataframes[source_file].drop(idx)

    # 저장
    for file_name, df in dataframes.items():
        df = df.reset_index(drop=True)
        df.to_csv(aspect_dir / f"{file_name}.csv", index=False)

    return moves


def main():
    print("="*70)
    print("Aspect 분류 오류 검증 및 수정 (GPT-5)")
    print("="*70)

    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        return

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # 각 파일에서 샘플 검사 (전체 검사는 비용이 너무 높음)
    # 파일당 최대 100개 샘플
    SAMPLE_SIZE = 100

    results = []
    total_cost = 0
    total_misclassified = 0

    csv_files = sorted(aspect_dir.glob("*.csv"))
    csv_files = [f for f in csv_files if not f.name.startswith("품질_correction") and not f.name.startswith("all_")]

    print(f"\n파일당 최대 {SAMPLE_SIZE}개 샘플 검사")
    print(f"총 {len(csv_files)}개 파일\n")

    for filepath in tqdm(csv_files, desc="전체 진행"):
        result = process_file_for_aspects(client, filepath, sample_size=SAMPLE_SIZE)
        results.append(result)
        total_cost += result.get('cost', 0)
        total_misclassified += result.get('misclassified', 0)

    # 결과 출력
    print("\n" + "="*70)
    print("검증 완료")
    print("="*70)

    print(f"\n{'파일':<15} {'검사':>6} {'오분류':>8} {'비용':>10}")
    print("-"*45)

    for r in results:
        print(f"{r['file']:<15} {r['checked']:>6} {r['misclassified']:>8} ${r['cost']:>8.4f}")

    print("-"*45)
    print(f"{'총계':<15} {'':>6} {total_misclassified:>8} ${total_cost:>8.4f}")

    # 오분류 상세 출력
    if total_misclassified > 0:
        print(f"\n[오분류 상세 - 총 {total_misclassified}건]")
        for r in results:
            if r['misclassified'] > 0:
                print(f"\n📁 {r['file']}")
                for item in r['details'][:5]:  # 상위 5개만
                    print(f"  [{item['original_idx']}] {item['current_aspect']} → {item['correct_aspect']}")
                    print(f"       {item['text']}...")
                if len(r['details']) > 5:
                    print(f"  ... 외 {len(r['details'])-5}건")

        # 이동 확인
        print("\n오분류된 리뷰를 올바른 파일로 이동합니다...")
        moves = move_misclassified_reviews(results)
        print(f"총 {len(moves)}건 이동 완료")

    # 로그 저장
    log_path = aspect_dir / "aspect_corrections_gpt5.json"
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n수정 로그 저장: {log_path}")


if __name__ == "__main__":
    main()
