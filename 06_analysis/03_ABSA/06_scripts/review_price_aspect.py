"""
가격/가성비 aspect CSV 자동 검수 스크립트 (v2 - 오타/변형 패턴 보강)
- 규칙 1: 가격 키워드 없는 가격/가성비 라벨 → 삭제
- 규칙 2: 짧은 일반 만족 표현 → 사용감/성능으로 변경
- 규칙 3: 배송 관련 내용 → 배송/포장으로 변경
"""

import pandas as pd
import ast
import re
import glob
import copy
import os
from pathlib import Path

# ── 경로 설정 ──
BASE = str(Path(__file__).parent.parent / "01_outputs" / "data" / "labeling")
ASPECT_CSV = f"{BASE}/v2/aspect/가격_가성비_aspect.csv"
FINAL_CSV = f"{BASE}/v2/absa_results_final.csv"
BULK_PATTERN = f"{BASE}/v1/step1_team*.csv"

# ── Step 1: 데이터 로드 ──
print("=" * 80)
print("[Step 1] 데이터 로드")
print("=" * 80)

aspect_df = pd.read_csv(ASPECT_CSV)
final_df = pd.read_csv(FINAL_CSV)

# bulk labels에서 텍스트 매핑 구축
bulk_files = sorted(glob.glob(BULK_PATTERN))
text_map = {}
for f in bulk_files:
    df_bulk = pd.read_csv(f)
    for _, row in df_bulk.iterrows():
        oidx = row['original_index']
        if oidx not in text_map:
            text_map[oidx] = row['text']

print(f"  aspect CSV: {len(aspect_df)}행")
print(f"  absa final: {len(final_df)}행")
print(f"  text_map:   {len(text_map)}건")
print()

# ── aspect_labels 파싱 헬퍼 ──
def parse_labels(val):
    """문자열로 저장된 list of dicts를 파싱"""
    if pd.isna(val):
        return []
    try:
        return ast.literal_eval(str(val))
    except:
        return []

# ── 가격 키워드 패턴 (v2: 오타/변형 패턴 대폭 보강) ──
PRICE_KEYWORDS_EXACT = [
    # 기본 가격 키워드
    "가격", "가성비", "비싸", "저렴", "세일", "할인",
    "천원", "이 가격", "돈", "값", "갓성비", "가심비",
    "쿠폰", "적립", "무료", "공짜", "혜자", "금액",
    "만원", "부담없", "부담 없", "부담스럽",
    # 다이소/경쟁사 가격 맥락
    "다이소", "올리브영", "올영",
    # 오타/변형 패턴
    "가겨", "가격대비", "가겨대비",
    "저렵", "저럼",  # 저렴 오타
    "가성템", "갓성템",
    "가서미", "가서비",  # 가성비 오타
    # 숫자+금액 표현
    "3000", "5000", "1000", "2000",
    "오백원", "천원", "이천원", "삼천원", "오천원",
    "천오백", "이천오백",
    "백원", "원짜리",
]

def has_price_keyword(text):
    """텍스트에 가격 관련 키워드가 있는지 확인 (v2: 더 넓은 매칭)"""
    if pd.isna(text) or not text:
        return False
    text_lower = str(text).lower()
    # 이모지/특수문자 제거한 버전도 준비
    text_clean = re.sub(r'[ㅋㅎㅠㅜㅡㅈㄱㄴㄷㅂㅅㅇ]', '', text_lower)

    for kw in PRICE_KEYWORDS_EXACT:
        if kw in text_lower or kw in text_clean:
            return True

    # 숫자+원 패턴 (예: 3000원, 3천 원, 3 천원)
    if re.search(r'\d+\s*원', text_lower):
        return True
    # "3천 원", "5천 원" 패턴
    if re.search(r'\d+\s*천\s*원', text_lower):
        return True

    # "싸다", "싸게", "싸서" 등 (싸움 제외)
    if re.search(r'싸[다게서요고니지ㅋ!~]|싼|싸구|싸\b', text_lower):
        return True

    # "가성ㅈ비", "가성 비" 같은 자음 끼어든 오타
    if re.search(r'가성.{0,2}비', text_lower):
        return True

    return False


# 배송 키워드
DELIVERY_KEYWORDS = ["배송", "택배", "도착", "배달"]

def has_delivery_keyword(text):
    if pd.isna(text) or not text:
        return False
    text = str(text)
    return any(kw in text for kw in DELIVERY_KEYWORDS)

def is_primarily_delivery(text):
    """텍스트의 주된 내용이 배송인지 판단 (보수적)"""
    if pd.isna(text) or not text:
        return False
    text = str(text)
    delivery_count = sum(1 for kw in DELIVERY_KEYWORDS if kw in text)
    # 배송 키워드가 있고, 가격 키워드가 없어야 함
    return delivery_count >= 1 and not has_price_keyword(text)


# 일반 만족 패턴 (규칙 2)
GENERIC_POSITIVE = [
    "좋아요", "좋아", "좋습니다", "좋네요", "좋다", "좋았",
    "괜찮아요", "괜찮", "만족", "추천",
    "나쁘지않", "나쁘지 않", "무난", "쓸만"
]

def is_generic_positive_no_price(text):
    """짧고 일반적 만족 표현만 있고, 가격 키워드가 전혀 없는지"""
    if pd.isna(text) or not text:
        return False
    text = str(text).strip()
    if len(text) >= 30:
        return False
    if has_price_keyword(text):
        return False
    return any(kw in text for kw in GENERIC_POSITIVE)


# ── Step 2: 각 규칙별 후보 추출 ──
print("=" * 80)
print("[Step 2] 규칙별 후보 추출")
print("=" * 80)

rule1_candidates = []  # 가격 키워드 없음 → 삭제
rule2_candidates = []  # 일반 만족 → 사용감/성능
rule3_candidates = []  # 배송 관련 → 배송/포장

for i, row in aspect_df.iterrows():
    idx = row['idx']
    evidence = str(row.get('evidence', '')) if pd.notna(row.get('evidence')) else ''
    labels = parse_labels(row.get('aspect_labels'))

    # 원문 텍스트: evidence 또는 text_map
    text = evidence if evidence else text_map.get(idx, '')

    # 가격/가성비 라벨들의 reason 추출
    price_labels = [l for l in labels if l.get('aspect') == '가격/가성비']
    other_labels = [l for l in labels if l.get('aspect') != '가격/가성비']
    price_reasons = ' '.join([l.get('reason', '') for l in price_labels])

    # 텍스트+reason 합쳐서 가격 키워드 확인
    combined_text = f"{text} {price_reasons}"

    # 규칙 3: 배송 관련인데 가격으로 분류된 경우
    # 조건: 텍스트에 배송 키워드 있고, 가격 키워드는 text+reason 모두에서 없음
    if has_delivery_keyword(text) and not has_price_keyword(combined_text):
        rule3_candidates.append({
            'df_index': i,
            'idx': idx,
            'sentiment': row['sentiment'],
            'text': text,
            'reason': price_reasons,
            'other_aspects': [l.get('aspect') for l in other_labels],
            'all_labels': labels
        })
        continue

    # 규칙 2: 짧은 일반 만족 표현 (가격 키워드 없음 확인 내장)
    if is_generic_positive_no_price(text) and not has_price_keyword(price_reasons):
        rule2_candidates.append({
            'df_index': i,
            'idx': idx,
            'sentiment': row['sentiment'],
            'text': text,
            'reason': price_reasons,
            'other_aspects': [l.get('aspect') for l in other_labels],
            'all_labels': labels
        })
        continue

    # 규칙 1: 가격 키워드 없음 (text + reason 모두)
    if not has_price_keyword(combined_text):
        rule1_candidates.append({
            'df_index': i,
            'idx': idx,
            'sentiment': row['sentiment'],
            'text': text,
            'reason': price_reasons,
            'other_aspects': [l.get('aspect') for l in other_labels],
            'all_labels': labels
        })

print(f"\n  규칙 1 후보 (가격 키워드 없음 -> 삭제): {len(rule1_candidates)}건")
print(f"  규칙 2 후보 (일반 만족 -> 사용감/성능): {len(rule2_candidates)}건")
print(f"  규칙 3 후보 (배송 관련 -> 배송/포장):   {len(rule3_candidates)}건")
print(f"  총 후보: {len(rule1_candidates) + len(rule2_candidates) + len(rule3_candidates)}건")

# ── Step 3: 후보 목록 상세 출력 ──
print("\n" + "=" * 80)
print("[Step 3] 후보 목록 상세")
print("=" * 80)

print("\n--- [규칙 1] 가격 키워드 없는 가격/가성비 라벨 -> 삭제 ---")
for c in rule1_candidates:
    print(f"\n  [규칙1] idx={c['idx']}, sentiment={c['sentiment']}")
    print(f"  text: {c['text'][:120]}")
    print(f"  reason: {c['reason'][:120]}")
    print(f"  다른 aspects: {c['other_aspects']}")
    print(f"  -> 조치: 가격/가성비 라벨 삭제")

print(f"\n--- [규칙 2] 짧은 일반 만족 표현 -> 사용감/성능으로 변경 ---")
for c in rule2_candidates:
    print(f"\n  [규칙2] idx={c['idx']}, sentiment={c['sentiment']}")
    print(f"  text: {c['text'][:120]}")
    print(f"  reason: {c['reason'][:120]}")
    print(f"  다른 aspects: {c['other_aspects']}")
    print(f"  -> 조치: 가격/가성비 -> 사용감/성능으로 변경")

print(f"\n--- [규칙 3] 배송 관련 -> 배송/포장으로 변경 ---")
for c in rule3_candidates:
    print(f"\n  [규칙3] idx={c['idx']}, sentiment={c['sentiment']}")
    print(f"  text: {c['text'][:120]}")
    print(f"  reason: {c['reason'][:120]}")
    print(f"  다른 aspects: {c['other_aspects']}")
    print(f"  -> 조치: 가격/가성비 -> 배송/포장으로 변경")

# ── Step 4: 수정 적용 ──
print("\n" + "=" * 80)
print("[Step 4] 수정 적용")
print("=" * 80)

# 수정 추적
modifications = []

# --- aspect CSV 수정: 모든 후보를 가격 aspect CSV에서 제거 ---
remove_indices = (
    [c['df_index'] for c in rule1_candidates] +
    [c['df_index'] for c in rule2_candidates] +
    [c['df_index'] for c in rule3_candidates]
)
aspect_df_modified = aspect_df.drop(index=remove_indices).copy()

for c in rule1_candidates:
    modifications.append({
        'rule': 1,
        'idx': c['idx'],
        'text': c['text'][:60],
        'action': '가격/가성비 라벨 삭제'
    })
for c in rule2_candidates:
    modifications.append({
        'rule': 2,
        'idx': c['idx'],
        'text': c['text'][:60],
        'action': '가격/가성비 -> 사용감/성능 변경'
    })
for c in rule3_candidates:
    modifications.append({
        'rule': 3,
        'idx': c['idx'],
        'text': c['text'][:60],
        'action': '가격/가성비 -> 배송/포장 변경'
    })

# --- absa_results_final.csv 수정 ---
final_df_modified = final_df.copy()

# 수정 대상 idx 목록
all_candidates = {}
for c in rule1_candidates:
    all_candidates[c['idx']] = ('delete', c)
for c in rule2_candidates:
    all_candidates[c['idx']] = ('change_usage', c)
for c in rule3_candidates:
    all_candidates[c['idx']] = ('change_delivery', c)

final_modifications_count = 0
for i, row in final_df_modified.iterrows():
    idx = row['idx']
    if idx not in all_candidates:
        continue

    action_type, candidate = all_candidates[idx]
    labels = parse_labels(row['aspect_labels'])
    if not labels:
        continue

    new_labels = []
    modified = False

    for label in labels:
        if label.get('aspect') == '가격/가성비':
            if action_type == 'delete':
                modified = True
                continue  # 이 라벨을 건너뜀 (삭제)
            elif action_type == 'change_usage':
                new_label = copy.deepcopy(label)
                new_label['aspect'] = '사용감/성능'
                new_label['reason'] = f"(검수 변경) {label.get('reason', '')}"
                new_labels.append(new_label)
                modified = True
                continue
            elif action_type == 'change_delivery':
                new_label = copy.deepcopy(label)
                new_label['aspect'] = '배송/포장'
                new_label['reason'] = f"(검수 변경) {label.get('reason', '')}"
                new_labels.append(new_label)
                modified = True
                continue
        new_labels.append(label)

    if modified:
        if len(new_labels) == 0:
            new_labels = [{'aspect': '미분류', 'sentiment': 'neutral', 'confidence': 0.5, 'reason': '검수: 가격/가성비 라벨 제거 후 미분류'}]
        final_df_modified.at[i, 'aspect_labels'] = str(new_labels)
        final_modifications_count += 1

print(f"  aspect CSV 수정: {len(aspect_df) - len(aspect_df_modified)}행 제거")
print(f"  absa final 수정: {final_modifications_count}건 라벨 변경")

# ── Step 5: 저장 ──
print("\n" + "=" * 80)
print("[Step 5] 저장")
print("=" * 80)

# 백업은 이미 v1에서 생성됨, 덮어쓰지 않음
aspect_backup = ASPECT_CSV.replace('.csv', '_backup_before_review.csv')
final_backup = FINAL_CSV.replace('.csv', '_backup_before_review.csv')

if os.path.exists(aspect_backup):
    print(f"  백업 이미 존재: {os.path.basename(aspect_backup)}")
if os.path.exists(final_backup):
    print(f"  백업 이미 존재: {os.path.basename(final_backup)}")

# 저장
aspect_df_modified.to_csv(ASPECT_CSV, index=False)
final_df_modified.to_csv(FINAL_CSV, index=False)
print(f"\n  저장 완료!")

# ── 결과 요약 ──
print("\n" + "=" * 80)
print("[결과 요약]")
print("=" * 80)
print(f"\n  수정 전 aspect CSV: {len(aspect_df)}행")
print(f"  수정 후 aspect CSV: {len(aspect_df_modified)}행")
print(f"  제거된 행:          {len(aspect_df) - len(aspect_df_modified)}행")
print(f"\n  규칙별 수정 건수:")
rule_counts = {}
for m in modifications:
    rule_counts[m['rule']] = rule_counts.get(m['rule'], 0) + 1
for rule, count in sorted(rule_counts.items()):
    rule_names = {1: '가격 키워드 없음 -> 삭제', 2: '일반 만족 -> 사용감/성능', 3: '배송 관련 -> 배송/포장'}
    print(f"    규칙 {rule} ({rule_names[rule]}): {count}건")

print(f"\n  수정된 항목 상세:")
print(f"  {'idx':<8} {'규칙':<8} {'text':<60} 조치")
print(f"  {'─'*8} {'─'*8} {'─'*60} {'─'*30}")
for m in modifications:
    t = m['text'] if len(m['text']) <= 58 else m['text'][:55] + '...'
    print(f"  {m['idx']:<8} 규칙{m['rule']:<6} {t:<60} {m['action']}")

print("\n  완료!")
