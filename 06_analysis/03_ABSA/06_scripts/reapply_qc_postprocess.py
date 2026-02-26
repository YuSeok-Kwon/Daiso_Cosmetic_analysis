"""
기존 추론 결과 CSV에 후처리 재적용 (모델 re-inference 없이)

기본 모드: Phase 2 QC 규칙만 재적용 (auxiliary, rescue, hallucination 등)
--include-phase1: Phase 1 (design_rule, keyword_gate, keyword_force_on) + Phase 2 전체 재적용

주의: Phase 1 변경사항 (예: KEYWORD_FORCE_ON_CONFIG 확장, negative_keywords 분기 등)은
      기본 모드로는 반영되지 않음. Phase 1 변경 시 --include-phase1 또는 전체 재추론 필요.

사용법:
  python reapply_qc_postprocess.py                        # Phase 2만 재적용 (덮어쓰기)
  python reapply_qc_postprocess.py --include-phase1       # Phase 1+2 전체 재적용
  python reapply_qc_postprocess.py --dry-run               # 변경 통계만 출력
  python reapply_qc_postprocess.py --output new.csv        # 별도 파일로 저장
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from RQ_absa.s1_config import (
    ASPECT_LABELS,
    ASPECT_SENTIMENT_ID_TO_LABEL,
    QC_POSTPROCESS_CONFIG,
    KEYWORD_GATE_CONFIG,
    KEYWORD_FORCE_ON_CONFIG,
)
from RQ_absa.s8_inference import ABSAInference
from RQ_absa.s7_evaluation import apply_design_rule_override, apply_full_postprocess

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "01_outputs" / "inference" / "absa_results_stage3a_v2_full.csv"


def parse_aspect_sentiments(val):
    """aspect_sentiments 컬럼을 파싱하여 리스트 반환."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val.replace("'", '"'))
        except json.JSONDecodeError:
            return eval(val)
    return []


def aspect_sentiments_to_preds(aspect_sentiments_list, aspect_labels):
    """aspect_sentiments JSON 리스트 → [N, K] numpy array 복원.

    매핑: none=0, positive=1, neutral=2, negative=3
    미분류 aspect는 무시 (모든 aspect = 0)
    """
    sent_to_id = {"none": 0, "positive": 1, "neutral": 2, "negative": 3}
    N = len(aspect_sentiments_list)
    K = len(aspect_labels)
    preds = np.zeros((N, K), dtype=np.int64)

    for i, asp_list in enumerate(aspect_sentiments_list):
        for asp in asp_list:
            name = asp.get("aspect", "")
            if name == "미분류" or name not in aspect_labels:
                continue
            j = aspect_labels.index(name)
            sent = asp.get("sentiment", "none")
            preds[i, j] = sent_to_id.get(sent, 0)

    return preds


def preds_to_aspect_sentiments(preds, aspect_labels):
    """[N, K] numpy array → aspect_sentiments 리스트로 변환."""
    id_to_sent = ASPECT_SENTIMENT_ID_TO_LABEL
    results = []
    for i in range(len(preds)):
        review_aspects = []
        for j, name in enumerate(aspect_labels):
            pred_id = int(preds[i, j])
            if pred_id == 0:
                continue
            review_aspects.append({
                "aspect": name,
                "sentiment": id_to_sent.get(pred_id, "unknown"),
                "confidence": 0.0,  # 재적용 시 confidence 복원 불가
            })
        if not review_aspects:
            review_aspects.append({
                "aspect": "미분류",
                "sentiment": "neutral",
                "confidence": 0.0,
            })
        results.append(review_aspects)
    return results


def generate_summary(sentiment, aspect_sentiments):
    """감성별 aspect 그룹핑 요약 재생성."""
    if not aspect_sentiments:
        return f"전반적으로 {sentiment}"
    sentiment_kr = {"positive": "긍정적", "neutral": "중립적", "negative": "부정적"}
    groups = {}
    for a in aspect_sentiments:
        sent = a["sentiment"]
        groups.setdefault(sent, []).append(a["aspect"])
    parts = []
    for sent, aspects in groups.items():
        aspects_str = ", ".join(aspects[:3])
        if len(aspects) > 3:
            aspects_str += " 등"
        kr = sentiment_kr.get(sent, sent)
        parts.append(f"{aspects_str} {kr}")
    return " / ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="QC 후처리 재적용")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT),
                        help="입력 CSV (기존 추론 결과)")
    parser.add_argument("--output", type=str, default=None,
                        help="출력 CSV (None이면 입력 파일 덮어쓰기)")
    parser.add_argument("--dry-run", action="store_true",
                        help="변경 통계만 출력 (파일 미저장)")
    parser.add_argument("--include-phase1", action="store_true",
                        help="Phase 1 (design_rule + keyword_gate + force_on) + Phase 2 전체 재적용")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    print("=" * 70)
    print("QC 후처리 재적용 (모델 re-inference 없이)")
    print("=" * 70)
    print(f"입력: {input_path}")
    print(f"출력: {output_path}" + (" (dry-run)" if args.dry_run else ""))

    # 활성화된 규칙 출력
    print("\n활성화된 QC 규칙:")
    for rule_name, rule_cfg in QC_POSTPROCESS_CONFIG.items():
        status = "ON" if rule_cfg.get("enabled") else "OFF"
        priority = rule_cfg.get("priority", "?")
        print(f"  [{status}] {priority} {rule_name}")

    # CSV 로드
    df = pd.read_csv(input_path)
    print(f"\n총 리뷰: {len(df):,}건")

    # aspect_sentiments 파싱
    df["_parsed_asp"] = df["aspect_sentiments"].apply(parse_aspect_sentiments)

    # [N, K] 복원
    aspect_labels = list(ASPECT_LABELS)
    old_preds = aspect_sentiments_to_preds(df["_parsed_asp"].tolist(), aspect_labels)

    # 텍스트, 별점 추출
    texts = df["text"].fillna("").astype(str).tolist()
    ratings = df["rating"].values if "rating" in df.columns else None

    if args.include_phase1:
        # Phase 1 + Phase 2 전체 재적용 (apply_full_postprocess 사용)
        print("\n[Phase 1+2] 전체 후처리 체인 재적용...")
        new_preds = apply_full_postprocess(
            texts, old_preds, ratings,
            aspect_probs=None,  # CSV에서는 probs 복원 불가
            aspect_labels=aspect_labels,
            use_design_rule=True,
        )
    else:
        # Phase 2만 재적용 (기존 동작)
        print("\n[Phase 2만] QC 후처리 규칙 재적용...")

        class _Proxy:
            pass

        runner = _Proxy()
        runner.aspect_labels = aspect_labels
        new_preds = old_preds.copy()

        # P1: 배송 보조 언급 (정규식 기반 정밀 감성 → 먼저 실행)
        new_preds = ABSAInference._apply_auxiliary_mention(runner, texts, new_preds)
        # P0: 배송 구출 (배송 aspect none + 배송 키워드 → 별점 기반 폴백)
        new_preds = ABSAInference._apply_rescue_delivery(runner, texts, new_preds, ratings)
        # P1: Aspect 환각 제거
        new_preds = ABSAInference._apply_hallucination_gate(runner, texts, new_preds)
        # P2: 미분류 2차 키워드 (비활성화, aspect_probs=None → p_none은 fallback 모드)
        new_preds = ABSAInference._apply_secondary_rescue(runner, texts, new_preds, None, ratings)
        # P2: 접속사 감성 보정 (비활성화)
        new_preds = ABSAInference._apply_contrast_correction(runner, texts, new_preds)

    # 변경 분석
    changed_mask = (old_preds != new_preds).any(axis=1)
    num_changed = changed_mask.sum()

    print(f"\n{'=' * 70}")
    print(f"변경 결과")
    print(f"{'=' * 70}")
    print(f"총 변경: {num_changed:,}건 ({num_changed / len(df) * 100:.2f}%)")

    # aspect별 변경 상세
    print(f"\nAspect별 변경:")
    for j, name in enumerate(aspect_labels):
        asp_changed = (old_preds[:, j] != new_preds[:, j]).sum()
        if asp_changed > 0:
            none_to_active = ((old_preds[:, j] == 0) & (new_preds[:, j] != 0)).sum()
            active_to_none = ((old_preds[:, j] != 0) & (new_preds[:, j] == 0)).sum()
            print(f"  {name:<14}: {asp_changed:>5}건 "
                  f"(none→활성: {none_to_active}, 활성→none: {active_to_none})")

    # 미분류 비율 변화
    old_unclassified = (old_preds.max(axis=1) == 0).sum()
    new_unclassified = (new_preds.max(axis=1) == 0).sum()
    print(f"\n미분류 비율: {old_unclassified:,} ({old_unclassified/len(df)*100:.1f}%) "
          f"→ {new_unclassified:,} ({new_unclassified/len(df)*100:.1f}%)")

    if args.dry_run:
        print(f"\n[dry-run] 파일 저장 없이 종료")
        return

    # 결과 적용
    new_asp_sents = preds_to_aspect_sentiments(new_preds, aspect_labels)
    df["aspect_sentiments"] = new_asp_sents
    df["aspect_labels"] = [
        [a["aspect"] for a in aspects] for aspects in new_asp_sents
    ]
    df["summary"] = df.apply(
        lambda row: generate_summary(row["sentiment"], row["aspect_sentiments"]),
        axis=1,
    )

    # ambiguous 강화 (별점-감성 불일치)
    cfg = QC_POSTPROCESS_CONFIG.get("ambiguous_enhancement", {})
    if cfg.get("enabled", False) and ratings is not None:
        sent_to_id = {"negative": 0, "neutral": 1, "positive": 2}
        rating_sent_map = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
        amb_count = 0
        for i in range(len(df)):
            if df.iloc[i]["is_ambiguous"]:
                continue
            r = ratings[i]
            if np.isnan(r):
                continue
            expected = rating_sent_map.get(int(r))
            actual = sent_to_id.get(df.iloc[i]["sentiment"], -1)
            if expected is not None and expected != actual:
                df.iat[i, df.columns.get_loc("is_ambiguous")] = True
                amb_count += 1
        if amb_count > 0:
            print(f"  Ambiguous enhancement: {amb_count}건 추가 플래그")

    # 임시 컬럼 제거
    df.drop(columns=["_parsed_asp"], inplace=True)

    # 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {output_path}")


if __name__ == "__main__":
    main()
