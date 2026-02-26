"""
Inference pipeline for ABSA model (Option A: aspect별 4-class 통합)

출력 형식:
{
    "review_sentiment": "positive",
    "review_sentiment_score": 0.85,
    "aspect_sentiments": [
        {"aspect": "사용감/성능", "sentiment": "positive", "confidence": 0.92},
        {"aspect": "재질/냄새", "sentiment": "negative", "confidence": 0.78}
    ]
}
"""
import re
import json
import hashlib
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional

from RQ_absa.s5_model import MultiTaskABSAModel, get_best_device
from RQ_absa.s1_config import (
    ASPECT_LABELS,
    SENTIMENT_ID_TO_LABEL,
    ASPECT_SENTIMENT_ID_TO_LABEL,
    KEYWORD_GATE_CONFIG,
    KEYWORD_FORCE_ON_CONFIG,
)
from RQ_absa.s7_evaluation import (
    apply_none_thresholds,
    apply_thresholds_with_polar,
    apply_design_rule_override,
)


class ABSAInference:
    """
    Inference pipeline for ABSA model (Option A).
    """

    def __init__(
        self,
        model: MultiTaskABSAModel,
        tokenizer,
        aspect_labels: List[str] = None,
        sentiment_labels: Dict[int, str] = None,
        aspect_sentiment_labels: Dict[int, str] = None,
        device: str = None,
        max_length: int = 128,
        batch_size: int = 128,
        ambiguous_sentiment_threshold: float = 0.6,
        none_thresholds: np.ndarray = None,
        polar_threshold: float = None,
        use_design_rule: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.ambiguous_sentiment_threshold = ambiguous_sentiment_threshold
        self.none_thresholds = none_thresholds  # [K] per-aspect threshold
        self.polar_threshold = polar_threshold  # polar 확신 기준 (neutral 복원용)
        self.use_design_rule = use_design_rule  # Design Rule Override 사용 여부

        if device is None:
            self.device = torch.device(get_best_device())
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

        # aspect_labels 우선순위: 명시 전달 > checkpoint(model.aspect_labels) > config
        self.aspect_labels = (
            aspect_labels
            or getattr(model, "aspect_labels", None)
            or list(ASPECT_LABELS)
        )
        self.sentiment_labels = sentiment_labels or SENTIMENT_ID_TO_LABEL
        self.aspect_sentiment_labels = aspect_sentiment_labels or ASPECT_SENTIMENT_ID_TO_LABEL

        if self.none_thresholds is not None:
            print(f"Using per-aspect none-thresholds: "
                  f"min={self.none_thresholds.min():.2f}, "
                  f"max={self.none_thresholds.max():.2f}, "
                  f"mean={self.none_thresholds.mean():.2f}")
            if self.polar_threshold is not None:
                print(f"Using polar_threshold={self.polar_threshold:.2f} (neutral 복원)")
        else:
            print("Using default argmax (no threshold tuning)")

        if self.use_design_rule:
            print("Using design rule override (디자인 aspect → 키워드 규칙)")

        print(f"Inference initialized on device: {self.device}")

    @classmethod
    def from_bundle(
        cls,
        bundle_path: str,
        model_name: str = "beomi/KcELECTRA-base",
        batch_size: int = 128,
        verify_integrity: bool = True,
    ) -> "ABSAInference":
        """운영 번들에서 모델 + threshold + 설정을 한 번에 로드.

        Args:
            bundle_path: prod_bundle 디렉토리 경로
            model_name: HuggingFace 모델 이름 (토크나이저용)
            batch_size: 추론 배치 크기
            verify_integrity: True이면 MD5 체크섬 검증

        Returns:
            ABSAInference 인스턴스 (운영 세팅 완전 적용)
        """
        from transformers import AutoTokenizer
        from RQ_absa.s5_model import load_model

        bundle = Path(bundle_path)
        manifest_path = bundle / "MANIFEST.json"

        if not manifest_path.exists():
            raise FileNotFoundError(f"MANIFEST.json이 없습니다: {bundle}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        print(f"=== 운영 번들 로드: {manifest['bundle_name']} ===")
        print(f"  Stage: {manifest['stage']}, Created: {manifest['created_at']}")

        # (1) MD5 무결성 검증
        if verify_integrity:
            print("  무결성 검증 중...")
            for fname, info in manifest["files"].items():
                fpath = bundle / fname
                if not fpath.exists():
                    raise FileNotFoundError(f"번들 파일 누락: {fpath}")
                if "md5" in info:
                    actual_md5 = hashlib.md5(fpath.read_bytes()).hexdigest()
                    if actual_md5 != info["md5"]:
                        raise ValueError(
                            f"MD5 불일치: {fname}\n"
                            f"  예상: {info['md5']}\n  실제: {actual_md5}"
                        )
            print("  무결성 검증 OK")

        # (2) 모델 파일 찾기 (best_model*.pt)
        model_files = list(bundle.glob("best_model*.pt"))
        if not model_files:
            raise FileNotFoundError(f"모델 파일(best_model*.pt)이 없습니다: {bundle}")
        model_path = model_files[0]

        # (3) 모델 + 토크나이저 로드
        print(f"  모델 로드: {model_path.name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = load_model(checkpoint_path=model_path, model_name=model_name)

        # (4) Threshold 로드
        threshold_path = bundle / "none_thresholds_stage3a.json"
        if not threshold_path.exists():
            threshold_path = list(bundle.glob("none_thresholds*.json"))
            threshold_path = threshold_path[0] if threshold_path else None

        none_thresholds = None
        polar_threshold = None
        if threshold_path and Path(threshold_path).exists():
            with open(threshold_path, "r", encoding="utf-8") as f:
                tdata = json.load(f)
            none_thresholds = np.array(tdata["thresholds"])
            polar_threshold = tdata.get("polar_threshold")

        # (5) Design Rule Config 로드 → 런타임 override
        use_design_rule = False
        drule_path = bundle / "design_rule_config.json"
        if drule_path.exists():
            with open(drule_path, "r", encoding="utf-8") as f:
                drule_data = json.load(f)
            import RQ_absa.s1_config as cfg
            if "design_rule_config" in drule_data:
                cfg.DESIGN_RULE_CONFIG = drule_data["design_rule_config"]
                print("  Design Rule Config: 번들에서 로드 완료")
            if "keyword_gate_config" in drule_data:
                cfg.KEYWORD_GATE_CONFIG = drule_data["keyword_gate_config"]
                print("  Keyword Gate Config: 번들에서 로드 완료")
            if "keyword_force_on_config" in drule_data:
                cfg.KEYWORD_FORCE_ON_CONFIG = drule_data["keyword_force_on_config"]
                print("  Keyword Force-On Config: 번들에서 로드 완료")
            # threshold JSON의 design_rule 플래그도 확인
            if threshold_path and Path(threshold_path).exists():
                use_design_rule = tdata.get("design_rule", False)
            else:
                use_design_rule = True  # design_rule_config.json 존재 = 사용

        print(f"  GO 기준 검증: {manifest.get('golden_test_performance', {}).get('go_decision', 'N/A')}")
        print("=== 번들 로드 완료 ===\n")

        return cls(
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            none_thresholds=none_thresholds,
            polar_threshold=polar_threshold,
            use_design_rule=use_design_rule,
        )

    def predict_batch(self, texts: List[str]) -> Dict:
        """
        배치 추론.

        Returns:
            sentiment_preds: [B]
            sentiment_probs: [B, 3]
            sentiment_scores: [B]
            sentiment_confidence: [B]
            aspect_preds: [B, 11] (각 값 0~3)
            aspect_probs: [B, 11, 4]
        """
        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        with torch.inference_mode():
            outputs = self.model(input_ids, attention_mask)

            # Sentiment
            sentiment_probs = torch.softmax(outputs["sentiment_logits"], dim=-1)
            sentiment_preds = torch.argmax(sentiment_probs, dim=-1)
            sentiment_scores = self.model.get_sentiment_score(sentiment_probs)
            sentiment_confidence = torch.max(sentiment_probs, dim=-1)[0]

            # Aspect: [B, 11, 4] → softmax
            aspect_probs = torch.softmax(outputs["aspect_logits"], dim=-1)  # [B, 11, 4]
            aspect_probs_np = aspect_probs.cpu().numpy()

            # Per-aspect threshold 적용 (polar threshold가 있으면 3단계 threshold)
            if self.none_thresholds is not None and self.polar_threshold is not None:
                aspect_preds_np = apply_thresholds_with_polar(
                    aspect_probs_np, self.none_thresholds, self.polar_threshold
                )
            elif self.none_thresholds is not None:
                aspect_preds_np = apply_none_thresholds(aspect_probs_np, self.none_thresholds)
            else:
                aspect_preds_np = np.argmax(aspect_probs_np, axis=-1)

            return {
                "sentiment_preds": sentiment_preds.cpu().numpy(),
                "sentiment_probs": sentiment_probs.cpu().numpy(),
                "sentiment_scores": sentiment_scores.cpu().numpy(),
                "sentiment_confidence": sentiment_confidence.cpu().numpy(),
                "aspect_preds": aspect_preds_np,
                "aspect_probs": aspect_probs_np,
            }

    def _apply_keyword_gate(
        self, texts: List[str], aspect_preds: np.ndarray
    ) -> np.ndarray:
        """키워드 게이트: non-none 예측이지만 키워드 미포함 시 none(0)으로 override.

        Stage 4C: 정규표현식 기반 매칭으로 전환.
        - 활용형 제한, lookahead/lookbehind, 경계 처리로 오탐 최소화
        - KEYWORD_GATE_CONFIG 값이 정규표현식 패턴 리스트

        Args:
            texts: [N] 리뷰 텍스트 리스트
            aspect_preds: [N, K] aspect 예측 (0=none, 1=pos, 2=neu, 3=neg)

        Returns:
            aspect_preds: [N, K] 키워드 게이트 적용 후
        """
        gated = aspect_preds.copy()
        gate_count = 0

        # aspect별 정규식 패턴을 사전 컴파일 (성능 최적화)
        compiled_gates = {}
        for aspect_name, patterns in KEYWORD_GATE_CONFIG.items():
            if aspect_name not in self.aspect_labels:
                continue
            j = self.aspect_labels.index(aspect_name)
            compiled_gates[aspect_name] = (
                j,
                [re.compile(p, re.IGNORECASE) for p in patterns],
            )

        for aspect_name, (j, compiled_patterns) in compiled_gates.items():
            for i, text in enumerate(texts):
                if gated[i, j] == 0:  # 이미 none이면 스킵
                    continue
                text_str = str(text)
                if not any(p.search(text_str) for p in compiled_patterns):
                    gated[i, j] = 0  # 키워드 미포함 → none으로 override
                    gate_count += 1

        if gate_count > 0:
            print(f"  Keyword gate (regex): {gate_count}건 override (non-none → none)")

        return gated

    def _apply_keyword_force_on(
        self, texts: List[str], aspect_preds: np.ndarray
    ) -> np.ndarray:
        """키워드 포함 시 해당 aspect를 강제 활성화 (FN 보정)."""
        gated = aspect_preds.copy()
        force_count = 0
        for aspect_name, config in KEYWORD_FORCE_ON_CONFIG.items():
            if aspect_name not in self.aspect_labels:
                continue
            j = self.aspect_labels.index(aspect_name)
            for i, text in enumerate(texts):
                if gated[i, j] != 0:  # 이미 활성화면 스킵
                    continue
                if any(kw in str(text) for kw in config["keywords"]):
                    gated[i, j] = config["sentiment"]
                    force_count += 1
        if force_count > 0:
            print(f"  Keyword force-on: {force_count}건 활성화")
        return gated

    def _extract_aspect_sentiments(
        self, aspect_preds: np.ndarray, aspect_probs: np.ndarray
    ) -> List[List[Dict]]:
        """
        aspect_preds [N, K]에서 none(0)이 아닌 것만 추출하여
        [{aspect, sentiment, confidence}, ...] 형태로 반환.

        미분류 후처리: 10개 aspect 모두 none이면 "미분류/neutral"을 파생 생성.
        """
        results = []
        for i in range(len(aspect_preds)):
            review_aspects = []
            for j, aspect_name in enumerate(self.aspect_labels):
                pred_id = int(aspect_preds[i, j])
                if pred_id == 0:  # none → 해당 aspect 미존재
                    continue

                sentiment_name = self.aspect_sentiment_labels.get(pred_id, "unknown")
                confidence = float(aspect_probs[i, j, pred_id])

                review_aspects.append({
                    "aspect": aspect_name,
                    "sentiment": sentiment_name,
                    "confidence": confidence,
                })

            # 미분류 후처리: 모든 aspect가 none이면 "미분류" 파생
            if not review_aspects:
                review_aspects.append({
                    "aspect": "미분류",
                    "sentiment": "neutral",
                    "confidence": 0.0,
                })

            results.append(review_aspects)
        return results

    def infer_dataframe(
        self, df: pd.DataFrame, text_column: str = "text"
    ) -> pd.DataFrame:
        """DataFrame 전체 추론"""
        print(f"Running inference on {len(df):,} reviews...")

        texts = df[text_column].fillna("").astype(str).tolist()

        all_sentiment_preds = []
        all_sentiment_scores = []
        all_sentiment_confidence = []
        all_aspect_preds = []
        all_aspect_probs = []

        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in tqdm(range(num_batches), desc="Inference"):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, len(texts))
            batch_texts = texts[start:end]

            predictions = self.predict_batch(batch_texts)

            all_sentiment_preds.extend(predictions["sentiment_preds"])
            all_sentiment_scores.extend(predictions["sentiment_scores"])
            all_sentiment_confidence.extend(predictions["sentiment_confidence"])
            all_aspect_preds.extend(predictions["aspect_preds"])
            all_aspect_probs.extend(predictions["aspect_probs"])

        all_sentiment_preds = np.array(all_sentiment_preds)
        all_sentiment_scores = np.array(all_sentiment_scores)
        all_sentiment_confidence = np.array(all_sentiment_confidence)
        all_aspect_preds = np.array(all_aspect_preds)
        all_aspect_probs = np.array(all_aspect_probs)

        # Design Rule Override (디자인 aspect → 키워드 규칙으로 완전 대체)
        if self.use_design_rule:
            all_aspect_preds = apply_design_rule_override(
                texts, all_aspect_preds, aspect_labels_list=self.aspect_labels
            )

        # 키워드 게이트 적용 (non-none이지만 키워드 미포함 → none override)
        if KEYWORD_GATE_CONFIG:
            all_aspect_preds = self._apply_keyword_gate(texts, all_aspect_preds)

        # 키워드 강제 ON (FN 보정: 키워드 포함인데 none인 경우 활성화)
        if KEYWORD_FORCE_ON_CONFIG:
            all_aspect_preds = self._apply_keyword_force_on(texts, all_aspect_preds)

        # aspect별 sentiment 추출 (none 제외)
        aspect_sentiments = self._extract_aspect_sentiments(all_aspect_preds, all_aspect_probs)

        # 출력 DataFrame 생성
        output_df = df.copy()
        output_df["sentiment"] = [self.sentiment_labels[p] for p in all_sentiment_preds]
        output_df["sentiment_score"] = all_sentiment_scores

        # aspect_sentiments: [{aspect, sentiment, confidence}, ...]
        output_df["aspect_sentiments"] = aspect_sentiments

        # 하위 호환: aspect_labels (이름 리스트)
        output_df["aspect_labels"] = [
            [a["aspect"] for a in aspects] for aspects in aspect_sentiments
        ]

        # Summary
        output_df["summary"] = output_df.apply(
            lambda row: self._generate_summary(row["sentiment"], row["aspect_sentiments"]),
            axis=1
        )

        # Ambiguous 식별
        output_df["is_ambiguous"] = self._identify_ambiguous(
            all_sentiment_confidence, all_aspect_probs, all_aspect_preds
        )

        print(f"\nInference complete!")
        print(f"Ambiguous samples: {output_df['is_ambiguous'].sum():,} "
              f"({output_df['is_ambiguous'].sum() / len(output_df) * 100:.1f}%)")

        return output_df

    def _generate_summary(self, sentiment: str, aspect_sentiments: List[Dict]) -> str:
        """감성별 aspect 그룹핑 요약"""
        if not aspect_sentiments:
            return f"전반적으로 {sentiment}"

        sentiment_kr = {
            "positive": "긍정적", "neutral": "중립적", "negative": "부정적"
        }

        # 감성별 그룹핑
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

    def _identify_ambiguous(
        self,
        sentiment_confidence: np.ndarray,
        aspect_probs: np.ndarray,
        aspect_preds: np.ndarray
    ) -> np.ndarray:
        """
        모호한 샘플 식별:
        - 감성 confidence 낮음
        - aspect 예측에서 top-1과 top-2 확률 차이가 작음
        """
        low_sent_conf = sentiment_confidence < self.ambiguous_sentiment_threshold

        # non-none으로 예측된 aspect에서만 margin 확인 (none은 대부분이므로 무시)
        sorted_probs = np.sort(aspect_probs, axis=-1)  # [N, 11, 4]
        top1 = sorted_probs[:, :, -1]
        top2 = sorted_probs[:, :, -2]
        close_margin_per_aspect = (top1 - top2) < 0.2  # [N, 11]
        is_non_none = (aspect_preds != 0)  # [N, 11]
        close_margin = (close_margin_per_aspect & is_non_none).any(axis=1)  # [N]

        return low_sent_conf | close_margin

    def save_results(
        self,
        df: pd.DataFrame,
        output_path: Path,
        save_ambiguous: bool = True,
        ambiguous_path: Path = None
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Saved full results to: {output_path}")

        if save_ambiguous:
            ambiguous_df = df[df["is_ambiguous"]].copy()
            if len(ambiguous_df) > 0:
                if ambiguous_path is None:
                    ambiguous_path = output_path.parent / f"{output_path.stem}_ambiguous.csv"
                ambiguous_df.to_csv(ambiguous_path, index=False, encoding="utf-8-sig")
                print(f"Saved ambiguous samples to: {ambiguous_path}")
                print(f"  Count: {len(ambiguous_df):,}")


def _load_none_thresholds(model_path: Path, model=None) -> tuple:
    """모델 체크포인트와 같은 디렉토리에서 none_thresholds.json 로드.

    Args:
        model_path: 체크포인트 경로
        model: 로드된 모델 (aspect 수 검증용, None이면 config 기준)

    Returns:
        (none_thresholds, polar_threshold) — polar_threshold는 없으면 None

    Raises:
        ValueError: threshold 길이가 모델의 aspect 수와 불일치할 때
    """
    import json
    threshold_path = Path(model_path).parent / "none_thresholds.json"
    if threshold_path.exists():
        with open(threshold_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        thresholds = np.array(data["thresholds"])
        polar_threshold = data.get("polar_threshold")

        # 길이 K 검증: model.aspect_labels > config ASPECT_LABELS
        model_aspects = getattr(model, "aspect_labels", None) if model else None
        expected_k = len(model_aspects) if model_aspects else len(ASPECT_LABELS)
        if len(thresholds) != expected_k:
            raise ValueError(
                f"Threshold 길이 불일치: json={len(thresholds)}, "
                f"model aspects={expected_k}. "
                f"체크포인트와 threshold의 aspect 수가 다릅니다. "
                f"threshold를 재튜닝하거나 config를 확인하세요."
            )

        print(f"Loaded none-thresholds from: {threshold_path}")
        print(f"  K={len(thresholds)}, Tuned F1: {data.get('tuned_f1', 'N/A')}")
        if polar_threshold is not None:
            print(f"  Polar threshold: {polar_threshold}")
        return thresholds, polar_threshold
    print("No none_thresholds.json found, using default argmax")
    return None, None


def run_inference_on_reviews(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    model_name: str = "beomi/KcELECTRA-base",
    batch_size: int = 128,
    chunk_size: int = 10000,
) -> pd.DataFrame:
    """
    리뷰 CSV에 대해 추론 실행.

    chunk_size 단위로 읽고 처리하여 streaming CSV 저장 (메모리 폭발 방지).
    """
    from transformers import AutoTokenizer
    from RQ_absa.s5_model import load_model

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = load_model(checkpoint_path=model_path, model_name=model_name)

    none_thresholds, polar_threshold = _load_none_thresholds(model_path, model=model)

    inference = ABSAInference(
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        none_thresholds=none_thresholds,
        polar_threshold=polar_threshold,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    total_ambiguous = 0
    header_written = False

    print(f"\nStreaming inference (chunk_size={chunk_size:,})...")

    for chunk_df in pd.read_csv(input_path, chunksize=chunk_size):
        chunk_result = inference.infer_dataframe(chunk_df)

        # CSV에 append 모드로 저장
        chunk_result.to_csv(
            output_path,
            mode="a" if header_written else "w",
            header=not header_written,
            index=False,
            encoding="utf-8-sig"
        )
        header_written = True

        total_rows += len(chunk_result)
        total_ambiguous += chunk_result["is_ambiguous"].sum()
        print(f"  누적: {total_rows:,} / ambiguous: {total_ambiguous:,}")

    print(f"\nSaved streaming results to: {output_path}")
    print(f"Total: {total_rows:,} reviews, ambiguous: {total_ambiguous:,} "
          f"({total_ambiguous / total_rows * 100:.1f}%)")

    # 최종 통계용으로 결과 로드 (선택적)
    results_df = pd.read_csv(output_path)

    print("\n" + "=" * 60)
    print("INFERENCE STATISTICS")
    print("=" * 60)
    print("\nSentiment distribution:")
    print(results_df["sentiment"].value_counts(normalize=True).sort_index())
    print("=" * 60)

    return results_df


def run_inference_from_bigquery(
    model_path: Path,
    model_name: str = "beomi/KcELECTRA-base",
    batch_size: int = 128,
    limit: int = None,
    save_to_bq: bool = True,
    output_csv: Path = None
) -> pd.DataFrame:
    """BigQuery에서 리뷰를 로드하여 추론 실행"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    try:
        from bq_connector import ABSABigQuery
    except ImportError:
        print("Error: bq_connector 모듈을 찾을 수 없습니다.")
        return None

    from transformers import AutoTokenizer
    from RQ_absa.s5_model import load_model

    bq = ABSABigQuery()

    print("BigQuery에서 미분석 리뷰 로드 중...")
    df = bq.load_unanalyzed_reviews(limit=limit)

    if len(df) == 0:
        print("분석할 리뷰가 없습니다.")
        return pd.DataFrame()

    print(f"총 {len(df):,}개 리뷰 로드 완료")

    print("\n모델 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = load_model(checkpoint_path=model_path, model_name=model_name)

    none_thresholds, polar_threshold = _load_none_thresholds(model_path, model=model)

    inference = ABSAInference(
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        none_thresholds=none_thresholds,
        polar_threshold=polar_threshold,
    )

    print("\n추론 실행 중...")
    results_df = inference.infer_dataframe(df)

    print("\n" + "=" * 60)
    print("INFERENCE STATISTICS")
    print("=" * 60)
    print("\nSentiment distribution:")
    print(results_df["sentiment"].value_counts(normalize=True).sort_index())

    if save_to_bq:
        print("\nBigQuery에 결과 저장 중...")
        bq.update_review_analysis(results_df)
        print("저장 완료!")

    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"CSV 저장: {output_csv}")

    print("=" * 60)

    return results_df
