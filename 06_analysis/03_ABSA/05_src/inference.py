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
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

from RQ_absa.model import MultiTaskABSAModel
from RQ_absa.config import (
    ASPECT_LABELS,
    SENTIMENT_ID_TO_LABEL,
    ASPECT_SENTIMENT_ID_TO_LABEL,
)
from RQ_absa.evaluation import apply_none_thresholds


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
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.ambiguous_sentiment_threshold = ambiguous_sentiment_threshold
        self.none_thresholds = none_thresholds  # [11] per-aspect threshold

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

        self.aspect_labels = aspect_labels or ASPECT_LABELS
        self.sentiment_labels = sentiment_labels or SENTIMENT_ID_TO_LABEL
        self.aspect_sentiment_labels = aspect_sentiment_labels or ASPECT_SENTIMENT_ID_TO_LABEL

        if self.none_thresholds is not None:
            print(f"Using per-aspect none-thresholds: "
                  f"min={self.none_thresholds.min():.2f}, "
                  f"max={self.none_thresholds.max():.2f}, "
                  f"mean={self.none_thresholds.mean():.2f}")
        else:
            print("Using default argmax (no threshold tuning)")

        print(f"Inference initialized on device: {self.device}")

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

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)

            # Sentiment
            sentiment_probs = torch.softmax(outputs["sentiment_logits"], dim=-1)
            sentiment_preds = torch.argmax(sentiment_probs, dim=-1)
            sentiment_scores = self.model.get_sentiment_score(sentiment_probs)
            sentiment_confidence = torch.max(sentiment_probs, dim=-1)[0]

            # Aspect: [B, 11, 4] → softmax
            aspect_probs = torch.softmax(outputs["aspect_logits"], dim=-1)  # [B, 11, 4]
            aspect_probs_np = aspect_probs.cpu().numpy()

            # Per-aspect threshold 적용
            if self.none_thresholds is not None:
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

    def _extract_aspect_sentiments(
        self, aspect_preds: np.ndarray, aspect_probs: np.ndarray
    ) -> List[List[Dict]]:
        """
        aspect_preds [N, 11]에서 none(0)이 아닌 것만 추출하여
        [{aspect, sentiment, confidence}, ...] 형태로 반환.
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
            results.append(review_aspects)
        return results

    def infer_dataframe(
        self, df: pd.DataFrame, text_column: str = "text"
    ) -> pd.DataFrame:
        """DataFrame 전체 추론"""
        print(f"Running inference on {len(df):,} reviews...")

        texts = df[text_column].astype(str).tolist()

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

        # aspect별 top-1 vs top-2 확률 차이가 0.2 미만인 aspect가 있으면 ambiguous
        sorted_probs = np.sort(aspect_probs, axis=-1)  # [N, 11, 4]
        top1 = sorted_probs[:, :, -1]
        top2 = sorted_probs[:, :, -2]
        close_margin = ((top1 - top2) < 0.2).any(axis=1)  # [N]

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


def _load_none_thresholds(model_path: Path) -> np.ndarray:
    """모델 체크포인트와 같은 디렉토리에서 none_thresholds.json 로드"""
    import json
    threshold_path = Path(model_path).parent / "none_thresholds.json"
    if threshold_path.exists():
        with open(threshold_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        thresholds = np.array(data["thresholds"])
        print(f"Loaded none-thresholds from: {threshold_path}")
        print(f"  Tuned F1: {data.get('tuned_f1', 'N/A')}")
        return thresholds
    print("No none_thresholds.json found, using default argmax")
    return None


def run_inference_on_reviews(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    model_name: str = "beomi/KcELECTRA-base",
    batch_size: int = 128,
) -> pd.DataFrame:
    """리뷰 CSV에 대해 추론 실행"""
    from transformers import AutoTokenizer
    from RQ_absa.model import load_model

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = load_model(checkpoint_path=model_path, model_name=model_name)

    # Threshold 자동 로드
    none_thresholds = _load_none_thresholds(model_path)

    print("\nLoading reviews...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} reviews")

    inference = ABSAInference(
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        none_thresholds=none_thresholds,
    )

    results_df = inference.infer_dataframe(df)
    inference.save_results(results_df, output_path)

    # 통계
    print("\n" + "=" * 60)
    print("INFERENCE STATISTICS")
    print("=" * 60)
    print("\nSentiment distribution:")
    print(results_df["sentiment"].value_counts(normalize=True).sort_index())

    print("\nAspect-Sentiment frequency:")
    all_aspects = []
    for aspects in results_df["aspect_sentiments"]:
        for a in aspects:
            all_aspects.append(f"{a['aspect']}:{a['sentiment']}")
    if all_aspects:
        aspect_counts = pd.Series(all_aspects).value_counts()
        for combo, count in aspect_counts.head(20).items():
            print(f"  {combo}: {count:,} ({count / len(results_df) * 100:.1f}%)")

    print("\nAspects per review:")
    results_df["num_aspects"] = results_df["aspect_sentiments"].apply(len)
    print(results_df["num_aspects"].describe())

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
    from RQ_absa.model import load_model

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

    none_thresholds = _load_none_thresholds(model_path)

    inference = ABSAInference(
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        none_thresholds=none_thresholds,
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
