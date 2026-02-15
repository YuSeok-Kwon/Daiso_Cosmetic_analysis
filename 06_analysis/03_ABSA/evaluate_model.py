"""
Golden Set을 사용한 ABSA 모델 평가
"""
import pandas as pd
import sys
import os
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv(Path(__file__).parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))
from openai_client import OpenAIClient


def evaluate_model(
    golden_set_path: str,
    model: str = "gpt-4o-mini",
    sample_size: int = None,
    verbose: bool = True
) -> dict:
    """
    Golden Set으로 모델 평가

    Args:
        golden_set_path: Golden Set 파일 경로 (xlsx)
        model: 평가할 모델
        sample_size: 샘플 수 (None이면 전체)
        verbose: 상세 출력 여부

    Returns:
        평가 결과 딕셔너리
    """
    # Golden Set 로드
    golden_df = pd.read_excel(golden_set_path)

    if sample_size:
        golden_df = golden_df.head(sample_size)

    print(f"=== 모델 평가 시작 ===")
    print(f"모델: {model}")
    print(f"평가 샘플 수: {len(golden_df)}")
    print(f"예상 비용: ${len(golden_df) * 0.0003:.2f} ~ ${len(golden_df) * 0.0005:.2f}")
    print()

    # OpenAI 클라이언트
    client = OpenAIClient()

    # 결과 저장
    results = []
    aspect_correct = 0
    sentiment_correct = 0
    both_correct = 0
    total = 0

    # Aspect별 성능
    aspect_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    sentiment_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    # 오류 케이스 수집
    errors = []

    for idx, row in tqdm(golden_df.iterrows(), total=len(golden_df), desc="평가 중"):
        try:
            # 모델 호출
            result = client.label_review(
                review_text=row['text'],
                product_code=0,  # Golden Set에 없음
                rating=row['rating'],
                name="",  # Golden Set에 없음
                category_1="",
                category_2="",
                model=model
            )

            # 정답
            true_aspect = row['corrected_aspect']
            true_sentiment = row['corrected_sentiment'].lower()

            # 예측 (aspect_labels는 딕셔너리 리스트 또는 문자열 리스트)
            pred_aspects_raw = result.aspect_labels

            # aspect 이름 추출 및 해당 aspect의 sentiment 찾기
            pred_aspect_names = []
            pred_sentiment_for_aspect = None

            for item in pred_aspects_raw:
                if isinstance(item, dict):
                    aspect_name = item.get('aspect', '')
                    pred_aspect_names.append(aspect_name)
                    if aspect_name == true_aspect:
                        pred_sentiment_for_aspect = item.get('sentiment', '').lower()
                else:
                    pred_aspect_names.append(item)

            # Aspect 평가 (예측된 aspect 중 정답이 있는지)
            aspect_match = true_aspect in pred_aspect_names

            # Sentiment 평가 (해당 aspect의 sentiment와 비교)
            if pred_sentiment_for_aspect:
                sentiment_match = pred_sentiment_for_aspect == true_sentiment
            else:
                # 해당 aspect가 없으면 전체 sentiment와 비교
                sentiment_match = result.sentiment.lower() == true_sentiment

            pred_aspects = pred_aspects_raw  # 출력용
            pred_sentiment = pred_sentiment_for_aspect if pred_sentiment_for_aspect else result.sentiment.lower()

            # 통계 업데이트
            total += 1
            if aspect_match:
                aspect_correct += 1
            if sentiment_match:
                sentiment_correct += 1
            if aspect_match and sentiment_match:
                both_correct += 1

            # Aspect별 통계
            aspect_stats[true_aspect]["total"] += 1
            if aspect_match:
                aspect_stats[true_aspect]["correct"] += 1

            # Sentiment별 통계
            sentiment_stats[true_sentiment]["total"] += 1
            if sentiment_match:
                sentiment_stats[true_sentiment]["correct"] += 1

            # 오류 케이스 기록
            if not (aspect_match and sentiment_match):
                errors.append({
                    "text": row['text'][:50] + "...",
                    "true_aspect": true_aspect,
                    "pred_aspects": pred_aspects,
                    "aspect_match": aspect_match,
                    "true_sentiment": true_sentiment,
                    "pred_sentiment": pred_sentiment,
                    "sentiment_match": sentiment_match
                })

            results.append({
                "text": row['text'],
                "true_aspect": true_aspect,
                "true_sentiment": true_sentiment,
                "pred_aspects": pred_aspects,
                "pred_sentiment": pred_sentiment,
                "aspect_match": aspect_match,
                "sentiment_match": sentiment_match
            })

        except Exception as e:
            print(f"\n오류 (행 {idx}): {e}")
            continue

    # 결과 계산
    aspect_accuracy = aspect_correct / total if total > 0 else 0
    sentiment_accuracy = sentiment_correct / total if total > 0 else 0
    both_accuracy = both_correct / total if total > 0 else 0

    # 비용 계산
    total_cost = client.get_total_cost()

    # 결과 출력
    print("\n" + "=" * 60)
    print("평가 결과")
    print("=" * 60)
    print(f"\n총 평가 샘플: {total}")
    print(f"총 비용: ${total_cost:.4f}")
    print(f"\n[전체 정확도]")
    print(f"  Aspect 정확도: {aspect_accuracy*100:.1f}% ({aspect_correct}/{total})")
    print(f"  Sentiment 정확도: {sentiment_accuracy*100:.1f}% ({sentiment_correct}/{total})")
    print(f"  Both 정확도: {both_accuracy*100:.1f}% ({both_correct}/{total})")

    print(f"\n[Aspect별 정확도]")
    for aspect, stats in sorted(aspect_stats.items(), key=lambda x: x[1]["total"], reverse=True):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {aspect}: {acc*100:.1f}% ({stats['correct']}/{stats['total']})")

    print(f"\n[Sentiment별 정확도]")
    for sentiment, stats in sorted(sentiment_stats.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {sentiment}: {acc*100:.1f}% ({stats['correct']}/{stats['total']})")

    if verbose and errors:
        print(f"\n[오류 케이스 샘플 (최대 10개)]")
        for err in errors[:10]:
            print(f"\n  텍스트: {err['text']}")
            print(f"    Aspect: {err['true_aspect']} → {err['pred_aspects']} ({'✓' if err['aspect_match'] else '✗'})")
            print(f"    Sentiment: {err['true_sentiment']} → {err['pred_sentiment']} ({'✓' if err['sentiment_match'] else '✗'})")

    print("=" * 60)

    return {
        "total": total,
        "aspect_accuracy": aspect_accuracy,
        "sentiment_accuracy": sentiment_accuracy,
        "both_accuracy": both_accuracy,
        "aspect_stats": dict(aspect_stats),
        "sentiment_stats": dict(sentiment_stats),
        "results": results,
        "errors": errors,
        "cost": total_cost
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ABSA 모델 평가")
    parser.add_argument("--golden-set", type=str, default="data/golden_set.xlsx",
                        help="Golden Set 파일 경로")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="평가할 모델")
    parser.add_argument("--sample", type=int, default=None,
                        help="샘플 수 (기본: 전체)")
    parser.add_argument("--quiet", action="store_true",
                        help="상세 출력 비활성화")

    args = parser.parse_args()

    evaluate_model(
        golden_set_path=args.golden_set,
        model=args.model,
        sample_size=args.sample,
        verbose=not args.quiet
    )
