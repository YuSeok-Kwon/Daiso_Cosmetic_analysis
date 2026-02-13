"""
ABSA BigQuery 연동 실행 스크립트

사용법:
    # 분석 현황 확인
    python run_absa_bq.py --stats

    # BigQuery에서 리뷰 로드 → CSV 내보내기
    python run_absa_bq.py --export --limit 10000 --output data/reviews_for_labeling.csv

    # GPT 라벨링 (BigQuery + CSV 저장)
    python run_absa_bq.py --label --limit 1000 --output data/labeled.jsonl --save-csv data/labeled.csv

    # 모델 추론 (BigQuery + CSV 저장)
    python run_absa_bq.py --infer --model models/best_model.pt --limit 10000 --save-csv data/inference_results.csv

    # BigQuery 저장 안하고 CSV만 저장
    python run_absa_bq.py --infer --model models/best_model.pt --no-save-bq --save-csv data/results.csv
"""
import argparse
from pathlib import Path
from datetime import datetime


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    parser = argparse.ArgumentParser(description='ABSA BigQuery 연동 실행')
    parser.add_argument('--stats', action='store_true', help='분석 현황 통계 출력')
    parser.add_argument('--export', action='store_true', help='리뷰 CSV 내보내기')
    parser.add_argument('--label', action='store_true', help='GPT 라벨링 실행')
    parser.add_argument('--infer', action='store_true', help='모델 추론 실행')
    parser.add_argument('--limit', type=int, default=None, help='최대 처리 건수')
    parser.add_argument('--output', type=str, default=None, help='출력 파일 경로 (라벨링: JSONL, 추론: CSV)')
    parser.add_argument('--save-csv', type=str, default=None, help='결과 CSV 저장 경로')
    parser.add_argument('--model', type=str, default='models/best_model.pt', help='모델 경로')
    parser.add_argument('--no-save-bq', action='store_true', help='BigQuery 저장 안함')
    args = parser.parse_args()

    from bq_connector import ABSABigQuery

    bq = ABSABigQuery()

    # 1. 통계 출력
    if args.stats:
        print("=" * 60)
        print("ABSA 분석 현황")
        print("=" * 60)
        stats = bq.get_analysis_stats()
        print(f"\n전체 리뷰: {stats['total']:,}개")
        print(f"분석 완료: {stats['analyzed']:,}개 ({stats.get('analyzed_pct', 0):.1f}%)")
        print(f"미분석: {stats['unanalyzed']:,}개 ({stats.get('unanalyzed_pct', 0):.1f}%)")

        if stats['analyzed'] > 0:
            print(f"\n감성 분포:")
            print(f"  Positive: {stats['positive']:,}개")
            print(f"  Negative: {stats['negative']:,}개")
            print(f"  Neutral: {stats['neutral']:,}개")
        return

    # 2. 리뷰 내보내기
    if args.export:
        output_path = args.output or f'data/reviews_export_{get_timestamp()}.csv'
        print(f"리뷰 내보내기: {output_path}")
        bq.export_for_training(output_path, limit=args.limit)
        return

    # 3. GPT 라벨링
    if args.label:
        from RQ.labeling import label_from_bigquery

        output_path = Path(args.output or f'data/labeled_{get_timestamp()}.jsonl')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # CSV 저장 경로
        save_csv = args.save_csv
        if save_csv is None and not args.no_save_bq:
            # 기본 CSV 저장 경로 생성
            save_csv = str(output_path.with_suffix('.csv'))

        print("=" * 60)
        print("GPT 라벨링 시작")
        print("=" * 60)
        print(f"출력 JSONL: {output_path}")
        if save_csv:
            print(f"출력 CSV: {save_csv}")
        print(f"BigQuery 저장: {'아니오' if args.no_save_bq else '예'}")
        print("=" * 60)

        results_df = label_from_bigquery(
            output_path=output_path,
            limit=args.limit,
            save_to_bq=not args.no_save_bq,
            save_csv=save_csv
        )

        if results_df is not None and len(results_df) > 0:
            print(f"\n라벨링 완료: {len(results_df):,}건")
        return

    # 4. 모델 추론
    if args.infer:
        from RQ.inference import run_inference_from_bigquery

        model_path = Path(args.model)
        if not model_path.exists():
            print(f"Error: 모델 파일을 찾을 수 없습니다: {model_path}")
            return

        # CSV 저장 경로
        save_csv = args.save_csv
        if save_csv is None and args.output:
            save_csv = args.output
        elif save_csv is None and not args.no_save_bq:
            save_csv = f'data/inference_{get_timestamp()}.csv'

        print("=" * 60)
        print("모델 추론 시작")
        print("=" * 60)
        print(f"모델: {model_path}")
        if save_csv:
            print(f"출력 CSV: {save_csv}")
        print(f"BigQuery 저장: {'아니오' if args.no_save_bq else '예'}")
        print("=" * 60)

        results_df = run_inference_from_bigquery(
            model_path=model_path,
            limit=args.limit,
            save_to_bq=not args.no_save_bq,
            output_csv=Path(save_csv) if save_csv else None
        )

        if results_df is not None and len(results_df) > 0:
            print(f"\n추론 완료: {len(results_df):,}건")
        return

    # 아무 옵션도 없으면 도움말 출력
    parser.print_help()


if __name__ == "__main__":
    main()
