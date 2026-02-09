"""
Step B5: 결과 병합 및 리포트 생성

검수 결과를 원본 라벨에 병합하고 최종 리포트를 생성합니다.
- 규칙 검수 결과 반영
- Judge 검수 결과 반영 (fix 케이스 수정)
- 최종 검증된 라벨 파일 생성
- 종합 리포트 생성
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

# Add RQ_absa to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from RQ_absa.config import (
    RAW_DATA_DIR,
    VALIDATION_DATA_DIR,
    RULE_VALIDATION_OUTPUT,
    RISK_CASES_OUTPUT,
    JUDGE_RESULTS_OUTPUT,
    VALIDATED_LABELS_OUTPUT
)


class ResultMerger:
    """결과 병합기"""

    def __init__(self):
        # 통계
        self.stats = {
            "total_records": 0,
            "rule_valid": 0,
            "rule_invalid": 0,
            "risk_high": 0,
            "risk_medium": 0,
            "risk_low": 0,
            "judge_ok": 0,
            "judge_fix": 0,
            "judge_uncertain": 0,
            "judge_error": 0,
            "final_verified": 0,
            "final_fixed": 0,
            "final_needs_review": 0,
            "final_removed": 0,
            "final_unchecked": 0
        }

    def load_jsonl(self, path: Path) -> Dict[int, Dict]:
        """JSONL 파일을 index를 키로 하는 딕셔너리로 로드"""
        results = {}

        if not path.exists():
            print(f"경고: 파일이 없습니다: {path}")
            return results

        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    index = record.get("index")
                    if index is not None:
                        results[index] = record
                except json.JSONDecodeError:
                    continue

        return results

    def merge_labels(
        self,
        original: Dict,
        judge_result: Optional[Dict]
    ) -> Tuple[Dict, str, List[str]]:
        """
        원본 라벨과 Judge 결과 병합

        Args:
            original: 원본 라벨 딕셔너리
            judge_result: Judge 검수 결과 (없으면 None)

        Returns:
            (병합된 라벨, 상태, 변경된 필드 목록)
        """
        merged = original.copy()
        changes = []

        if judge_result is None:
            return merged, "unchecked", []

        judgment = judge_result.get("judgment", "uncertain")

        if judgment == "ok":
            return merged, "verified", []

        elif judgment == "fix":
            fixed_label = judge_result.get("fixed_label", {})

            # 수정 사항 적용
            if "sentiment" in fixed_label and fixed_label["sentiment"] != original.get("sentiment"):
                merged["original_sentiment"] = original.get("sentiment")
                merged["sentiment"] = fixed_label["sentiment"]
                changes.append("sentiment")

            if "aspect_labels" in fixed_label:
                original_aspects = set(original.get("aspect_labels", []))
                fixed_aspects = set(fixed_label["aspect_labels"])
                if original_aspects != fixed_aspects:
                    merged["original_aspects"] = original.get("aspect_labels", [])
                    merged["aspect_labels"] = fixed_label["aspect_labels"]
                    changes.append("aspect_labels")

            if "sentiment_score" in fixed_label:
                merged["original_sentiment_score"] = original.get("sentiment_score")
                merged["sentiment_score"] = fixed_label["sentiment_score"]
                changes.append("sentiment_score")

            if "evidence" in fixed_label:
                merged["original_evidence"] = original.get("evidence")
                merged["evidence"] = fixed_label["evidence"]
                changes.append("evidence")

            return merged, "fixed", changes

        else:  # uncertain or error
            return merged, "needs_human_review", []

    def merge_all(
        self,
        labels_path: Path,
        rule_validation_path: Path,
        risk_cases_path: Path,
        judge_results_path: Path,
        output_path: Path,
        dry_run: bool = False
    ) -> Dict:
        """
        모든 결과 병합

        Args:
            labels_path: 원본 라벨 파일 경로
            rule_validation_path: 규칙 검수 결과 경로
            risk_cases_path: 위험 케이스 파일 경로
            judge_results_path: Judge 결과 경로
            output_path: 출력 파일 경로
            dry_run: True면 실제 파일을 쓰지 않음

        Returns:
            통계 딕셔너리
        """
        # 각 결과 파일 로드
        print("결과 파일 로드 중...")

        print(f"  원본 라벨: {labels_path}")
        original_labels = self.load_jsonl(labels_path)
        print(f"    {len(original_labels):,}건")

        print(f"  규칙 검수: {rule_validation_path}")
        rule_results = self.load_jsonl(rule_validation_path)
        print(f"    {len(rule_results):,}건")

        print(f"  위험 케이스: {risk_cases_path}")
        risk_cases = self.load_jsonl(risk_cases_path)
        print(f"    {len(risk_cases):,}건")

        print(f"  Judge 결과: {judge_results_path}")
        judge_results = self.load_jsonl(judge_results_path)
        print(f"    {len(judge_results):,}건")

        # 통계 초기화
        self.stats = {key: 0 for key in self.stats}
        self.stats["total_records"] = len(original_labels)

        # 규칙 검수 통계
        for result in rule_results.values():
            if result.get("status") == "valid":
                self.stats["rule_valid"] += 1
            else:
                self.stats["rule_invalid"] += 1

        # 위험 케이스 통계
        for case in risk_cases.values():
            level = case.get("risk_level", "").upper()
            if level == "HIGH":
                self.stats["risk_high"] += 1
            elif level == "MEDIUM":
                self.stats["risk_medium"] += 1
            elif level == "LOW":
                self.stats["risk_low"] += 1

        # Judge 통계
        for result in judge_results.values():
            judgment = result.get("judgment", "uncertain")
            if judgment == "ok":
                self.stats["judge_ok"] += 1
            elif judgment == "fix":
                self.stats["judge_fix"] += 1
            elif judgment == "uncertain":
                self.stats["judge_uncertain"] += 1
            else:
                self.stats["judge_error"] += 1

        # 출력 디렉토리 생성
        if not dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # 병합 수행
        print("\n병합 수행 중...")
        merged_records = []

        output_file = open(output_path, 'w', encoding='utf-8') if not dry_run else None

        try:
            for index, original in tqdm(original_labels.items(), desc="병합"):
                # 규칙 검수 상태 확인
                rule_result = rule_results.get(index, {})
                rule_status = rule_result.get("status", "valid")

                # 규칙 위반 레코드 처리
                if rule_status == "invalid":
                    self.stats["final_removed"] += 1
                    continue  # 규칙 위반 레코드는 제외

                # Judge 결과 확인 및 병합
                judge_result = judge_results.get(index)
                merged, validation_status, changes = self.merge_labels(original, judge_result)

                # 상태별 통계
                if validation_status == "verified":
                    self.stats["final_verified"] += 1
                elif validation_status == "fixed":
                    self.stats["final_fixed"] += 1
                elif validation_status == "needs_human_review":
                    self.stats["final_needs_review"] += 1
                else:  # unchecked
                    self.stats["final_unchecked"] += 1

                # 메타데이터 추가
                merged["validation_status"] = validation_status
                merged["validation_changes"] = changes

                if not dry_run:
                    output_file.write(json.dumps(merged, ensure_ascii=False) + '\n')

                merged_records.append(merged)

        finally:
            if output_file:
                output_file.close()

        # 최종 통계 계산
        total_final = (
            self.stats["final_verified"] +
            self.stats["final_fixed"] +
            self.stats["final_needs_review"] +
            self.stats["final_unchecked"]
        )

        self.stats["total_final"] = total_final

        return self.stats

    def print_report(self, stats: Dict, output_path: Path = None):
        """종합 리포트 출력"""
        report_lines = []

        def add_line(line=""):
            report_lines.append(line)
            print(line)

        add_line()
        add_line("=" * 60)
        add_line("ABSA 검수 종합 리포트")
        add_line(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        add_line("=" * 60)

        add_line(f"\n총 레코드: {stats['total_records']:,}")

        add_line("\n[Step 1: 규칙 검수]")
        total = stats['total_records']
        valid = stats['rule_valid']
        invalid = stats['rule_invalid']
        add_line(f"  유효: {valid:,} ({valid/total*100:.1f}%)" if total > 0 else "  유효: 0")
        add_line(f"  무효: {invalid:,} ({invalid/total*100:.1f}%)" if total > 0 else "  무효: 0")

        add_line("\n[Step 2: 위험 케이스]")
        risk_total = stats['risk_high'] + stats['risk_medium'] + stats['risk_low']
        add_line(f"  HIGH: {stats['risk_high']:,} ({stats['risk_high']/total*100:.1f}%)" if total > 0 else "  HIGH: 0")
        add_line(f"  MEDIUM: {stats['risk_medium']:,} ({stats['risk_medium']/total*100:.1f}%)" if total > 0 else "  MEDIUM: 0")
        add_line(f"  LOW: {stats['risk_low']:,} ({stats['risk_low']/total*100:.1f}%)" if total > 0 else "  LOW: 0")
        add_line(f"  총 위험 케이스: {risk_total:,} ({risk_total/total*100:.1f}%)" if total > 0 else "  총 위험 케이스: 0")

        add_line("\n[Step 3: Judge 검수]")
        judge_total = stats['judge_ok'] + stats['judge_fix'] + stats['judge_uncertain'] + stats['judge_error']
        if judge_total > 0:
            add_line(f"  OK: {stats['judge_ok']:,} ({stats['judge_ok']/judge_total*100:.1f}%)")
            add_line(f"  Fix: {stats['judge_fix']:,} ({stats['judge_fix']/judge_total*100:.1f}%)")
            add_line(f"  Uncertain: {stats['judge_uncertain']:,} ({stats['judge_uncertain']/judge_total*100:.1f}%)")
            add_line(f"  Error: {stats['judge_error']:,} ({stats['judge_error']/judge_total*100:.1f}%)")
        else:
            add_line("  검수된 케이스 없음")

        add_line("\n[최종 결과]")
        final_total = stats.get('total_final', 0)
        add_line(f"  원본 유지 (verified): {stats['final_verified']:,}")
        add_line(f"  자동 수정 (fixed): {stats['final_fixed']:,}")
        add_line(f"  사람 검수 필요: {stats['final_needs_review']:,}")
        add_line(f"  검수 미대상: {stats['final_unchecked']:,}")
        add_line(f"  규칙 위반 제거: {stats['final_removed']:,}")
        add_line(f"  최종 레코드 수: {final_total:,}")

        # 품질 점수 계산
        if final_total > 0:
            verified_ratio = stats['final_verified'] / final_total
            fixed_ratio = stats['final_fixed'] / final_total
            quality_score = verified_ratio + (fixed_ratio * 0.8)  # 수정된 것은 80% 가중치
            add_line(f"\n[품질 점수]")
            add_line(f"  검증 완료율: {(verified_ratio + fixed_ratio)*100:.1f}%")
            add_line(f"  품질 점수: {quality_score*100:.1f}/100")

        add_line("\n" + "=" * 60)

        # 리포트 파일 저장
        if output_path:
            report_path = output_path.parent / "validation_report.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            add_line(f"리포트 저장됨: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="결과 병합 및 리포트 생성")
    parser.add_argument("--dry-run", action="store_true", help="실제 파일을 쓰지 않고 통계만 출력")
    parser.add_argument("--labels", type=str, default=None, help="원본 라벨 파일 경로")
    parser.add_argument("--rule-validation", type=str, default=None, help="규칙 검수 결과 경로")
    parser.add_argument("--risk-cases", type=str, default=None, help="위험 케이스 파일 경로")
    parser.add_argument("--judge-results", type=str, default=None, help="Judge 결과 경로")
    parser.add_argument("--output", type=str, default=None, help="출력 파일 경로")

    args = parser.parse_args()

    print("=" * 60)
    print("STEP B5: 결과 병합 및 리포트 생성")
    print("=" * 60)

    # 경로 설정
    labels_path = Path(args.labels) if args.labels else RAW_DATA_DIR / "chatgpt_labels_20k.jsonl"
    rule_validation_path = Path(args.rule_validation) if args.rule_validation else RULE_VALIDATION_OUTPUT
    risk_cases_path = Path(args.risk_cases) if args.risk_cases else RISK_CASES_OUTPUT
    judge_results_path = Path(args.judge_results) if args.judge_results else JUDGE_RESULTS_OUTPUT
    output_path = Path(args.output) if args.output else VALIDATED_LABELS_OUTPUT

    # 입력 파일 확인
    if not labels_path.exists():
        print(f"오류: 원본 라벨 파일이 없습니다: {labels_path}")
        return

    if args.dry_run:
        print("\n[Dry-run 모드] 실제 파일을 쓰지 않습니다.")

    # 병합 수행
    merger = ResultMerger()
    stats = merger.merge_all(
        labels_path=labels_path,
        rule_validation_path=rule_validation_path,
        risk_cases_path=risk_cases_path,
        judge_results_path=judge_results_path,
        output_path=output_path,
        dry_run=args.dry_run
    )

    # 리포트 출력
    merger.print_report(stats, output_path if not args.dry_run else None)

    if not args.dry_run:
        print(f"\n검증된 라벨 저장됨: {output_path}")

    print("\n검수 파이프라인 완료!")
    print("다음 단계: python scripts/step_c_create_dataset.py")


if __name__ == "__main__":
    main()
