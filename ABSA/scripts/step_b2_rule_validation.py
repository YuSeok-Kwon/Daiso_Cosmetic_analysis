"""
Step B2: 규칙 기반 검수

ChatGPT로 라벨링된 데이터를 규칙 기반으로 검증합니다.
- JSON 파싱 검증
- 필수 필드 존재 여부
- sentiment/aspect 값 유효성
- evidence 검증 (원문 substring 여부)
- sentiment_score 범위 검증
- aspect 중복 검증
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm

# Add RQ_absa to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from RQ_absa.config import (
    RAW_DATA_DIR,
    VALIDATION_DATA_DIR,
    RULE_VALIDATION_OUTPUT,
    VALIDATION_CONFIG,
    ASPECT_LABELS
)


class RuleValidator:
    """규칙 기반 검수기"""

    def __init__(self):
        self.valid_sentiments = VALIDATION_CONFIG["valid_sentiments"]
        self.valid_aspects = VALIDATION_CONFIG["valid_aspects"]
        self.score_min, self.score_max = VALIDATION_CONFIG["score_range"]

        # 통계
        self.stats = defaultdict(int)
        self.issue_counts = defaultdict(int)

    def validate_record(self, record: Dict) -> Dict:
        """
        단일 레코드 검증

        Args:
            record: JSONL에서 읽은 레코드

        Returns:
            검증 결과 딕셔너리
        """
        issues = []
        details = {}
        index = record.get("index", -1)

        # 1. 필수 필드 검증
        required_fields = ["sentiment", "aspect_labels", "evidence"]
        for field in required_fields:
            if field not in record or record[field] is None:
                issues.append("missing_field")
                details["missing_field"] = f"필드 '{field}' 누락"

        # 필수 필드가 없으면 더 이상 검증 불가
        if "missing_field" in issues:
            return self._create_result(index, issues, details)

        # 2. sentiment 값 검증
        sentiment = record.get("sentiment", "")
        if sentiment not in self.valid_sentiments:
            issues.append("invalid_sentiment")
            details["invalid_sentiment"] = f"'{sentiment}'은 유효하지 않음 (유효값: {self.valid_sentiments})"

        # 3. sentiment_score 범위 검증
        score = record.get("sentiment_score")
        if score is not None:
            try:
                score = float(score)
                if score < self.score_min or score > self.score_max:
                    issues.append("invalid_score")
                    details["invalid_score"] = f"점수 {score}이 범위 [{self.score_min}, {self.score_max}] 밖임"
            except (ValueError, TypeError):
                issues.append("invalid_score")
                details["invalid_score"] = f"점수 '{score}'을 숫자로 변환할 수 없음"

        # 4. aspect_labels 검증
        aspect_labels = record.get("aspect_labels", [])
        if not isinstance(aspect_labels, list):
            issues.append("invalid_aspect")
            details["invalid_aspect"] = f"aspect_labels가 리스트가 아님: {type(aspect_labels)}"
        else:
            invalid_aspects = [a for a in aspect_labels if a not in self.valid_aspects]
            if invalid_aspects:
                issues.append("invalid_aspect")
                details["invalid_aspect"] = f"유효하지 않은 aspect: {invalid_aspects}"

            # 중복 aspect 검증
            if len(aspect_labels) != len(set(aspect_labels)):
                issues.append("duplicate_aspect")
                duplicates = [a for a in aspect_labels if aspect_labels.count(a) > 1]
                details["duplicate_aspect"] = f"중복된 aspect: {list(set(duplicates))}"

        # 5. evidence 검증 (원문 substring 여부)
        evidence = record.get("evidence", "")
        text = record.get("text", "")

        if evidence and text and evidence != "N/A":
            # evidence가 원문에 포함되어 있는지 확인
            # 일부 전처리 (공백, 줄바꿈 정규화)
            normalized_text = " ".join(text.split())
            normalized_evidence = " ".join(evidence.split())

            if normalized_evidence not in normalized_text:
                # 부분 매칭 시도 (evidence가 약간 변형된 경우)
                # evidence의 핵심 부분만 확인
                evidence_words = normalized_evidence.split()
                if len(evidence_words) > 3:
                    # 첫 3단어 또는 마지막 3단어가 매칭되는지 확인
                    first_part = " ".join(evidence_words[:3])
                    last_part = " ".join(evidence_words[-3:])
                    if first_part not in normalized_text and last_part not in normalized_text:
                        issues.append("evidence_mismatch")
                        # evidence가 너무 길면 축약
                        display_evidence = evidence[:50] + "..." if len(evidence) > 50 else evidence
                        details["evidence_mismatch"] = f"evidence '{display_evidence}'가 원문에 없음"
                elif normalized_evidence not in normalized_text:
                    issues.append("evidence_mismatch")
                    display_evidence = evidence[:50] + "..." if len(evidence) > 50 else evidence
                    details["evidence_mismatch"] = f"evidence '{display_evidence}'가 원문에 없음"

        return self._create_result(index, issues, details)

    def _create_result(self, index: int, issues: List[str], details: Dict) -> Dict:
        """검증 결과 딕셔너리 생성"""
        status = "invalid" if issues else "valid"

        # 통계 업데이트
        self.stats[status] += 1
        for issue in issues:
            self.issue_counts[issue] += 1

        return {
            "index": index,
            "status": status,
            "issues": issues,
            "details": details
        }

    def validate_file(
        self,
        input_path: Path,
        output_path: Path,
        limit: Optional[int] = None
    ) -> Dict:
        """
        JSONL 파일 전체 검증

        Args:
            input_path: 입력 JSONL 파일 경로
            output_path: 출력 JSONL 파일 경로
            limit: 검증할 레코드 수 제한 (테스트용)

        Returns:
            통계 딕셔너리
        """
        # 통계 초기화
        self.stats = defaultdict(int)
        self.issue_counts = defaultdict(int)

        # 입력 파일 읽기
        records = []
        parse_errors = 0

        print(f"입력 파일 로드 중: {input_path}")

        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if limit and line_num >= limit:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    parse_errors += 1
                    # JSON 파싱 실패 레코드 기록
                    records.append({
                        "index": line_num,
                        "_parse_error": str(e),
                        "_raw_line": line[:200]  # 처음 200자만
                    })

        print(f"총 {len(records):,}개 레코드 로드 (파싱 오류: {parse_errors})")

        # 출력 디렉토리 생성
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 검증 수행
        print("검증 수행 중...")
        results = []

        with open(output_path, 'w', encoding='utf-8') as f:
            for record in tqdm(records, desc="검증"):
                # 파싱 오류 레코드 처리
                if "_parse_error" in record:
                    result = {
                        "index": record.get("index", -1),
                        "status": "invalid",
                        "issues": ["invalid_json"],
                        "details": {"invalid_json": record["_parse_error"]}
                    }
                    self.stats["invalid"] += 1
                    self.issue_counts["invalid_json"] += 1
                else:
                    result = self.validate_record(record)

                results.append(result)
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        # 통계 계산
        total = len(results)
        valid_count = self.stats.get("valid", 0)
        invalid_count = self.stats.get("invalid", 0)

        stats = {
            "total_records": total,
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "valid_ratio": valid_count / total if total > 0 else 0,
            "invalid_ratio": invalid_count / total if total > 0 else 0,
            "issue_counts": dict(self.issue_counts),
            "parse_errors": parse_errors
        }

        return stats

    def print_report(self, stats: Dict):
        """검증 리포트 출력"""
        print("\n" + "=" * 60)
        print("규칙 기반 검수 리포트")
        print("=" * 60)

        print(f"\n[총계]")
        print(f"  총 레코드: {stats['total_records']:,}")
        print(f"  유효: {stats['valid_count']:,} ({stats['valid_ratio']*100:.1f}%)")
        print(f"  무효: {stats['invalid_count']:,} ({stats['invalid_ratio']*100:.1f}%)")

        print(f"\n[이슈별 빈도]")
        issue_counts = stats.get("issue_counts", {})
        for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            ratio = count / stats['total_records'] * 100 if stats['total_records'] > 0 else 0
            print(f"  {issue}: {count:,} ({ratio:.2f}%)")

        if stats.get("parse_errors", 0) > 0:
            print(f"\n[경고]")
            print(f"  JSON 파싱 오류: {stats['parse_errors']:,}건")

        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="규칙 기반 검수")
    parser.add_argument("--test", action="store_true", help="테스트 모드 (100건만 검증)")
    parser.add_argument("--limit", type=int, default=None, help="검증할 레코드 수 제한")
    parser.add_argument("--input", type=str, default=None, help="입력 파일 경로 (기본: chatgpt_labels_20k.jsonl)")
    parser.add_argument("--output", type=str, default=None, help="출력 파일 경로")

    args = parser.parse_args()

    print("=" * 60)
    print("STEP B2: 규칙 기반 검수")
    print("=" * 60)

    # 경로 설정
    input_path = Path(args.input) if args.input else RAW_DATA_DIR / "chatgpt_labels_20k.jsonl"
    output_path = Path(args.output) if args.output else RULE_VALIDATION_OUTPUT

    # 입력 파일 확인
    if not input_path.exists():
        print(f"오류: 입력 파일이 없습니다: {input_path}")
        print("\n먼저 step_b_labeling.py를 실행하세요.")
        return

    # 테스트 모드
    limit = args.limit
    if args.test:
        limit = 100
        print(f"[테스트 모드] {limit}건만 검증합니다.")

    # 검증 수행
    validator = RuleValidator()
    stats = validator.validate_file(input_path, output_path, limit=limit)

    # 리포트 출력
    validator.print_report(stats)

    print(f"\n검증 결과 저장됨: {output_path}")
    print("\n다음 단계: python scripts/step_b3_risk_sampling.py")


if __name__ == "__main__":
    main()
