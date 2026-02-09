"""
Step B3: 위험 케이스 샘플링

규칙 검수를 통과한 레코드 중 위험 케이스를 추출합니다.
- no_aspect: aspect_labels가 빈 배열
- all_neutral: sentiment=neutral + aspect 없음
- long_single_aspect: 텍스트 50자 이상 + aspect 1개
- neg_keyword_positive: 부정 키워드 + positive 라벨
- contrast_single: 대비 접속사 + aspect 1개
- rating_sentiment_mismatch: 평점-감정 불일치
- low_confidence: sentiment_score 0.3~0.5 범위
"""
import sys
import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from tqdm import tqdm

# Add RQ_absa to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from RQ_absa.config import (
    RAW_DATA_DIR,
    VALIDATION_DATA_DIR,
    RULE_VALIDATION_OUTPUT,
    RISK_CASES_OUTPUT,
    VALIDATION_CONFIG
)


class RiskSampler:
    """위험 케이스 샘플러"""

    def __init__(self):
        self.negative_keywords = VALIDATION_CONFIG["negative_keywords"]
        self.contrast_markers = VALIDATION_CONFIG["contrast_markers"]
        self.long_text_threshold = VALIDATION_CONFIG["long_text_threshold"]
        self.low_confidence_min, self.low_confidence_max = VALIDATION_CONFIG["low_confidence_range"]
        self.rating_sentiment_map = VALIDATION_CONFIG["rating_sentiment_map"]

        # 통계
        self.stats = defaultdict(int)
        self.risk_counts = defaultdict(int)

    def check_no_aspect(self, record: Dict) -> Tuple[bool, List[str]]:
        """aspect_labels가 빈 배열인지 확인"""
        aspect_labels = record.get("aspect_labels", [])
        if not aspect_labels or len(aspect_labels) == 0:
            return True, []
        return False, []

    def check_all_neutral(self, record: Dict) -> Tuple[bool, List[str]]:
        """sentiment=neutral + aspect 없음 확인"""
        sentiment = record.get("sentiment", "")
        aspect_labels = record.get("aspect_labels", [])

        if sentiment == "neutral" and (not aspect_labels or len(aspect_labels) == 0):
            return True, []
        return False, []

    def check_long_single_aspect(self, record: Dict) -> Tuple[bool, List[str]]:
        """텍스트 50자 이상 + aspect 1개 확인"""
        text = record.get("text", "")
        aspect_labels = record.get("aspect_labels", [])

        if len(text) >= self.long_text_threshold and len(aspect_labels) == 1:
            return True, []
        return False, []

    def check_neg_keyword_positive(self, record: Dict) -> Tuple[bool, List[str]]:
        """부정 키워드 + positive 라벨 확인"""
        text = record.get("text", "")
        sentiment = record.get("sentiment", "")

        if sentiment != "positive":
            return False, []

        matched_keywords = []
        text_lower = text.lower()

        for keyword in self.negative_keywords:
            if keyword in text_lower:
                matched_keywords.append(keyword)

        if matched_keywords:
            return True, matched_keywords
        return False, []

    def check_contrast_single(self, record: Dict) -> Tuple[bool, List[str]]:
        """대비 접속사 + aspect 1개 확인"""
        text = record.get("text", "")
        aspect_labels = record.get("aspect_labels", [])

        if len(aspect_labels) != 1:
            return False, []

        matched_markers = []
        for marker in self.contrast_markers:
            if marker in text:
                matched_markers.append(marker)

        if matched_markers:
            return True, matched_markers
        return False, []

    def check_rating_sentiment_mismatch(self, record: Dict) -> Tuple[bool, List[str]]:
        """평점-감정 불일치 확인"""
        rating = record.get("rating")
        sentiment = record.get("sentiment", "")

        if rating is None:
            return False, []

        try:
            rating = int(rating)
        except (ValueError, TypeError):
            return False, []

        expected_sentiment = self.rating_sentiment_map.get(rating)
        if expected_sentiment is None:
            return False, []

        # 강한 불일치만 체크 (rating 1-2 + positive 또는 rating 4-5 + negative)
        if (rating in [1, 2] and sentiment == "positive") or \
           (rating in [4, 5] and sentiment == "negative"):
            return True, [f"rating={rating}, sentiment={sentiment}"]

        return False, []

    def check_low_confidence(self, record: Dict) -> Tuple[bool, List[str]]:
        """낮은 신뢰도 확인 (sentiment_score 절대값 0.3~0.5)"""
        score = record.get("sentiment_score")

        if score is None:
            return False, []

        try:
            score = float(score)
            abs_score = abs(score)

            if self.low_confidence_min <= abs_score <= self.low_confidence_max:
                return True, [f"score={score:.2f}"]
        except (ValueError, TypeError):
            pass

        return False, []

    def analyze_record(self, record: Dict, rule_status: str = "valid") -> Optional[Dict]:
        """
        단일 레코드의 위험도 분석

        Args:
            record: 라벨링 레코드
            rule_status: 규칙 검수 상태 (valid/invalid)

        Returns:
            위험 케이스면 결과 딕셔너리, 아니면 None
        """
        # 규칙 검수에서 무효 판정된 레코드는 제외
        if rule_status == "invalid":
            return None

        index = record.get("index", -1)
        risk_reasons = []
        matched_items = []
        risk_level = None

        # HIGH 위험도 체크
        # 1. no_aspect
        is_risk, items = self.check_no_aspect(record)
        if is_risk:
            risk_reasons.append("no_aspect")
            matched_items.extend(items)

        # 2. all_neutral
        is_risk, items = self.check_all_neutral(record)
        if is_risk:
            risk_reasons.append("all_neutral")
            matched_items.extend(items)

        # 3. neg_keyword_positive
        is_risk, items = self.check_neg_keyword_positive(record)
        if is_risk:
            risk_reasons.append("neg_keyword_positive")
            matched_items.extend(items)

        # MEDIUM 위험도 체크
        # 4. long_single_aspect
        is_risk, items = self.check_long_single_aspect(record)
        if is_risk:
            risk_reasons.append("long_single_aspect")
            matched_items.extend(items)

        # 5. contrast_single
        is_risk, items = self.check_contrast_single(record)
        if is_risk:
            risk_reasons.append("contrast_single")
            matched_items.extend(items)

        # 6. rating_sentiment_mismatch
        is_risk, items = self.check_rating_sentiment_mismatch(record)
        if is_risk:
            risk_reasons.append("rating_sentiment_mismatch")
            matched_items.extend(items)

        # LOW 위험도 체크
        # 7. low_confidence
        is_risk, items = self.check_low_confidence(record)
        if is_risk:
            risk_reasons.append("low_confidence")
            matched_items.extend(items)

        # 위험 케이스가 아니면 None 반환
        if not risk_reasons:
            return None

        # 위험도 결정
        high_risks = {"no_aspect", "all_neutral", "neg_keyword_positive"}
        medium_risks = {"long_single_aspect", "contrast_single", "rating_sentiment_mismatch"}
        low_risks = {"low_confidence"}

        has_high = any(r in high_risks for r in risk_reasons)
        has_medium = any(r in medium_risks for r in risk_reasons)

        if has_high:
            risk_level = "HIGH"
        elif has_medium:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # 통계 업데이트
        self.stats[risk_level] += 1
        for reason in risk_reasons:
            self.risk_counts[reason] += 1

        return {
            "index": index,
            "risk_level": risk_level,
            "risk_reasons": risk_reasons,
            "matched_keywords": matched_items,
            "original_label": {
                "sentiment": record.get("sentiment"),
                "sentiment_score": record.get("sentiment_score"),
                "aspect_labels": record.get("aspect_labels", []),
                "evidence": record.get("evidence", "")
            },
            "text": record.get("text", ""),
            "rating": record.get("rating")
        }

    def load_rule_validation_results(self, path: Path) -> Dict[int, str]:
        """규칙 검수 결과 로드"""
        results = {}

        if not path.exists():
            print(f"경고: 규칙 검수 결과 파일이 없습니다: {path}")
            return results

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    results[record["index"]] = record["status"]
                except (json.JSONDecodeError, KeyError):
                    continue

        return results

    def sample_risks(
        self,
        labels_path: Path,
        output_path: Path,
        rule_validation_path: Optional[Path] = None,
        limit: Optional[int] = None
    ) -> Dict:
        """
        위험 케이스 샘플링

        Args:
            labels_path: 라벨링 결과 JSONL 경로
            output_path: 출력 JSONL 경로
            rule_validation_path: 규칙 검수 결과 경로
            limit: 처리할 레코드 수 제한

        Returns:
            통계 딕셔너리
        """
        # 통계 초기화
        self.stats = defaultdict(int)
        self.risk_counts = defaultdict(int)

        # 규칙 검수 결과 로드
        rule_results = {}
        if rule_validation_path:
            rule_results = self.load_rule_validation_results(rule_validation_path)
            print(f"규칙 검수 결과 로드: {len(rule_results):,}건")

        # 라벨링 결과 로드
        print(f"라벨링 결과 로드 중: {labels_path}")
        records = []

        with open(labels_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if limit and line_num >= limit:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    continue

        print(f"총 {len(records):,}개 레코드 로드")

        # 출력 디렉토리 생성
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 위험 케이스 추출
        print("위험 케이스 추출 중...")
        risk_cases = []
        total_checked = 0

        with open(output_path, 'w', encoding='utf-8') as f:
            for record in tqdm(records, desc="분석"):
                index = record.get("index", -1)

                # 규칙 검수 상태 확인
                rule_status = rule_results.get(index, "valid")

                result = self.analyze_record(record, rule_status)
                total_checked += 1

                if result:
                    risk_cases.append(result)
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')

        # 통계 계산
        stats = {
            "total_records": len(records),
            "total_checked": total_checked,
            "total_risk_cases": len(risk_cases),
            "risk_ratio": len(risk_cases) / len(records) if records else 0,
            "by_level": {
                "HIGH": self.stats.get("HIGH", 0),
                "MEDIUM": self.stats.get("MEDIUM", 0),
                "LOW": self.stats.get("LOW", 0)
            },
            "by_reason": dict(self.risk_counts)
        }

        return stats

    def print_report(self, stats: Dict):
        """위험 케이스 리포트 출력"""
        print("\n" + "=" * 60)
        print("위험 케이스 샘플링 리포트")
        print("=" * 60)

        print(f"\n[총계]")
        print(f"  총 레코드: {stats['total_records']:,}")
        print(f"  검사 대상: {stats['total_checked']:,}")
        print(f"  위험 케이스: {stats['total_risk_cases']:,} ({stats['risk_ratio']*100:.1f}%)")

        print(f"\n[위험도별 분포]")
        by_level = stats.get("by_level", {})
        for level in ["HIGH", "MEDIUM", "LOW"]:
            count = by_level.get(level, 0)
            ratio = count / stats['total_records'] * 100 if stats['total_records'] > 0 else 0
            print(f"  {level}: {count:,} ({ratio:.1f}%)")

        print(f"\n[위험 유형별 빈도]")
        by_reason = stats.get("by_reason", {})
        for reason, count in sorted(by_reason.items(), key=lambda x: -x[1]):
            ratio = count / stats['total_records'] * 100 if stats['total_records'] > 0 else 0
            print(f"  {reason}: {count:,} ({ratio:.2f}%)")

        # 예상 비용
        high_count = by_level.get("HIGH", 0)
        medium_count = by_level.get("MEDIUM", 0)
        low_count = by_level.get("LOW", 0)

        # gpt-4.1-mini 기준 예상 비용 (~$0.002/request)
        cost_per_request = 0.002
        total_judge_count = high_count + medium_count + low_count
        estimated_cost = total_judge_count * cost_per_request

        print(f"\n[예상 Judge 비용]")
        print(f"  HIGH ({high_count:,}건): ~${high_count * cost_per_request:.2f}")
        print(f"  MEDIUM ({medium_count:,}건): ~${medium_count * cost_per_request:.2f}")
        print(f"  LOW ({low_count:,}건): ~${low_count * cost_per_request:.2f}")
        print(f"  총 예상 비용: ~${estimated_cost:.2f}")

        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="위험 케이스 샘플링")
    parser.add_argument("--test", action="store_true", help="테스트 모드 (100건만 분석)")
    parser.add_argument("--limit", type=int, default=None, help="분석할 레코드 수 제한")
    parser.add_argument("--input", type=str, default=None, help="입력 파일 경로")
    parser.add_argument("--output", type=str, default=None, help="출력 파일 경로")
    parser.add_argument("--no-rule-filter", action="store_true",
                       help="규칙 검수 결과를 무시하고 모든 레코드 분석")

    args = parser.parse_args()

    print("=" * 60)
    print("STEP B3: 위험 케이스 샘플링")
    print("=" * 60)

    # 경로 설정
    labels_path = Path(args.input) if args.input else RAW_DATA_DIR / "chatgpt_labels_20k.jsonl"
    output_path = Path(args.output) if args.output else RISK_CASES_OUTPUT
    rule_validation_path = None if args.no_rule_filter else RULE_VALIDATION_OUTPUT

    # 입력 파일 확인
    if not labels_path.exists():
        print(f"오류: 입력 파일이 없습니다: {labels_path}")
        return

    # 테스트 모드
    limit = args.limit
    if args.test:
        limit = 100
        print(f"[테스트 모드] {limit}건만 분석합니다.")

    # 샘플링 수행
    sampler = RiskSampler()
    stats = sampler.sample_risks(
        labels_path=labels_path,
        output_path=output_path,
        rule_validation_path=rule_validation_path,
        limit=limit
    )

    # 리포트 출력
    sampler.print_report(stats)

    print(f"\n위험 케이스 저장됨: {output_path}")
    print("\n다음 단계: python scripts/step_b4_judge_review.py")


if __name__ == "__main__":
    main()
