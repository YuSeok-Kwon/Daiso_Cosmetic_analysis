"""
Step B4: Judge 모델 검수

위험 케이스로 분류된 레코드를 Judge 모델(gpt-4.1-mini)로 검수합니다.
- 기존 라벨의 오류 판단
- 필요시 수정된 라벨 제안
- 신뢰도(confidence) 점수 부여
"""
import sys
import json
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
from tqdm import tqdm

# Add RQ_absa to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from RQ_absa.config import (
    RAW_DATA_DIR,
    VALIDATION_DATA_DIR,
    RISK_CASES_OUTPUT,
    JUDGE_RESULTS_OUTPUT,
    VALIDATION_CONFIG
)


class JudgeReviewer:
    """Judge 모델 검수기"""

    def __init__(self, model: str = None):
        self.model = model or VALIDATION_CONFIG["judge_model"]
        self.client = None  # Lazy initialization

        # 통계
        self.stats = defaultdict(int)
        self.issue_counts = defaultdict(int)
        self.total_cost = 0.0
        self.total_tokens = 0

    def _init_client(self):
        """OpenAI 클라이언트 초기화 (lazy)"""
        if self.client is None:
            from openai_client import OpenAIClient
            self.client = OpenAIClient()

    def review_case(self, risk_case: Dict) -> Dict:
        """
        단일 위험 케이스 검수

        Args:
            risk_case: 위험 케이스 딕셔너리

        Returns:
            Judge 결과 딕셔너리
        """
        self._init_client()

        index = risk_case.get("index", -1)
        text = risk_case.get("text", "")
        rating = risk_case.get("rating")
        original_label = risk_case.get("original_label", {})
        risk_reasons = risk_case.get("risk_reasons", [])

        try:
            # Judge API 호출
            result = self.client.judge_review(
                text=text,
                rating=rating,
                original_label=original_label,
                risk_reasons=risk_reasons,
                model=self.model
            )

            # 통계 업데이트
            judgment = result.get("judgment", "uncertain")
            self.stats[judgment] += 1

            for issue in result.get("issues", []):
                self.issue_counts[issue] += 1

            self.total_cost += result.get("cost", 0)
            self.total_tokens += result.get("tokens", 0)

            # 결과 구성
            return {
                "index": index,
                "judgment": judgment,
                "issues": result.get("issues", []),
                "original_label": original_label,
                "fixed_label": result.get("fixed_label", {}),
                "confidence": result.get("confidence", 0.5),
                "reason": result.get("reason", ""),
                "risk_level": risk_case.get("risk_level"),
                "risk_reasons": risk_reasons,
                "model": result.get("model", self.model),
                "tokens": result.get("tokens", 0),
                "cost": result.get("cost", 0)
            }

        except Exception as e:
            print(f"\n오류 (index={index}): {e}")
            self.stats["error"] += 1

            return {
                "index": index,
                "judgment": "error",
                "issues": [],
                "original_label": original_label,
                "fixed_label": {},
                "confidence": 0,
                "reason": str(e),
                "risk_level": risk_case.get("risk_level"),
                "risk_reasons": risk_reasons,
                "model": self.model,
                "tokens": 0,
                "cost": 0
            }

    def review_file(
        self,
        risk_cases_path: Path,
        output_path: Path,
        risk_levels: List[str] = None,
        limit: Optional[int] = None,
        resume: bool = True
    ) -> Dict:
        """
        위험 케이스 파일 전체 검수

        Args:
            risk_cases_path: 위험 케이스 JSONL 경로
            output_path: 출력 JSONL 경로
            risk_levels: 검수할 위험도 레벨 (기본: ["HIGH", "MEDIUM"])
            limit: 검수할 레코드 수 제한
            resume: 기존 결과에서 이어서 진행

        Returns:
            통계 딕셔너리
        """
        if risk_levels is None:
            risk_levels = ["HIGH", "MEDIUM"]

        # 통계 초기화
        self.stats = defaultdict(int)
        self.issue_counts = defaultdict(int)
        self.total_cost = 0.0
        self.total_tokens = 0

        # 위험 케이스 로드
        print(f"위험 케이스 로드 중: {risk_cases_path}")
        risk_cases = []

        with open(risk_cases_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    case = json.loads(line)
                    # 지정된 위험도만 필터링
                    if case.get("risk_level") in risk_levels:
                        risk_cases.append(case)
                except json.JSONDecodeError:
                    continue

        print(f"검수 대상: {len(risk_cases):,}건 (위험도: {risk_levels})")

        # 기존 결과 로드 (resume 모드)
        existing_indices = set()
        if resume and output_path.exists():
            print(f"기존 결과 로드 중: {output_path}")
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        result = json.loads(line)
                        existing_indices.add(result["index"])
                    except (json.JSONDecodeError, KeyError):
                        continue
            print(f"기존 결과: {len(existing_indices):,}건")

        # 검수 대상 필터링
        cases_to_review = [c for c in risk_cases if c["index"] not in existing_indices]

        if limit:
            cases_to_review = cases_to_review[:limit]

        print(f"신규 검수 대상: {len(cases_to_review):,}건")

        if not cases_to_review:
            print("검수할 케이스가 없습니다.")
            return self._calculate_stats(risk_cases_path, output_path, risk_levels)

        # 출력 디렉토리 생성
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 검수 수행
        print(f"\nJudge 모델: {self.model}")
        print("검수 시작...")

        mode = 'a' if resume and output_path.exists() else 'w'

        with open(output_path, mode, encoding='utf-8') as f:
            for i, case in enumerate(tqdm(cases_to_review, desc="검수")):
                result = self.review_case(case)
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()

                # 진행 상황 출력 (100건마다)
                if (i + 1) % 100 == 0:
                    print(f"\n[진행] {i+1}/{len(cases_to_review)}, "
                          f"비용: ${self.total_cost:.2f}, "
                          f"토큰: {self.total_tokens:,}")

        # 최종 통계 계산
        return self._calculate_stats(risk_cases_path, output_path, risk_levels)

    def _calculate_stats(
        self,
        risk_cases_path: Path,
        output_path: Path,
        risk_levels: List[str]
    ) -> Dict:
        """전체 통계 계산"""
        # 결과 파일에서 통계 재계산
        stats = defaultdict(int)
        issue_counts = defaultdict(int)
        total_cost = 0.0
        total_tokens = 0

        if output_path.exists():
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        result = json.loads(line)
                        judgment = result.get("judgment", "uncertain")
                        stats[judgment] += 1

                        for issue in result.get("issues", []):
                            issue_counts[issue] += 1

                        total_cost += result.get("cost", 0)
                        total_tokens += result.get("tokens", 0)
                    except json.JSONDecodeError:
                        continue

        total_reviewed = sum(stats.values())

        return {
            "total_reviewed": total_reviewed,
            "risk_levels": risk_levels,
            "by_judgment": dict(stats),
            "by_issue": dict(issue_counts),
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "avg_cost_per_review": total_cost / total_reviewed if total_reviewed > 0 else 0
        }

    def print_report(self, stats: Dict):
        """검수 리포트 출력"""
        print("\n" + "=" * 60)
        print("Judge 모델 검수 리포트")
        print("=" * 60)

        print(f"\n[총계]")
        print(f"  검수 완료: {stats['total_reviewed']:,}건")
        print(f"  대상 위험도: {stats['risk_levels']}")

        print(f"\n[판정 결과]")
        by_judgment = stats.get("by_judgment", {})
        total = stats['total_reviewed']
        for judgment in ["ok", "fix", "uncertain", "error"]:
            count = by_judgment.get(judgment, 0)
            ratio = count / total * 100 if total > 0 else 0
            print(f"  {judgment}: {count:,} ({ratio:.1f}%)")

        print(f"\n[이슈 유형]")
        by_issue = stats.get("by_issue", {})
        for issue, count in sorted(by_issue.items(), key=lambda x: -x[1]):
            print(f"  {issue}: {count:,}")

        print(f"\n[비용]")
        print(f"  총 비용: ${stats['total_cost']:.4f}")
        print(f"  총 토큰: {stats['total_tokens']:,}")
        print(f"  평균 비용/건: ${stats['avg_cost_per_review']:.6f}")

        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Judge 모델 검수")
    parser.add_argument("--test", action="store_true", help="테스트 모드 (10건만 검수)")
    parser.add_argument("--limit", type=int, default=None, help="검수할 레코드 수 제한")
    parser.add_argument("--input", type=str, default=None, help="입력 파일 경로")
    parser.add_argument("--output", type=str, default=None, help="출력 파일 경로")
    parser.add_argument("--model", type=str, default=None, help="Judge 모델 (기본: gpt-4.1-mini)")
    parser.add_argument("--risk-levels", type=str, default="HIGH,MEDIUM",
                       help="검수할 위험도 (쉼표로 구분, 기본: HIGH,MEDIUM)")
    parser.add_argument("--include-low", action="store_true", help="LOW 위험도도 포함")
    parser.add_argument("--no-resume", action="store_true", help="처음부터 다시 시작")

    args = parser.parse_args()

    print("=" * 60)
    print("STEP B4: JUDGE 모델 검수")
    print("=" * 60)

    # API 키 확인
    if "OPENAI_API_KEY" not in os.environ:
        print("오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("\nexport OPENAI_API_KEY='your-key-here'")
        return

    # 경로 설정
    risk_cases_path = Path(args.input) if args.input else RISK_CASES_OUTPUT
    output_path = Path(args.output) if args.output else JUDGE_RESULTS_OUTPUT

    # 입력 파일 확인
    if not risk_cases_path.exists():
        print(f"오류: 입력 파일이 없습니다: {risk_cases_path}")
        print("\n먼저 step_b3_risk_sampling.py를 실행하세요.")
        return

    # 위험도 레벨 설정
    risk_levels = args.risk_levels.split(",")
    if args.include_low:
        risk_levels.append("LOW")
    risk_levels = [level.strip().upper() for level in risk_levels]

    # 테스트 모드
    limit = args.limit
    if args.test:
        limit = 10
        print(f"[테스트 모드] {limit}건만 검수합니다.")

    # 검수 수행
    reviewer = JudgeReviewer(model=args.model)
    stats = reviewer.review_file(
        risk_cases_path=risk_cases_path,
        output_path=output_path,
        risk_levels=risk_levels,
        limit=limit,
        resume=not args.no_resume
    )

    # 리포트 출력
    reviewer.print_report(stats)

    print(f"\n검수 결과 저장됨: {output_path}")
    print("\n다음 단계: python scripts/step_b5_merge_results.py")


if __name__ == "__main__":
    main()
