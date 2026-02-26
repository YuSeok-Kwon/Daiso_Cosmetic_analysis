"""
2단계 층화 샘플링 + 전체 레벨 sentiment 균형 조정

1단계: 저품질 리뷰 필터링 (중복 텍스트, 외국어, 빈 텍스트 등)
2단계: 대분류(category_1)별 쿼터 배정 (비례 + 최소 보장)
3단계: 소분류(category_2)별 쿼터 배정 (비례 + 최소 보장)
4단계: 소분류 단위에서 가용 데이터 범위 내 자연 추출 (sentiment 강제 X)
5단계: 전체 2만 레벨에서 sentiment 균형 조정 (부족분 대분류에서 추가 확보)
"""
import pandas as pd
import numpy as np
import re
from typing import Dict, Optional, List, Tuple
from pathlib import Path


class NaturalStratifiedSampler:
    """
    자연 분포 기반 층화 샘플러
    - 소분류 단위에서는 sentiment 강제 없이 자연 추출
    - 전체 레벨에서 sentiment 균형 조정
    - 복원추출 없음
    """

    def __init__(
        self,
        target_size: int = 20000,
        category_1_column: str = 'category_1',
        category_2_column: str = 'category_2',
        category_1_min_floor: int = 600,
        category_2_min_floor: int = 200,
        skip_cat2_categories: Optional[List[str]] = None,
        target_sentiment_distribution: Optional[Dict[str, float]] = None,
        random_state: int = 42,
        filter_duplicates: bool = True,
        filter_foreign: bool = True,
        min_korean_ratio: float = 0.3,
        text_column: str = 'text'
    ):
        """
        Args:
            target_size: 목표 샘플 수
            category_1_column: 대분류 컬럼명
            category_2_column: 소분류 컬럼명
            category_1_min_floor: 대분류별 최소 보장 개수
            category_2_min_floor: 소분류별 최소 보장 개수
            skip_cat2_categories: 소분류 쿼터 배정을 스킵할 대분류 목록
            target_sentiment_distribution: 목표 sentiment 비율 (전체 레벨)
            random_state: 랜덤 시드
            filter_duplicates: 중복 텍스트 리뷰 필터링 여부
            filter_foreign: 외국어/저품질 리뷰 필터링 여부
            min_korean_ratio: 최소 한국어 비율 (이 비율 미만이면 외국어로 판단)
            text_column: 텍스트 컬럼명
        """
        self.target_size = target_size
        self.category_1_column = category_1_column
        self.category_2_column = category_2_column
        self.category_1_min_floor = category_1_min_floor
        self.category_2_min_floor = category_2_min_floor
        self.skip_cat2_categories = skip_cat2_categories or []
        self.random_state = random_state
        self.filter_duplicates = filter_duplicates
        self.filter_foreign = filter_foreign
        self.min_korean_ratio = min_korean_ratio
        self.text_column = text_column

        # 목표 sentiment 분포 (전체 레벨 기준)
        if target_sentiment_distribution is None:
            self.target_sentiment_distribution = {
                "negative": 0.30,   # 1-2점
                "neutral": 0.30,    # 3점
                "positive": 0.40    # 4-5점
            }
        else:
            self.target_sentiment_distribution = target_sentiment_distribution

    def _filter_low_quality_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        저품질 리뷰 필터링
        - 중복 텍스트 제거 (첫 번째 유지)
        - 외국어/저품질 리뷰 제거
        """
        original_count = len(df)
        df = df.copy()

        print("\n" + "="*60)
        print("저품질 리뷰 필터링")
        print("="*60)

        # 1. 중복 텍스트 제거
        if self.filter_duplicates:
            before = len(df)
            df = df.drop_duplicates(subset=[self.text_column], keep='first')
            removed = before - len(df)
            print(f"중복 텍스트 제거: {removed:,}개 ({removed/original_count*100:.2f}%)")

        # 2. 외국어/저품질 리뷰 제거
        if self.filter_foreign:
            before = len(df)
            quality_mask = df[self.text_column].apply(self._is_valid_korean_review)
            df = df[quality_mask]
            removed = before - len(df)
            print(f"외국어/저품질 제거: {removed:,}개 ({removed/original_count*100:.2f}%)")

        total_removed = original_count - len(df)
        print(f"\n총 필터링: {total_removed:,}개 ({total_removed/original_count*100:.2f}%)")
        print(f"남은 리뷰: {len(df):,}개")

        return df

    def _is_valid_korean_review(self, text: str) -> bool:
        """
        유효한 한국어 리뷰인지 판별

        제외 대상:
        - 빈 텍스트 또는 NaN
        - 한국어 비율이 min_korean_ratio 미만
        - 반복 패턴만 있는 텍스트 (ㅋㅋㅋ, ㅎㅎㅎ 등)
        - 자음/모음만 있는 텍스트
        """
        # NaN 또는 빈 텍스트
        if pd.isna(text) or not isinstance(text, str):
            return False

        text = text.strip()
        if len(text) == 0:
            return False

        # 한글 (자모 포함)
        korean_chars = len(re.findall(r'[가-힣ㄱ-ㅎㅏ-ㅣ]', text))
        # 영어
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        # 일본어 (히라가나, 가타카나)
        japanese_chars = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF]', text))
        # 중국어 (한자, 한국어 한자 제외 범위)
        chinese_chars = len(re.findall(r'[\u4E00-\u9FFF]', text))
        # 숫자, 공백, 특수문자 제외한 총 문자
        total_chars = korean_chars + english_chars + japanese_chars + chinese_chars

        if total_chars == 0:
            return False

        korean_ratio = korean_chars / total_chars

        # 한국어 비율이 낮으면 외국어로 판단
        if korean_ratio < self.min_korean_ratio:
            return False

        # 반복 패턴만 있는지 확인 (ㅋㅋㅋ, ㅎㅎㅎ, ... 등)
        # 자음/모음 제거 후 실제 한글 단어가 있는지 확인
        actual_korean = re.findall(r'[가-힣]+', text)
        if len(actual_korean) == 0:
            # 자음/모음만 있는 경우
            consonants_only = re.findall(r'[ㄱ-ㅎㅏ-ㅣ]+', text)
            # 의미 있는 내용 없이 자음/모음만 반복
            if consonants_only:
                # 유니크한 자음/모음 개수가 2개 이하면 의미없는 반복으로 판단
                unique_jamo = set(''.join(consonants_only))
                if len(unique_jamo) <= 2:
                    return False

        return True

    def _assign_sentiment_group(self, df: pd.DataFrame) -> pd.DataFrame:
        """평점 기반 sentiment 그룹 할당"""
        df = df.copy()
        df['sentiment_group'] = pd.cut(
            df['rating'],
            bins=[0, 2, 3, 5],
            labels=['negative', 'neutral', 'positive'],
            include_lowest=True
        )
        return df

    def _calculate_quotas_with_min_floor(
        self,
        counts: pd.Series,
        total_quota: int,
        min_floor: int
    ) -> Dict[str, int]:
        """최소 보장 + 비례 배분 쿼터 계산"""
        categories = counts.index.tolist()
        n_categories = len(categories)

        min_total = min_floor * n_categories
        remaining = total_quota - min_total

        if remaining < 0:
            quota_per_cat = total_quota // n_categories
            quotas = {cat: quota_per_cat for cat in categories}
            remainder = total_quota - (quota_per_cat * n_categories)
            for i, cat in enumerate(categories):
                if i < remainder:
                    quotas[cat] += 1
            return quotas

        total_original = counts.sum()
        proportions = {cat: count / total_original for cat, count in counts.items()}

        quotas = {}
        for cat in categories:
            proportional_share = int(remaining * proportions[cat])
            quotas[cat] = min_floor + proportional_share

        total_quota_calculated = sum(quotas.values())
        diff = total_quota - total_quota_calculated
        if diff != 0:
            largest_cat = max(quotas, key=quotas.get)
            quotas[largest_cat] += diff

        return quotas

    def sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        자연 분포 기반 층화 샘플링 수행

        1. 저품질 리뷰 필터링 (중복, 외국어 등)
        2. 대분류별 쿼터 배정
        3. 소분류별 쿼터 배정
        4. 소분류 단위 자연 추출 (sentiment 강제 X)
        5. 전체 레벨 sentiment 균형 조정
        """
        print(f"원본 데이터: {len(df):,}개")
        print(f"목표 샘플: {self.target_size:,}개")
        print(f"대분류별 최소 보장: {self.category_1_min_floor}개")
        print(f"소분류별 최소 보장: {self.category_2_min_floor}개")
        if self.skip_cat2_categories:
            print(f"소분류 스킵 대분류: {self.skip_cat2_categories}")

        # =========================================
        # 0단계: 저품질 리뷰 필터링
        # =========================================
        if self.filter_duplicates or self.filter_foreign:
            df = self._filter_low_quality_reviews(df)

        # sentiment 그룹 할당
        df = self._assign_sentiment_group(df)

        # =========================================
        # 1단계: 대분류별 쿼터 계산
        # =========================================
        print("\n" + "="*60)
        print("1단계: 대분류(category_1)별 쿼터 배정")
        print("="*60)

        cat1_counts = df[self.category_1_column].value_counts()
        cat1_quotas = self._calculate_quotas_with_min_floor(
            cat1_counts,
            self.target_size,
            self.category_1_min_floor
        )

        print("\n대분류별 쿼터:")
        for cat1, quota in sorted(cat1_quotas.items(), key=lambda x: -x[1]):
            original_count = cat1_counts.get(cat1, 0)
            original_pct = original_count / len(df) * 100
            quota_pct = quota / self.target_size * 100
            skip_note = " [소분류 스킵]" if cat1 in self.skip_cat2_categories else ""
            print(f"  {cat1}: {quota:,}개 ({quota_pct:.1f}%) [원본: {original_count:,}개, {original_pct:.1f}%]{skip_note}")

        # =========================================
        # 2단계: 소분류별 쿼터 계산
        # =========================================
        print("\n" + "="*60)
        print("2단계: 소분류(category_2)별 쿼터 배정")
        print("="*60)

        sampling_units = {}  # {(cat1, cat2): quota} or {(cat1, None): quota}

        for cat1, cat1_quota in cat1_quotas.items():
            print(f"\n[{cat1}] 총 쿼터: {cat1_quota}개")

            cat1_df = df[df[self.category_1_column] == cat1]

            if cat1 in self.skip_cat2_categories:
                print(f"  -> 소분류 쿼터 배정 스킵")
                sampling_units[(cat1, None)] = cat1_quota
                continue

            cat2_counts = cat1_df[self.category_2_column].value_counts()
            cat2_quotas = self._calculate_quotas_with_min_floor(
                cat2_counts,
                cat1_quota,
                self.category_2_min_floor
            )

            for cat2, quota in sorted(cat2_quotas.items(), key=lambda x: -x[1]):
                sampling_units[(cat1, cat2)] = quota
                original_count = cat2_counts.get(cat2, 0)
                original_pct = original_count / len(cat1_df) * 100
                quota_pct = quota / cat1_quota * 100
                print(f"  {cat2}: {quota:,}개 ({quota_pct:.1f}%) [원본: {original_count:,}개, {original_pct:.1f}%]")

        # =========================================
        # 3단계: 소분류 단위 자연 추출 (sentiment 강제 X)
        # =========================================
        print("\n" + "="*60)
        print("3단계: 소분류 단위 자연 추출 (sentiment 비율 강제 없음)")
        print("="*60)

        sampled_dfs = []
        remaining_pool = {}  # 추가 샘플링을 위한 잔여 풀

        for (cat1, cat2), quota in sampling_units.items():
            if cat2 is None:
                unit_df = df[df[self.category_1_column] == cat1].copy()
                label = cat1
            else:
                unit_df = df[
                    (df[self.category_1_column] == cat1) &
                    (df[self.category_2_column] == cat2)
                ].copy()
                label = f"{cat1}/{cat2}"

            if len(unit_df) == 0:
                print(f"\n[{label}] 경고: 데이터 없음")
                continue

            # 자연 추출 (복원추출 없음, 가용 범위 내)
            actual_quota = min(quota, len(unit_df))
            sampled = unit_df.sample(n=actual_quota, random_state=self.random_state)
            sampled_dfs.append(sampled)

            # 잔여 풀 저장 (추가 샘플링용)
            remaining = unit_df[~unit_df.index.isin(sampled.index)]
            remaining_pool[(cat1, cat2)] = remaining

            # sentiment 분포 출력
            sent_dist = sampled['sentiment_group'].value_counts()
            sent_str = ", ".join([f"{s}: {c}" for s, c in sent_dist.items()])
            print(f"\n[{label}] {len(sampled)}개 추출 ({sent_str})")

        # 1차 샘플 결합
        sampled_df = pd.concat(sampled_dfs, ignore_index=True)
        print(f"\n1차 샘플: {len(sampled_df):,}개")

        # =========================================
        # 4단계: 전체 레벨 sentiment 균형 조정
        # =========================================
        print("\n" + "="*60)
        print("4단계: 전체 레벨 sentiment 균형 조정")
        print("="*60)

        sampled_df = self._balance_sentiment_at_total_level(
            sampled_df, remaining_pool, df
        )

        # 불필요한 컬럼 제거
        drop_cols = ['sentiment_group', 'user_masked', 'user', 'year']
        sampled_df = sampled_df.drop(columns=drop_cols, errors='ignore')

        # 셔플
        sampled_df = sampled_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        print(f"\n최종 샘플: {len(sampled_df):,}개")

        # 검증
        self._validate_sample(sampled_df)

        return sampled_df

    def _balance_sentiment_at_total_level(
        self,
        sampled_df: pd.DataFrame,
        remaining_pool: Dict[Tuple, pd.DataFrame],
        full_df: pd.DataFrame
    ) -> pd.DataFrame:
        """전체 레벨에서 sentiment 균형 조정"""

        # 현재 sentiment 분포
        current_dist = sampled_df['sentiment_group'].value_counts()
        total_current = len(sampled_df)

        print(f"\n현재 sentiment 분포:")
        for sent in ['negative', 'neutral', 'positive']:
            count = current_dist.get(sent, 0)
            pct = count / total_current * 100
            target_pct = self.target_sentiment_distribution.get(sent, 0) * 100
            print(f"  {sent}: {count:,}개 ({pct:.1f}%) [목표: {target_pct:.0f}%]")

        # 목표 개수 계산
        target_counts = {
            sent: int(self.target_size * ratio)
            for sent, ratio in self.target_sentiment_distribution.items()
        }

        # 반올림 오차 보정
        total_target = sum(target_counts.values())
        if total_target != self.target_size:
            largest = max(target_counts, key=target_counts.get)
            target_counts[largest] += self.target_size - total_target

        print(f"\n목표 sentiment 개수:")
        for sent, count in target_counts.items():
            print(f"  {sent}: {count:,}개")

        # 각 sentiment별 과부족 계산
        adjustments = {}
        for sent in ['negative', 'neutral', 'positive']:
            current = current_dist.get(sent, 0)
            target = target_counts[sent]
            adjustments[sent] = target - current

        print(f"\n조정 필요:")
        for sent, adj in adjustments.items():
            if adj > 0:
                print(f"  {sent}: +{adj:,}개 추가 필요")
            elif adj < 0:
                print(f"  {sent}: {adj:,}개 제거 필요")
            else:
                print(f"  {sent}: 조정 불필요")

        # 부족한 sentiment 추가 확보
        additional_samples = []
        for sent, needed in adjustments.items():
            if needed > 0:
                # 잔여 풀에서 해당 sentiment 추가 확보
                available_for_sent = []
                for key, pool_df in remaining_pool.items():
                    sent_pool = pool_df[pool_df['sentiment_group'] == sent]
                    if len(sent_pool) > 0:
                        available_for_sent.append(sent_pool)

                if available_for_sent:
                    combined_pool = pd.concat(available_for_sent, ignore_index=True)
                    actual_add = min(needed, len(combined_pool))
                    if actual_add > 0:
                        added = combined_pool.sample(n=actual_add, random_state=self.random_state)
                        additional_samples.append(added)
                        print(f"  {sent}: {actual_add}개 추가 확보 (잔여 풀에서)")

                        # 잔여 풀 업데이트
                        added_indices = set(added.index)
                        for key in remaining_pool:
                            remaining_pool[key] = remaining_pool[key][
                                ~remaining_pool[key].index.isin(added_indices)
                            ]

        # 추가 샘플 결합
        if additional_samples:
            sampled_df = pd.concat([sampled_df] + additional_samples, ignore_index=True)
            print(f"\n추가 샘플 후: {len(sampled_df):,}개")

        # 초과한 sentiment 제거 (목표 크기 맞추기)
        if len(sampled_df) > self.target_size:
            excess = len(sampled_df) - self.target_size
            print(f"\n{excess}개 초과 -> 조정 중...")

            # 초과한 sentiment에서 제거
            for sent in ['positive', 'neutral', 'negative']:  # positive 먼저 제거
                current = len(sampled_df[sampled_df['sentiment_group'] == sent])
                target = target_counts[sent]
                removable = current - target

                if removable > 0 and excess > 0:
                    to_remove = min(removable, excess)
                    sent_indices = sampled_df[sampled_df['sentiment_group'] == sent].index
                    remove_indices = np.random.choice(sent_indices, size=to_remove, replace=False)
                    sampled_df = sampled_df.drop(remove_indices)
                    excess -= to_remove
                    print(f"  {sent}: {to_remove}개 제거")

                if excess <= 0:
                    break

        # 부족한 경우 (잔여 풀 소진)
        if len(sampled_df) < self.target_size:
            shortage = self.target_size - len(sampled_df)
            print(f"\n{shortage}개 부족 (잔여 풀 소진)")
            print("  -> 가용 데이터 한계로 목표보다 적은 샘플")

        return sampled_df

    def _validate_sample(self, df: pd.DataFrame):
        """샘플링 결과 검증"""
        print("\n" + "="*60)
        print("샘플링 검증")
        print("="*60)

        # 대분류 분포
        print("\n대분류 분포:")
        cat1_dist = df[self.category_1_column].value_counts()
        for cat1, count in cat1_dist.items():
            pct = count / len(df) * 100
            print(f"  {cat1}: {count:,}개 ({pct:.1f}%)")

        # 소분류 분포
        print("\n소분류 분포:")
        cat2_dist = df.groupby([self.category_1_column, self.category_2_column]).size()
        for (cat1, cat2), count in cat2_dist.items():
            pct = count / len(df) * 100
            print(f"  {cat1}/{cat2}: {count:,}개 ({pct:.1f}%)")

        # Sentiment 분포
        df_with_sent = self._assign_sentiment_group(df)
        print("\nSentiment 분포:")
        sent_dist = df_with_sent['sentiment_group'].value_counts()
        for sent in ['negative', 'neutral', 'positive']:
            count = sent_dist.get(sent, 0)
            pct = count / len(df) * 100
            target_pct = self.target_sentiment_distribution.get(sent, 0) * 100
            diff = pct - target_pct
            print(f"  {sent}: {count:,}개 ({pct:.1f}%) [목표: {target_pct:.0f}%, 차이: {diff:+.1f}%p]")

        # 상세 평점 분포
        print("\n상세 평점 분포:")
        rating_dist = df['rating'].value_counts().sort_index()
        for rating, count in rating_dist.items():
            pct = count / len(df) * 100
            print(f"  {rating}점: {count:,}개 ({pct:.1f}%)")

        # 복원추출 여부 확인
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"\n경고: 중복 샘플 {duplicates}개 발견")
        else:
            print(f"\n복원추출 없음 (중복 샘플 0개)")

        print("="*60)


def load_and_sample_reviews(
    input_path: Path,
    output_path: Path,
    target_size: int = 20000,
    category_1_column: str = 'category_1',
    category_2_column: str = 'category_2',
    category_1_min_floor: int = 600,
    category_2_min_floor: int = 200,
    skip_cat2_categories: List[str] = None,
    target_sentiment_distribution: Dict[str, float] = None,
    random_state: int = 42,
    filter_duplicates: bool = True,
    filter_foreign: bool = True,
    min_korean_ratio: float = 0.3,
    text_column: str = 'text',
    **kwargs
) -> pd.DataFrame:
    """
    리뷰 로드 및 자연 분포 기반 층화 샘플링

    Args:
        input_path: 입력 CSV 경로
        output_path: 출력 CSV 경로
        target_size: 목표 샘플 수
        category_1_column: 대분류 컬럼명
        category_2_column: 소분류 컬럼명
        category_1_min_floor: 대분류별 최소 보장 개수
        category_2_min_floor: 소분류별 최소 보장 개수
        skip_cat2_categories: 소분류 쿼터 배정 스킵할 대분류 목록
        target_sentiment_distribution: 목표 sentiment 비율 (전체 레벨)
        random_state: 랜덤 시드
        filter_duplicates: 중복 텍스트 리뷰 필터링 여부
        filter_foreign: 외국어/저품질 리뷰 필터링 여부
        min_korean_ratio: 최소 한국어 비율 (이 비율 미만이면 외국어로 판단)
        text_column: 텍스트 컬럼명

    Returns:
        샘플링된 데이터프레임
    """
    print("리뷰 로드 중...")
    df = pd.read_csv(input_path)
    print(f"로드 완료: {len(df):,}개")

    sampler = NaturalStratifiedSampler(
        target_size=target_size,
        category_1_column=category_1_column,
        category_2_column=category_2_column,
        category_1_min_floor=category_1_min_floor,
        category_2_min_floor=category_2_min_floor,
        skip_cat2_categories=skip_cat2_categories,
        target_sentiment_distribution=target_sentiment_distribution,
        random_state=random_state,
        filter_duplicates=filter_duplicates,
        filter_foreign=filter_foreign,
        min_korean_ratio=min_korean_ratio,
        text_column=text_column
    )

    sampled_df = sampler.sample(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n저장 완료: {output_path}")

    return sampled_df
