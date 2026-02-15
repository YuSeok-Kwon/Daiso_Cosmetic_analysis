"""
다이소 외국인 분석 실행 스크립트 (v3)

출력 파일 (총 13개):
- 월별 분석: 12개 (01월 ~ 12월)
- 연간 종합: 1개

각 파일 내 시트 구성:
- 시간대별_전체 (방법 A)
- 시간대별_평일
- 시간대별_주말
- 구별_평균_전체 (방법 B)
- 구별_평균_평일
- 구별_평균_주말
- 구별_PH_전체 (방법 C)
- 월별_PH_추이 (연간 파일만)
- 복합순위 (S-DoT 결합)
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from foreigner_analysis_v3 import (
    BASE_PATH, DAISO_HOURS, GU_CODE_MAP,
    load_foreigner_data, process_foreigner_data_v3,
    analyze_by_hour, analyze_by_hour_weekday,
    analyze_average_snapshot, analyze_average_snapshot_weekday,
    analyze_person_hour, analyze_person_hour_monthly,
    load_sdot_data, process_sdot_data, aggregate_sdot_by_gu,
    merge_with_sdot, validate_results, get_date_info
)
import warnings
warnings.filterwarnings('ignore')


def run_monthly_analysis(month_folder, output_dir, sdot_agg=None):
    """
    월별 분석 실행

    Parameters:
        month_folder: 'TEMP_FOREIGNER_202501' 형식
        output_dir: 결과 저장 경로
        sdot_agg: S-DoT 집계 데이터 (optional)

    Returns:
        dict: 월별 요약 정보
    """
    month_name = month_folder.replace('TEMP_FOREIGNER_', '')
    print(f"\n{'='*70}")
    print(f"📅 {month_name[:4]}년 {month_name[4:]}월 분석")
    print(f"{'='*70}")

    try:
        # 데이터 로드
        df_raw = load_foreigner_data(month_folders=[month_folder])
        df_processed = process_foreigner_data_v3(df_raw)

        date_info = get_date_info(df_processed)
        total_days = date_info['total_days']
        weekday_days = date_info['weekday_days']
        weekend_days = date_info['weekend_days']

        print(f"  분석 기간: {total_days}일 (평일 {weekday_days}일, 주말 {weekend_days}일)")

        results = {}

        # ============================================
        # 방법 A: 시간대별 분석
        # ============================================
        print("  [A] 시간대별 분석...")
        hourly_detail, pivot_all = analyze_by_hour(df_processed, total_days)
        results['시간대별_전체'] = pivot_all.reset_index()

        # 평일/주말 구분
        hourly_weekday = analyze_by_hour_weekday(df_processed, weekday_days, weekend_days)
        if '평일' in hourly_weekday:
            results['시간대별_평일'] = hourly_weekday['평일'].reset_index()
        if '주말' in hourly_weekday:
            results['시간대별_주말'] = hourly_weekday['주말'].reset_index()

        # ============================================
        # 방법 B: 평균 스냅샷
        # ============================================
        print("  [B] 평균 스냅샷 분석...")
        avg_snapshot = analyze_average_snapshot(df_processed, total_days)
        results['구별_평균_전체'] = avg_snapshot

        # 평일/주말 구분
        avg_weekday = analyze_average_snapshot_weekday(df_processed, weekday_days, weekend_days)
        if '평일' in avg_weekday:
            results['구별_평균_평일'] = avg_weekday['평일']
        if '주말' in avg_weekday:
            results['구별_평균_주말'] = avg_weekday['주말']

        # ============================================
        # 방법 C: Person-Hour
        # ============================================
        print("  [C] Person-Hour 분석...")
        person_hour = analyze_person_hour(df_processed, total_days)
        results['구별_PH_전체'] = person_hour

        # ============================================
        # S-DoT 결합 (있을 경우)
        # ============================================
        if sdot_agg is not None and len(sdot_agg) > 0:
            print("  [+] S-DoT 결합 분석...")
            merged = merge_with_sdot(avg_snapshot, sdot_agg)
            merged_sorted = merged.sort_values('복합점수', ascending=False, na_position='last')
            results['복합순위'] = merged_sorted

        # ============================================
        # 결과 저장
        # ============================================
        output_file = os.path.join(output_dir, f'daiso_analysis_{month_name}.xlsx')

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            for sheet_name, df in results.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"  저장 완료: {output_file}")

        # 요약 정보 반환
        total_foreigner = avg_snapshot['평균_외국인'].sum()
        total_chinese = avg_snapshot['평균_중국인'].sum()
        total_other = avg_snapshot['평균_비중국'].sum()
        china_ratio = (total_chinese / (total_chinese + total_other) * 100) if (total_chinese + total_other) > 0 else 0

        return {
            '월': month_name,
            '일수': total_days,
            '평일일수': weekday_days,
            '주말일수': weekend_days,
            '일평균_외국인': total_foreigner,
            '일평균_중국인': total_chinese,
            '일평균_비중국': total_other,
            '중국인비율(%)': round(china_ratio, 2),
            '성공': True
        }

    except Exception as e:
        print(f"  오류 발생: {e}")
        return {
            '월': month_name,
            '성공': False,
            '오류': str(e)
        }


def run_yearly_analysis(monthly_summaries, all_gu_data, output_dir, sdot_agg=None):
    """
    연간 종합 분석

    Parameters:
        monthly_summaries: 월별 요약 리스트
        all_gu_data: 전체 기간 처리된 데이터
        output_dir: 결과 저장 경로
        sdot_agg: S-DoT 집계 데이터
    """
    print(f"\n{'='*70}")
    print("📊 2025년 연간 종합 분석")
    print(f"{'='*70}")

    results = {}

    # ============================================
    # 월별 요약
    # ============================================
    monthly_df = pd.DataFrame([m for m in monthly_summaries if m.get('성공', False)])
    results['월별_요약'] = monthly_df

    # ============================================
    # 전체 기간 분석
    # ============================================
    date_info = get_date_info(all_gu_data)
    total_days = date_info['total_days']
    weekday_days = date_info['weekday_days']
    weekend_days = date_info['weekend_days']

    print(f"  총 분석 기간: {total_days}일")

    # 방법 A: 시간대별
    print("  [A] 연간 시간대별 분석...")
    _, pivot_yearly = analyze_by_hour(all_gu_data, total_days)
    results['시간대별_연간'] = pivot_yearly.reset_index()

    hourly_weekday = analyze_by_hour_weekday(all_gu_data, weekday_days, weekend_days)
    if '평일' in hourly_weekday:
        results['시간대별_연간_평일'] = hourly_weekday['평일'].reset_index()
    if '주말' in hourly_weekday:
        results['시간대별_연간_주말'] = hourly_weekday['주말'].reset_index()

    # 방법 B: 평균 스냅샷
    print("  [B] 연간 평균 스냅샷...")
    avg_yearly = analyze_average_snapshot(all_gu_data, total_days)
    results['구별_평균_연간'] = avg_yearly

    avg_weekday = analyze_average_snapshot_weekday(all_gu_data, weekday_days, weekend_days)
    if '평일' in avg_weekday:
        results['구별_평균_연간_평일'] = avg_weekday['평일']
    if '주말' in avg_weekday:
        results['구별_평균_연간_주말'] = avg_weekday['주말']

    # 방법 C: Person-Hour
    print("  [C] 연간 Person-Hour...")
    ph_yearly = analyze_person_hour(all_gu_data, total_days)
    results['구별_PH_연간'] = ph_yearly

    # 월별 PH 추이
    ph_monthly = analyze_person_hour_monthly(all_gu_data)
    results['월별_PH_추이'] = ph_monthly

    # S-DoT 결합
    if sdot_agg is not None and len(sdot_agg) > 0:
        print("  [+] S-DoT 결합...")
        merged = merge_with_sdot(avg_yearly, sdot_agg)
        merged_sorted = merged.sort_values('복합점수', ascending=False, na_position='last')
        results['복합순위_연간'] = merged_sorted

    # 검증
    print("  [검증] 방법 B × 13 ≈ 방법 C...")
    validation = validate_results(avg_yearly, ph_yearly)
    results['검증_B_vs_C'] = validation

    # ============================================
    # 결과 저장
    # ============================================
    output_file = os.path.join(output_dir, 'daiso_analysis_2025_연간종합.xlsx')

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sheet_name, df in results.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\n  연간 종합 저장 완료: {output_file}")

    # ============================================
    # 최종 요약 출력
    # ============================================
    print(f"\n{'='*70}")
    print("🎯 최종 분석 요약")
    print(f"{'='*70}")

    yearly_avg = monthly_df['일평균_외국인'].mean() if len(monthly_df) > 0 else 0
    yearly_china = monthly_df['일평균_중국인'].sum() if len(monthly_df) > 0 else 0
    yearly_other = monthly_df['일평균_비중국'].sum() if len(monthly_df) > 0 else 0
    yearly_ratio = (yearly_china / (yearly_china + yearly_other) * 100) if (yearly_china + yearly_other) > 0 else 0

    print(f"""
[핵심 수치 - 다이소 영업시간(10~22시) 기준]
• 서울시 일평균 단기체류 외국인 (평균 스냅샷): {yearly_avg:,.0f}명
• 연간 연인원 추정 (×365일): 약 {yearly_avg * 365 / 10000:.0f}만 명
• 중국인 비율: {yearly_ratio:.1f}%
• 분석 시간대: 13개 (10시~22시)
""")

    print("▶ 외국인 밀집 TOP 5 구 (평균 스냅샷 기준):")
    for i, row in avg_yearly.head(5).iterrows():
        print(f"  {row['자치구']}: 일평균 {row['평균_외국인']:,.0f}명, 중국인비율 {row['중국인비율(%)']}%")

    if sdot_agg is not None and '복합순위_연간' in results:
        print("\n▶ 외국인 + 유동량 복합 TOP 5 구:")
        top5_merged = results['복합순위_연간'].dropna(subset=['복합점수']).head(5)
        for _, row in top5_merged.iterrows():
            print(f"  {row['자치구']}: 외국인 {row['평균_외국인']:,.0f}명, "
                  f"유동량 {row['일평균_방문자']:,.0f}명/일")

    return results


def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("🏪 다이소 외국인 분석 v3 (영업시간 10~22시 기준)")
    print("=" * 70)
    print(f"분석 대상 시간대: {DAISO_HOURS}")
    print(f"출력 파일: 월별 12개 + 연간종합 1개 = 총 13개")

    # 출력 디렉토리 설정
    output_dir = os.path.join(BASE_PATH, 'analysis_results')
    os.makedirs(output_dir, exist_ok=True)

    # 월별 폴더 목록
    all_months = [
        'TEMP_FOREIGNER_202501', 'TEMP_FOREIGNER_202502', 'TEMP_FOREIGNER_202503',
        'TEMP_FOREIGNER_202504', 'TEMP_FOREIGNER_202505', 'TEMP_FOREIGNER_202506',
        'TEMP_FOREIGNER_202507', 'TEMP_FOREIGNER_202508', 'TEMP_FOREIGNER_202509',
        'TEMP_FOREIGNER_202510', 'TEMP_FOREIGNER_202511', 'TEMP_FOREIGNER_202512'
    ]

    # ============================================
    # S-DoT 데이터 로드 (전체 기간)
    # ============================================
    print("\n[사전 작업] S-DoT 데이터 로드 중...")
    try:
        sdot_files = glob.glob(os.path.join(BASE_PATH, 'S-DoT_WALK_*.csv'))
        if sdot_files:
            sdot_raw = load_sdot_data(files=sdot_files)
            sdot_processed = process_sdot_data(sdot_raw)

            # 전체 기간 일수 계산 (파일 수 × 7일)
            total_sdot_days = len(sdot_files) * 7
            sdot_agg = aggregate_sdot_by_gu(sdot_processed, days=total_sdot_days)
            print(f"  S-DoT 로드 완료: {len(sdot_agg)} 구, {total_sdot_days}일")
        else:
            sdot_agg = None
            print("  S-DoT 파일 없음")
    except Exception as e:
        print(f"  S-DoT 로드 실패: {e}")
        sdot_agg = None

    # ============================================
    # 월별 분석 (12개 파일)
    # ============================================
    print("\n" + "=" * 70)
    print("📅 월별 분석 시작 (12개 파일 생성)")
    print("=" * 70)

    monthly_summaries = []
    all_processed_data = []

    for month_folder in all_months:
        summary = run_monthly_analysis(month_folder, output_dir, sdot_agg)
        monthly_summaries.append(summary)

        # 연간 분석용 데이터 수집
        if summary.get('성공', False):
            try:
                df_raw = load_foreigner_data(month_folders=[month_folder])
                df_processed = process_foreigner_data_v3(df_raw)
                all_processed_data.append(df_processed)
            except:
                pass

    # ============================================
    # 연간 종합 분석 (1개 파일)
    # ============================================
    if all_processed_data:
        print("\n[연간 데이터 병합 중...]")
        all_gu_data = pd.concat(all_processed_data, ignore_index=True)
        print(f"  전체 레코드 수: {len(all_gu_data):,}")

        run_yearly_analysis(monthly_summaries, all_gu_data, output_dir, sdot_agg)

    # ============================================
    # 최종 결과 확인
    # ============================================
    print("\n" + "=" * 70)
    print("📁 생성된 파일 목록")
    print("=" * 70)

    result_files = glob.glob(os.path.join(output_dir, 'daiso_analysis_*.xlsx'))
    for f in sorted(result_files):
        print(f"  {os.path.basename(f)}")

    print(f"\n총 {len(result_files)}개 파일 생성 완료!")

    # 방법론 안내
    print("\n" + "=" * 70)
    print("📋 분석 방법론 안내")
    print("=" * 70)
    print("""
[데이터 해석]
• 통신데이터 = 시간대별 체류인구 스냅샷 (같은 사람이 여러 시간대에 카운트 가능)
• 다이소 영업시간(10~22시) 기준 분석

[3가지 분석 방법]
A. 시간대별 분석
   - 10~22시 각 시간대별 외국인 체류인구
   - 용도: 피크타임 파악

B. 평균 스냅샷 (주요 랭킹용)
   - 10~22시 내 평균 외국인 체류인구
   - 공식: Σ(10~22시 외국인) / 13시간 / 일수
   - 용도: 구별 순위 비교

C. Person-Hour (체류 가치)
   - 10~22시 합산 (외국인 × 시간)
   - 공식: Σ(10~22시 외국인) / 일수
   - 용도: 리테일 가치 분석

[주의사항]
• '*' 마스킹 값: 0으로 대체 (최소치 추정)
• 방법 B × 13 ≈ 방법 C (검증용)
""")

    print("=" * 70)
    print("분석 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
