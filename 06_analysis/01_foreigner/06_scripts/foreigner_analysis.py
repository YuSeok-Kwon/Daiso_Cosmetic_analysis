"""
서울시 외국인 밀집 지역 분석 파이프라인 (v3 - 다이소 영업시간 기준)

수정 사항:
- 기존 v2: 12시(정오) 단일 시간대만 사용
- v3: 다이소 영업시간 10~22시 (13개 시간대) 기준 분석

3가지 분석 방법 제공:
- 방법 A: 시간대별 분석 (피크타임 파악)
- 방법 B: 평균 스냅샷 (구별 랭킹 비교)
- 방법 C: Person-Hour (체류 가치 분석)
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 설정
# ============================================================
BASE_PATH = '/Users/yu_seok/Documents/workspace/nbCamp/Project/Why-pi/TEMP_FOREIGNER'

# 다이소 영업시간: 10시 ~ 22시 (13개 시간대)
DAISO_HOURS = list(range(10, 23))  # [10, 11, 12, ..., 22]

# 서울시 자치구 코드 매핑
GU_CODE_MAP = {
    '11110': '종로구', '11140': '중구', '11170': '용산구', '11200': '성동구',
    '11215': '광진구', '11230': '동대문구', '11260': '중랑구', '11290': '성북구',
    '11305': '강북구', '11320': '도봉구', '11350': '노원구', '11380': '은평구',
    '11410': '서대문구', '11440': '마포구', '11470': '양천구', '11500': '강서구',
    '11530': '구로구', '11545': '금천구', '11560': '영등포구', '11590': '동작구',
    '11620': '관악구', '11650': '서초구', '11680': '강남구', '11710': '송파구',
    '11740': '강동구'
}

# S-DoT 영문 구명 → 한글 매핑
SDOT_GU_MAP = {
    'Jongno-gu': '종로구', 'Jung-gu': '중구', 'Yongsan-gu': '용산구',
    'Seongdong-gu': '성동구', 'Gwangjin-gu': '광진구', 'Dongdaemun-gu': '동대문구',
    'Jungnang-gu': '중랑구', 'Seongbuk-gu': '성북구', 'Gangbuk-gu': '강북구',
    'Dobong-gu': '도봉구', 'Nowon-gu': '노원구', 'Eunpyeong-gu': '은평구',
    'Seodaemun-gu': '서대문구', 'Mapo-gu': '마포구', 'Yangcheon-gu': '양천구',
    'Gangseo-gu': '강서구', 'Guro-gu': '구로구', 'Geumcheon-gu': '금천구',
    'Yeongdeungpo-gu': '영등포구', 'Dongjak-gu': '동작구', 'Gwanak-gu': '관악구',
    'Seocho-gu': '서초구', 'Gangnam-gu': '강남구', 'Songpa-gu': '송파구',
    'Gangdong-gu': '강동구'
}


# ============================================================
# 2. 데이터 로드
# ============================================================

def load_foreigner_data(date_range=None, month_folders=None):
    """통신데이터 로드 및 병합"""
    all_files = []

    if month_folders:
        for folder in month_folders:
            folder_path = os.path.join(BASE_PATH, folder)
            files = glob.glob(os.path.join(folder_path, 'TEMP_FOREIGNER_*.csv'))
            all_files.extend(files)
    else:
        pattern = os.path.join(BASE_PATH, 'TEMP_FOREIGNER_*/TEMP_FOREIGNER_*.csv')
        all_files = glob.glob(pattern)

    if date_range:
        start, end = date_range
        filtered = []
        for f in all_files:
            fname = os.path.basename(f)
            date_str = fname.replace('TEMP_FOREIGNER_', '').replace('.csv', '')
            if start <= date_str <= end:
                filtered.append(f)
        all_files = filtered

    print(f"[INFO] 로드할 파일 수: {len(all_files)}")

    dfs = []
    for f in sorted(all_files):
        try:
            df = pd.read_csv(f, encoding='cp949')
            df.columns = [col.replace('\ufeff', '').replace('?', '').strip('"') for col in df.columns]
            dfs.append(df)
        except Exception as e:
            pass

    if not dfs:
        raise ValueError("로드된 데이터가 없습니다.")

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] 총 레코드 수: {len(df_all):,}")

    return df_all


# ============================================================
# 3. 데이터 전처리 (v3 - 다이소 영업시간 필터)
# ============================================================

def process_foreigner_data_v3(df):
    """
    통신데이터 전처리 (v3 - 다이소 영업시간 10~22시 필터링)

    Returns:
        DataFrame with columns: 기준일, 시간대, 행정동코드, 자치구,
                               총생활인구수, 중국인체류인구수, 중국외외국인체류인구수, 외국인체류인구수
    """
    df = df.copy()

    # 컬럼명 정리
    col_renames = {}
    for col in df.columns:
        if '기준일' in col:
            col_renames[col] = '기준일'
        elif '시간대' in col:
            col_renames[col] = '시간대'
        elif '행정동코드' in col:
            col_renames[col] = '행정동코드'
        elif '총생활인구수' in col:
            col_renames[col] = '총생활인구수'
        elif '중국인체류인구수' in col:
            col_renames[col] = '중국인체류인구수'
        elif '중국외외국인체류인구수' in col:
            col_renames[col] = '중국외외국인체류인구수'

    df.rename(columns=col_renames, inplace=True)

    # 시간대를 숫자로 변환
    df['시간대'] = pd.to_numeric(df['시간대'], errors='coerce')

    # 다이소 영업시간(10~22시) 필터링
    df = df[df['시간대'].isin(DAISO_HOURS)].copy()
    print(f"[INFO] 다이소 영업시간(10~22시) 필터링 완료: {len(df):,} 레코드")

    # '*' 처리 (마스킹 값 → 0)
    for col in ['중국인체류인구수', '중국외외국인체류인구수']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace('*', np.nan), errors='coerce').fillna(0)

    df['총생활인구수'] = pd.to_numeric(df['총생활인구수'], errors='coerce').fillna(0)

    # 외국인 체류인구수 계산
    df['외국인체류인구수'] = df['중국인체류인구수'] + df['중국외외국인체류인구수']

    # 구 코드 매핑
    df['행정동코드'] = df['행정동코드'].astype(str)
    df['구코드'] = df['행정동코드'].str[:5]
    df['자치구'] = df['구코드'].map(GU_CODE_MAP)

    # 요일 정보 추가
    df['기준일'] = df['기준일'].astype(str)
    df['날짜'] = pd.to_datetime(df['기준일'], format='%Y%m%d')
    df['요일'] = df['날짜'].dt.dayofweek  # 0=월, 6=일
    df['주말여부'] = df['요일'].isin([5, 6])  # 토,일 = True

    return df


# ============================================================
# 4. 방법 A: 시간대별 분석
# ============================================================

def analyze_by_hour(df, days):
    """
    방법 A: 시간대별 분석 (피크타임 파악)

    Returns:
        DataFrame: 구별 × 시간대별 일평균 외국인 체류인구 + 피크시간
    """
    df = df[df['자치구'].notna()].copy()

    # 구별 × 시간대별 집계
    hourly = df.groupby(['자치구', '시간대']).agg({
        '외국인체류인구수': 'sum',
        '중국인체류인구수': 'sum',
        '중국외외국인체류인구수': 'sum',
        '총생활인구수': 'sum'
    }).reset_index()

    # 일평균 계산
    hourly['일평균_외국인'] = (hourly['외국인체류인구수'] / days).round(0)
    hourly['일평균_중국인'] = (hourly['중국인체류인구수'] / days).round(0)
    hourly['일평균_비중국'] = (hourly['중국외외국인체류인구수'] / days).round(0)
    hourly['일평균_총생활인구'] = (hourly['총생활인구수'] / days).round(0)

    # 피벗 테이블 (구 × 시간대)
    pivot = hourly.pivot_table(
        index='자치구',
        columns='시간대',
        values='일평균_외국인',
        aggfunc='sum'
    ).fillna(0)

    # 피크시간 찾기
    pivot['피크시간'] = pivot[DAISO_HOURS].idxmax(axis=1)
    pivot['피크_외국인수'] = pivot[DAISO_HOURS].max(axis=1)

    return hourly, pivot


def analyze_by_hour_weekday(df, days_weekday, days_weekend):
    """
    시간대별 분석 - 평일/주말 구분
    """
    df = df[df['자치구'].notna()].copy()

    results = {}

    # 평일 분석
    df_weekday = df[~df['주말여부']].copy()
    if len(df_weekday) > 0:
        hourly_wd = df_weekday.groupby(['자치구', '시간대']).agg({
            '외국인체류인구수': 'sum'
        }).reset_index()
        hourly_wd['일평균_외국인'] = (hourly_wd['외국인체류인구수'] / days_weekday).round(0)

        pivot_wd = hourly_wd.pivot_table(
            index='자치구', columns='시간대', values='일평균_외국인'
        ).fillna(0)
        pivot_wd['피크시간'] = pivot_wd[DAISO_HOURS].idxmax(axis=1)
        results['평일'] = pivot_wd

    # 주말 분석
    df_weekend = df[df['주말여부']].copy()
    if len(df_weekend) > 0:
        hourly_we = df_weekend.groupby(['자치구', '시간대']).agg({
            '외국인체류인구수': 'sum'
        }).reset_index()
        hourly_we['일평균_외국인'] = (hourly_we['외국인체류인구수'] / days_weekend).round(0)

        pivot_we = hourly_we.pivot_table(
            index='자치구', columns='시간대', values='일평균_외국인'
        ).fillna(0)
        pivot_we['피크시간'] = pivot_we[DAISO_HOURS].idxmax(axis=1)
        results['주말'] = pivot_we

    return results


# ============================================================
# 5. 방법 B: 평균 스냅샷 분석
# ============================================================

def analyze_average_snapshot(df, days):
    """
    방법 B: 평균 스냅샷 (10~22시 평균)

    지표:
    - 일평균_외국인 = Σ(10~22시 외국인체류인구) / 13시간 / 일수
    - 중국인비율 = 일평균_중국인 / 일평균_외국인 × 100

    Returns:
        DataFrame: 구별 평균 스냅샷 데이터
    """
    df = df[df['자치구'].notna()].copy()

    # 구별 × 시간대별 집계 후 시간대 평균
    hourly = df.groupby(['자치구', '시간대']).agg({
        '외국인체류인구수': 'sum',
        '중국인체류인구수': 'sum',
        '중국외외국인체류인구수': 'sum',
        '총생활인구수': 'sum'
    }).reset_index()

    # 시간대 수 (13개)
    num_hours = len(DAISO_HOURS)

    # 구별 평균 (시간대별 평균의 평균 = 하루 평균 상태)
    avg_snapshot = hourly.groupby('자치구').agg({
        '외국인체류인구수': 'sum',
        '중국인체류인구수': 'sum',
        '중국외외국인체류인구수': 'sum',
        '총생활인구수': 'sum'
    }).reset_index()

    # 평균 계산: 전체 합 / 시간대 수 / 일수
    avg_snapshot['평균_외국인'] = (avg_snapshot['외국인체류인구수'] / num_hours / days).round(0)
    avg_snapshot['평균_중국인'] = (avg_snapshot['중국인체류인구수'] / num_hours / days).round(0)
    avg_snapshot['평균_비중국'] = (avg_snapshot['중국외외국인체류인구수'] / num_hours / days).round(0)
    avg_snapshot['평균_총생활인구'] = (avg_snapshot['총생활인구수'] / num_hours / days).round(0)

    # 중국인 비율
    total_foreigner = avg_snapshot['평균_중국인'] + avg_snapshot['평균_비중국']
    avg_snapshot['중국인비율(%)'] = np.where(
        total_foreigner > 0,
        (avg_snapshot['평균_중국인'] / total_foreigner * 100).round(2),
        0
    )

    # 외국인 비율 (총생활인구 대비)
    avg_snapshot['외국인비율(%)'] = np.where(
        avg_snapshot['평균_총생활인구'] > 0,
        (avg_snapshot['평균_외국인'] / avg_snapshot['평균_총생활인구'] * 100).round(2),
        0
    )

    # 컬럼 정리
    result = avg_snapshot[[
        '자치구', '평균_외국인', '평균_중국인', '평균_비중국',
        '중국인비율(%)', '외국인비율(%)', '평균_총생활인구'
    ]].copy()

    result = result.sort_values('평균_외국인', ascending=False)

    return result


def analyze_average_snapshot_weekday(df, days_weekday, days_weekend):
    """
    평균 스냅샷 - 평일/주말 구분
    """
    results = {}
    num_hours = len(DAISO_HOURS)

    # 평일
    df_weekday = df[~df['주말여부'] & df['자치구'].notna()].copy()
    if len(df_weekday) > 0 and days_weekday > 0:
        agg_wd = df_weekday.groupby('자치구').agg({
            '외국인체류인구수': 'sum',
            '중국인체류인구수': 'sum',
            '중국외외국인체류인구수': 'sum'
        }).reset_index()

        agg_wd['평균_외국인'] = (agg_wd['외국인체류인구수'] / num_hours / days_weekday).round(0)
        agg_wd['평균_중국인'] = (agg_wd['중국인체류인구수'] / num_hours / days_weekday).round(0)
        agg_wd['평균_비중국'] = (agg_wd['중국외외국인체류인구수'] / num_hours / days_weekday).round(0)

        total = agg_wd['평균_중국인'] + agg_wd['평균_비중국']
        agg_wd['중국인비율(%)'] = np.where(total > 0, (agg_wd['평균_중국인'] / total * 100).round(2), 0)

        results['평일'] = agg_wd[['자치구', '평균_외국인', '평균_중국인', '평균_비중국', '중국인비율(%)']].sort_values('평균_외국인', ascending=False)

    # 주말
    df_weekend = df[df['주말여부'] & df['자치구'].notna()].copy()
    if len(df_weekend) > 0 and days_weekend > 0:
        agg_we = df_weekend.groupby('자치구').agg({
            '외국인체류인구수': 'sum',
            '중국인체류인구수': 'sum',
            '중국외외국인체류인구수': 'sum'
        }).reset_index()

        agg_we['평균_외국인'] = (agg_we['외국인체류인구수'] / num_hours / days_weekend).round(0)
        agg_we['평균_중국인'] = (agg_we['중국인체류인구수'] / num_hours / days_weekend).round(0)
        agg_we['평균_비중국'] = (agg_we['중국외외국인체류인구수'] / num_hours / days_weekend).round(0)

        total = agg_we['평균_중국인'] + agg_we['평균_비중국']
        agg_we['중국인비율(%)'] = np.where(total > 0, (agg_we['평균_중국인'] / total * 100).round(2), 0)

        results['주말'] = agg_we[['자치구', '평균_외국인', '평균_중국인', '평균_비중국', '중국인비율(%)']].sort_values('평균_외국인', ascending=False)

    return results


# ============================================================
# 6. 방법 C: Person-Hour 분석
# ============================================================

def analyze_person_hour(df, days):
    """
    방법 C: Person-Hour 분석 (체류 가치)

    지표:
    - 외국인_PH = Σ(10~22시 외국인체류인구) / 일수  (하루 기준 Person-Hour)
    - 외국인비율_PH = Σ외국인_PH / Σ총생활인구_PH × 100
    - 일평균_PH = 외국인_PH (= 시간대 합산이므로 이미 일평균)

    Returns:
        DataFrame: 구별 Person-Hour 데이터
    """
    df = df[df['자치구'].notna()].copy()

    # 구별 합산 (시간대 합계)
    ph_data = df.groupby('자치구').agg({
        '외국인체류인구수': 'sum',
        '중국인체류인구수': 'sum',
        '중국외외국인체류인구수': 'sum',
        '총생활인구수': 'sum'
    }).reset_index()

    # 일평균 Person-Hour
    ph_data['외국인_PH'] = (ph_data['외국인체류인구수'] / days).round(0)
    ph_data['중국인_PH'] = (ph_data['중국인체류인구수'] / days).round(0)
    ph_data['비중국_PH'] = (ph_data['중국외외국인체류인구수'] / days).round(0)
    ph_data['총생활인구_PH'] = (ph_data['총생활인구수'] / days).round(0)

    # 외국인 비율 (Person-Hour 기준)
    ph_data['외국인비율_PH(%)'] = np.where(
        ph_data['총생활인구_PH'] > 0,
        (ph_data['외국인_PH'] / ph_data['총생활인구_PH'] * 100).round(2),
        0
    )

    # 중국인 비율
    total_foreigner = ph_data['중국인_PH'] + ph_data['비중국_PH']
    ph_data['중국인비율_PH(%)'] = np.where(
        total_foreigner > 0,
        (ph_data['중국인_PH'] / total_foreigner * 100).round(2),
        0
    )

    # 컬럼 정리
    result = ph_data[[
        '자치구', '외국인_PH', '중국인_PH', '비중국_PH',
        '총생활인구_PH', '외국인비율_PH(%)', '중국인비율_PH(%)'
    ]].copy()

    result = result.sort_values('외국인_PH', ascending=False)

    return result


def analyze_person_hour_monthly(df):
    """
    월별 Person-Hour 추이 분석
    """
    df = df[df['자치구'].notna()].copy()

    # 월 정보 추출
    df['월'] = df['기준일'].str[:6]

    # 월별 집계
    monthly = df.groupby('월').agg({
        '외국인체류인구수': 'sum',
        '중국인체류인구수': 'sum',
        '중국외외국인체류인구수': 'sum',
        '총생활인구수': 'sum',
        '기준일': 'nunique'
    }).reset_index()

    monthly.rename(columns={'기준일': '일수'}, inplace=True)

    # 일평균 Person-Hour
    monthly['일평균_외국인_PH'] = (monthly['외국인체류인구수'] / monthly['일수']).round(0)
    monthly['일평균_총생활인구_PH'] = (monthly['총생활인구수'] / monthly['일수']).round(0)

    # 외국인 비율
    monthly['외국인비율_PH(%)'] = np.where(
        monthly['일평균_총생활인구_PH'] > 0,
        (monthly['일평균_외국인_PH'] / monthly['일평균_총생활인구_PH'] * 100).round(2),
        0
    )

    return monthly[['월', '일평균_외국인_PH', '일평균_총생활인구_PH', '외국인비율_PH(%)', '일수']]


# ============================================================
# 7. S-DoT 데이터 처리
# ============================================================

def load_sdot_data(files=None):
    """S-DoT 유동인구 데이터 로드"""
    if files:
        all_files = files
    else:
        all_files = glob.glob(os.path.join(BASE_PATH, 'S-DoT_WALK_*.csv'))

    print(f"[INFO] S-DoT 파일 수: {len(all_files)}")

    dfs = []
    for f in sorted(all_files):
        try:
            for enc in ['utf-8', 'cp949', 'euc-kr']:
                try:
                    df = pd.read_csv(f, encoding=enc)
                    dfs.append(df)
                    break
                except:
                    continue
        except Exception as e:
            pass

    if not dfs:
        raise ValueError("S-DoT 데이터가 없습니다.")

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] S-DoT 총 레코드 수: {len(df_all):,}")

    return df_all


def process_sdot_data(df):
    """S-DoT 데이터 전처리"""
    df = df.copy()

    df['자치구_한글'] = df['자치구'].map(SDOT_GU_MAP)
    df['방문자수'] = pd.to_numeric(df['방문자수'], errors='coerce').fillna(0)

    return df


def aggregate_sdot_by_gu(df, days=1):
    """S-DoT 구별 집계"""
    df = df[df['자치구_한글'].notna()].copy()

    agg_df = df.groupby('자치구_한글').agg({
        '방문자수': 'sum',
        '시리얼': 'nunique'
    }).reset_index()

    agg_df.rename(columns={
        '자치구_한글': '자치구',
        '방문자수': 'S-DoT_방문자수_합',
        '시리얼': '센서수'
    }, inplace=True)

    agg_df['일평균_방문자'] = (agg_df['S-DoT_방문자수_합'] / days).round(0)
    agg_df['센서당_평균방문자'] = (agg_df['S-DoT_방문자수_합'] / agg_df['센서수']).round(2)

    return agg_df


# ============================================================
# 8. 복합 분석 (S-DoT 결합)
# ============================================================

def merge_with_sdot(foreigner_df, sdot_df):
    """외국인 데이터 + S-DoT 결합 및 복합 순위 계산"""
    merged = pd.merge(foreigner_df, sdot_df, on='자치구', how='outer')

    # 복합 점수 계산 (정규화)
    df_temp = merged.dropna(subset=['평균_외국인', '일평균_방문자']).copy()

    if len(df_temp) > 0:
        df_temp['외국인_정규화'] = (df_temp['평균_외국인'] - df_temp['평균_외국인'].min()) / \
                                  (df_temp['평균_외국인'].max() - df_temp['평균_외국인'].min() + 1e-10)
        df_temp['유동량_정규화'] = (df_temp['일평균_방문자'] - df_temp['일평균_방문자'].min()) / \
                                  (df_temp['일평균_방문자'].max() - df_temp['일평균_방문자'].min() + 1e-10)
        df_temp['복합점수'] = df_temp['외국인_정규화'] + df_temp['유동량_정규화']

        merged = pd.merge(merged, df_temp[['자치구', '복합점수']], on='자치구', how='left')

    return merged


# ============================================================
# 9. 검증 함수
# ============================================================

def validate_results(avg_snapshot, person_hour):
    """
    검증: 방법 B(평균×13) ≈ 방법 C(합계) 확인
    """
    print("\n[검증] 방법 B × 13 ≈ 방법 C 확인")
    print("-" * 60)

    # 병합
    validation = pd.merge(
        avg_snapshot[['자치구', '평균_외국인']],
        person_hour[['자치구', '외국인_PH']],
        on='자치구'
    )

    validation['평균×13'] = validation['평균_외국인'] * 13
    validation['차이(%)'] = np.abs(validation['평균×13'] - validation['외국인_PH']) / validation['외국인_PH'] * 100

    print(validation[['자치구', '평균_외국인', '평균×13', '외국인_PH', '차이(%)']].head(10).to_string(index=False))

    avg_diff = validation['차이(%)'].mean()
    print(f"\n평균 차이: {avg_diff:.2f}% (0%에 가까울수록 정확)")

    return validation


# ============================================================
# 10. 유틸리티 함수
# ============================================================

def get_date_info(df):
    """날짜 정보 추출"""
    dates = df['기준일'].unique()
    days_total = len(dates)

    # 요일 정보
    if '요일' in df.columns:
        weekday_dates = df[~df['주말여부']]['기준일'].nunique()
        weekend_dates = df[df['주말여부']]['기준일'].nunique()
    else:
        weekday_dates = 0
        weekend_dates = 0

    return {
        'total_days': days_total,
        'weekday_days': weekday_dates,
        'weekend_days': weekend_dates
    }


if __name__ == "__main__":
    # 테스트: 12월 1~7일 분석
    print("=" * 70)
    print("v3 분석 모듈 테스트 (다이소 영업시간 10~22시)")
    print("=" * 70)

    # 데이터 로드
    df_raw = load_foreigner_data(date_range=('20251201', '20251207'))
    df_processed = process_foreigner_data_v3(df_raw)

    date_info = get_date_info(df_processed)
    print(f"\n분석 기간: {date_info['total_days']}일 (평일 {date_info['weekday_days']}일, 주말 {date_info['weekend_days']}일)")

    # 방법 A: 시간대별 분석
    print("\n[방법 A] 시간대별 분석")
    hourly, pivot = analyze_by_hour(df_processed, date_info['total_days'])
    print(pivot.head())

    # 방법 B: 평균 스냅샷
    print("\n[방법 B] 평균 스냅샷")
    avg_snapshot = analyze_average_snapshot(df_processed, date_info['total_days'])
    print(avg_snapshot.head(10).to_string(index=False))

    # 방법 C: Person-Hour
    print("\n[방법 C] Person-Hour")
    person_hour = analyze_person_hour(df_processed, date_info['total_days'])
    print(person_hour.head(10).to_string(index=False))

    # 검증
    validate_results(avg_snapshot, person_hour)

    print("\n테스트 완료!")
