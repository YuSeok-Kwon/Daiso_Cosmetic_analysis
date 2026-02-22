"""
S-DoT 유동인구 시기별 분석 스크립트
- 년도별 (2024, 2025)
- 계절별 (봄/여름/가을/겨울)
- 월별 (2024.01 ~ 2025.09)

각 기간마다 5가지 분석:
1. 자치구별 유동인구 TOP 10
2. 지역유형별 유동인구
3. 시간대별 유동인구
4. 핵심 관광지 동별 유동인구
5. 복합점수 (외국인 + 유동인구)

집계 공식:
- S-DoT 일평균 방문자 = 해당 기간 방문자수 합계 / 해당 기간 일수
- 외국인 일평균 = Σ(10~22시 외국인) / 13시간 / 해당 기간 일수 (방법 B)
- 복합점수 = MinMax(외국인_일평균) + MinMax(S-DoT_일평균) (범위 0~2)
- Hub/Spoke 기준 = 복합점수 상위 30%
"""

import pandas as pd
import numpy as np
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 경로 설정
# ============================================================
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_SDOT = os.path.join(BASE, '01_raw_data', 'S-DoT_WALK')
RAW_FOREIGNER = os.path.join(BASE, '01_raw_data')
PROCESSED = os.path.join(BASE, '02_processed_data')
OUTPUT_DIR = os.path.join(PROCESSED, 'temporal')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 분석 기간
DATE_START = '2024-01-01'
DATE_END = '2025-09-30'

# 다이소 영업시간
DAISO_HOURS = list(range(10, 23))  # 10~22시
NUM_HOURS = len(DAISO_HOURS)  # 13시간

# ============================================================
# 2. 매핑 테이블
# ============================================================
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

GU_CODE_MAP = {
    '11110': '종로구', '11140': '중구', '11170': '용산구', '11200': '성동구',
    '11215': '광진구', '11230': '동대문구', '11260': '중랑구', '11290': '성북구',
    '11305': '강북구', '11320': '도봉구', '11350': '노원구', '11380': '은평구',
    '11410': '서대문구', '11440': '마포구', '11470': '양천구', '11500': '강서구',
    '11530': '구로구', '11545': '금천구', '11560': '영등포구', '11590': '동작구',
    '11620': '관악구', '11650': '서초구', '11680': '강남구', '11710': '송파구',
    '11740': '강동구'
}

TYPE_MAP = {
    'main_street': '주요 거리', 'traditional_markets': '전통시장',
    'parks': '공원', 'commercial_area': '상업지역',
    'residential_area': '주거지역', 'public_facilities': '공공시설'
}

# 핵심 관광지 동
TOURIST_DONGS = {
    'Myeong-dong': '명동', 'Gwanghui-dong': '광희동(DDP)',
    'Hoehyeon-dong': '회현동(남대문)', 'Sinsa-dong': '신사동',
    'Gahoe-dong': '가회동(북촌)', 'Apgujeong-dong': '압구정동',
    'Samcheong-dong': '삼청동', 'Itaewon2-dong': '이태원2동'
}

# 계절 정의
SEASON_MAP = {
    1: '겨울', 2: '겨울', 3: '봄', 4: '봄', 5: '봄',
    6: '여름', 7: '여름', 8: '여름', 9: '가을', 10: '가을',
    11: '가을', 12: '겨울'
}
SEASON_ORDER = ['봄', '여름', '가을', '겨울']


# ============================================================
# 3. 데이터 로드
# ============================================================
def load_sdot_data():
    """S-DoT 유동인구 데이터 로드 (101개 주간 CSV)"""
    files = sorted(glob.glob(os.path.join(RAW_SDOT, 'S-DoT_WALK_*.csv')))
    print(f'[S-DoT] 로드할 파일 수: {len(files)}')

    dfs = []
    for f in files:
        for enc in ['utf-8', 'cp949', 'euc-kr']:
            try:
                df = pd.read_csv(f, encoding=enc)
                dfs.append(df)
                break
            except:
                continue

    df_all = pd.concat(dfs, ignore_index=True)
    print(f'[S-DoT] 총 레코드 수: {len(df_all):,}')
    return df_all


def load_foreigner_data():
    """외국인 생활인구 데이터 로드 (2024.01~2025.09)"""
    # 분석 대상 월 폴더 (2024.01 ~ 2025.09)
    target_months = []
    for y in [2024, 2025]:
        end_m = 12 if y == 2024 else 9
        for m in range(1, end_m + 1):
            target_months.append(f'TEMP_FOREIGNER_{y}{m:02d}')

    all_files = []
    for folder in target_months:
        folder_path = os.path.join(RAW_FOREIGNER, folder)
        if os.path.exists(folder_path):
            files = glob.glob(os.path.join(folder_path, 'TEMP_FOREIGNER_*.csv'))
            all_files.extend(files)

    print(f'[외국인] 로드할 파일 수: {len(all_files)}')

    dfs = []
    for f in sorted(all_files):
        try:
            df = pd.read_csv(f, encoding='cp949')
            df.columns = [col.replace('\ufeff', '').replace('?', '').strip('"') for col in df.columns]
            dfs.append(df)
        except:
            pass

    df_all = pd.concat(dfs, ignore_index=True)
    print(f'[외국인] 총 레코드 수: {len(df_all):,}')
    return df_all


# ============================================================
# 4. 데이터 전처리
# ============================================================
def preprocess_sdot(df):
    """S-DoT 데이터 전처리"""
    df = df.copy()
    df['자치구_한글'] = df['자치구'].map(SDOT_GU_MAP)
    df['방문자수'] = pd.to_numeric(df['방문자수'], errors='coerce').fillna(0)
    df['측정일'] = df['측정시간'].str[:10]
    df['시간'] = df['측정시간'].str[11:13].astype(int)

    # 영업시간 필터 (10~22시)
    df = df[df['시간'].between(10, 22)].copy()

    # 기간 필터
    df['측정일_dt'] = pd.to_datetime(df['측정일'], errors='coerce')
    df = df[(df['측정일_dt'] >= DATE_START) & (df['측정일_dt'] <= DATE_END)].copy()

    # 매핑 안 되는 자치구 제거
    df = df[df['자치구_한글'].notna()].copy()

    # 시간 정보 추가
    df['년도'] = df['측정일_dt'].dt.year
    df['월'] = df['측정일_dt'].dt.month
    df['년월'] = df['측정일_dt'].dt.strftime('%Y-%m')
    df['계절'] = df['월'].map(SEASON_MAP)

    print(f'[S-DoT 전처리] 최종 레코드: {len(df):,}, 일수: {df["측정일"].nunique()}일')
    return df


def preprocess_foreigner(df):
    """외국인 생활인구 데이터 전처리"""
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

    # 시간대 필터 (10~22시)
    df['시간대'] = pd.to_numeric(df['시간대'], errors='coerce')
    df = df[df['시간대'].isin(DAISO_HOURS)].copy()

    # 수치 변환
    for col in ['중국인체류인구수', '중국외외국인체류인구수']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace('*', np.nan), errors='coerce').fillna(0)
    df['총생활인구수'] = pd.to_numeric(df['총생활인구수'], errors='coerce').fillna(0)
    df['외국인체류인구수'] = df['중국인체류인구수'] + df['중국외외국인체류인구수']

    # 구 코드 매핑
    df['행정동코드'] = df['행정동코드'].astype(str)
    df['구코드'] = df['행정동코드'].str[:5]
    df['자치구'] = df['구코드'].map(GU_CODE_MAP)

    # 날짜 정보
    df['기준일'] = df['기준일'].astype(str)
    df['날짜'] = pd.to_datetime(df['기준일'], format='%Y%m%d', errors='coerce')

    # 기간 필터
    df = df[(df['날짜'] >= DATE_START) & (df['날짜'] <= DATE_END)].copy()
    df = df[df['자치구'].notna()].copy()

    # 시간 정보
    df['년도'] = df['날짜'].dt.year
    df['월'] = df['날짜'].dt.month
    df['년월'] = df['날짜'].dt.strftime('%Y-%m')
    df['계절'] = df['월'].map(SEASON_MAP)

    print(f'[외국인 전처리] 최종 레코드: {len(df):,}, 일수: {df["기준일"].nunique()}일')
    return df


# ============================================================
# 5. 분석 함수
# ============================================================
def analyze_gu(sdot_df, days):
    """분석 1: 자치구별 유동인구"""
    agg = sdot_df.groupby('자치구_한글').agg(
        방문자수_합=('방문자수', 'sum'),
        센서수=('시리얼', 'nunique')
    ).reset_index()
    agg.rename(columns={'자치구_한글': '자치구'}, inplace=True)
    agg['일평균_방문자'] = (agg['방문자수_합'] / days).round(0)
    agg['센서당_일평균'] = (agg['일평균_방문자'] / agg['센서수']).round(0)
    return agg.sort_values('일평균_방문자', ascending=False)


def analyze_type(sdot_df, days):
    """분석 2: 지역유형별 유동인구"""
    agg = sdot_df.groupby('지역').agg(
        방문자수_합=('방문자수', 'sum'),
        센서수=('시리얼', 'nunique')
    ).reset_index()
    agg['유형'] = agg['지역'].map(TYPE_MAP)
    agg['일평균_방문자'] = (agg['방문자수_합'] / days).round(0)
    agg['센서당_일평균'] = (agg['일평균_방문자'] / agg['센서수']).round(0)
    return agg.sort_values('일평균_방문자', ascending=False)


def analyze_hourly(sdot_df, days):
    """분석 3: 시간대별 유동인구"""
    agg = sdot_df.groupby('시간').agg(
        방문자수_합=('방문자수', 'sum')
    ).reset_index()
    agg['일평균_방문자'] = (agg['방문자수_합'] / days).round(0)
    peak = agg.loc[agg['일평균_방문자'].idxmax()]
    return agg, int(peak['시간']), peak['일평균_방문자']


def analyze_tourist_dong(sdot_df, days):
    """분석 4: 핵심 관광지 동별 유동인구"""
    df_tour = sdot_df[sdot_df['행정동'].isin(TOURIST_DONGS.keys())].copy()
    if len(df_tour) == 0:
        return pd.DataFrame()

    agg = df_tour.groupby(['자치구_한글', '행정동']).agg(
        방문자수_합=('방문자수', 'sum'),
        센서수=('시리얼', 'nunique')
    ).reset_index()
    agg.rename(columns={'자치구_한글': '자치구'}, inplace=True)
    agg['동_한글'] = agg['행정동'].map(TOURIST_DONGS)
    agg['일평균_방문자'] = (agg['방문자수_합'] / days).round(0)
    agg['센서당_일평균'] = (agg['일평균_방문자'] / agg['센서수']).round(0)

    # 시간대별 피크
    hourly = df_tour.groupby(['행정동', '시간'])['방문자수'].sum().reset_index()
    hourly['일평균'] = (hourly['방문자수'] / days).round(0)
    peak_hours = hourly.loc[hourly.groupby('행정동')['일평균'].idxmax()][['행정동', '시간']].rename(columns={'시간': '피크시간'})
    agg = agg.merge(peak_hours, on='행정동', how='left')

    return agg.sort_values('일평균_방문자', ascending=False)


def analyze_composite(sdot_df, foreigner_df, sdot_days, foreigner_days):
    """분석 5: 복합점수 (외국인 + 유동인구)"""
    # S-DoT 자치구별 일평균
    sdot_gu = sdot_df.groupby('자치구_한글')['방문자수'].sum().reset_index()
    sdot_gu.rename(columns={'자치구_한글': '자치구', '방문자수': 'sdot_합'}, inplace=True)
    sdot_gu['S-DoT_일평균'] = (sdot_gu['sdot_합'] / sdot_days).round(0)

    # 외국인 자치구별 일평균 (방법 B: Σ / 13시간 / 일수)
    fg_gu = foreigner_df.groupby('자치구')['외국인체류인구수'].sum().reset_index()
    fg_gu['외국인_일평균'] = (fg_gu['외국인체류인구수'] / NUM_HOURS / foreigner_days).round(0)

    # 병합
    merged = pd.merge(
        fg_gu[['자치구', '외국인_일평균']],
        sdot_gu[['자치구', 'S-DoT_일평균']],
        on='자치구', how='inner'
    )

    if len(merged) < 2:
        return merged

    # MinMax 정규화
    for col, new_col in [('외국인_일평균', '외국인_정규화'), ('S-DoT_일평균', '유동량_정규화')]:
        min_v = merged[col].min()
        max_v = merged[col].max()
        merged[new_col] = ((merged[col] - min_v) / (max_v - min_v + 1e-10)).round(4)

    merged['복합점수'] = (merged['외국인_정규화'] + merged['유동량_정규화']).round(4)

    # Hub/Spoke
    threshold = merged['복합점수'].quantile(0.70)
    merged['분류'] = merged['복합점수'].apply(lambda x: 'Hub' if x >= threshold else 'Spoke')
    merged = merged.sort_values('복합점수', ascending=False)

    return merged


# ============================================================
# 6. 시기별 분석 실행
# ============================================================
def run_period_analysis(sdot_df, foreigner_df, period_name, period_label):
    """한 기간에 대해 5가지 분석 수행"""
    sdot_days = sdot_df['측정일'].nunique()
    fg_days = foreigner_df['기준일'].nunique()

    if sdot_days == 0 or fg_days == 0:
        print(f'  [SKIP] {period_label}: S-DoT {sdot_days}일, 외국인 {fg_days}일')
        return None

    print(f'\n  [{period_label}] S-DoT {sdot_days}일 ({len(sdot_df):,}건), 외국인 {fg_days}일 ({len(foreigner_df):,}건)')

    results = {'기간': period_label, 'S-DoT_일수': sdot_days, '외국인_일수': fg_days}

    # 분석 1: 자치구별
    r1 = analyze_gu(sdot_df, sdot_days)
    r1.to_csv(os.path.join(OUTPUT_DIR, f'{period_name}_1_자치구별.csv'), index=False, encoding='utf-8-sig')
    results['자치구별'] = r1

    # 분석 2: 지역유형별
    r2 = analyze_type(sdot_df, sdot_days)
    r2.to_csv(os.path.join(OUTPUT_DIR, f'{period_name}_2_지역유형별.csv'), index=False, encoding='utf-8-sig')
    results['지역유형별'] = r2

    # 분석 3: 시간대별
    r3, peak_h, peak_v = analyze_hourly(sdot_df, sdot_days)
    r3.to_csv(os.path.join(OUTPUT_DIR, f'{period_name}_3_시간대별.csv'), index=False, encoding='utf-8-sig')
    results['시간대별'] = r3
    results['피크시간'] = peak_h
    results['피크방문자'] = peak_v

    # 분석 4: 관광지 동별
    r4 = analyze_tourist_dong(sdot_df, sdot_days)
    if len(r4) > 0:
        r4.to_csv(os.path.join(OUTPUT_DIR, f'{period_name}_4_관광지동별.csv'), index=False, encoding='utf-8-sig')
    results['관광지동별'] = r4

    # 분석 5: 복합점수
    r5 = analyze_composite(sdot_df, foreigner_df, sdot_days, fg_days)
    r5.to_csv(os.path.join(OUTPUT_DIR, f'{period_name}_5_복합점수.csv'), index=False, encoding='utf-8-sig')
    results['복합점수'] = r5

    return results


def main():
    print('=' * 70)
    print('S-DoT 유동인구 시기별 분석 시작')
    print('=' * 70)

    # 데이터 로드
    print('\n[Step 1] 데이터 로드')
    sdot_raw = load_sdot_data()
    fg_raw = load_foreigner_data()

    # 전처리
    print('\n[Step 2] 데이터 전처리')
    sdot = preprocess_sdot(sdot_raw)
    fg = preprocess_foreigner(fg_raw)

    all_results = {}

    # ---- 년도별 분석 ----
    print('\n' + '=' * 70)
    print('[분석 A] 년도별 분석')
    print('=' * 70)
    for year in [2024, 2025]:
        s = sdot[sdot['년도'] == year]
        f = fg[fg['년도'] == year]
        r = run_period_analysis(s, f, f'년도_{year}', f'{year}년')
        if r:
            all_results[f'{year}년'] = r

    # ---- 동일 기간 비교 (1~9월) ----
    # 2024는 1~12월, 2025는 1~9월이므로 공정한 비교를 위해 동일 기간(1~9월)만 추출
    print('\n' + '=' * 70)
    print('[분석 A-2] 동일 기간 비교 (1~9월)')
    print('=' * 70)
    for year in [2024, 2025]:
        s = sdot[(sdot['년도'] == year) & (sdot['월'] <= 9)]
        f = fg[(fg['년도'] == year) & (fg['월'] <= 9)]
        r = run_period_analysis(s, f, f'동일기간_{year}_1-9월', f'{year}년 1~9월')
        if r:
            all_results[f'{year}년_1-9월'] = r

    # ---- 계절별 분석 ----
    print('\n' + '=' * 70)
    print('[분석 B] 계절별 분석')
    print('=' * 70)
    for season in SEASON_ORDER:
        s = sdot[sdot['계절'] == season]
        f = fg[fg['계절'] == season]
        r = run_period_analysis(s, f, f'계절_{season}', f'{season}')
        if r:
            all_results[season] = r

    # ---- 월별 분석 ----
    print('\n' + '=' * 70)
    print('[분석 C] 월별 분석')
    print('=' * 70)
    months = sorted(sdot['년월'].unique())
    for ym in months:
        s = sdot[sdot['년월'] == ym]
        f = fg[fg['년월'] == ym]
        r = run_period_analysis(s, f, f'월별_{ym}', ym)
        if r:
            all_results[ym] = r

    # ---- 요약 테이블 저장 ----
    print('\n' + '=' * 70)
    print('[결과 요약]')
    print('=' * 70)

    summary_rows = []
    for label, r in all_results.items():
        gu_top3 = r['자치구별'].head(3)['자치구'].tolist()
        hub_list = r['복합점수'][r['복합점수']['분류'] == 'Hub']['자치구'].tolist()

        summary_rows.append({
            '기간': label,
            'S-DoT_일수': r['S-DoT_일수'],
            '외국인_일수': r['외국인_일수'],
            '유동인구_TOP1': gu_top3[0] if len(gu_top3) > 0 else '',
            '유동인구_TOP2': gu_top3[1] if len(gu_top3) > 1 else '',
            '유동인구_TOP3': gu_top3[2] if len(gu_top3) > 2 else '',
            '피크시간': f'{r["피크시간"]}시',
            '피크_일평균': r['피크방문자'],
            'Hub_자치구': ', '.join(hub_list[:3]),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(OUTPUT_DIR, '00_분석요약.csv'), index=False, encoding='utf-8-sig')

    print(df_summary.to_string(index=False))

    # ---- 교차검증: 년도별 합산 ≈ 전체기간 ----
    print('\n' + '=' * 70)
    print('[교차검증] 년도별 일수 합산')
    print('=' * 70)
    total_sdot_days = sdot['측정일'].nunique()
    total_fg_days = fg['기준일'].nunique()
    y2024_sdot = sdot[sdot['년도'] == 2024]['측정일'].nunique()
    y2025_sdot = sdot[sdot['년도'] == 2025]['측정일'].nunique()
    y2024_fg = fg[fg['년도'] == 2024]['기준일'].nunique()
    y2025_fg = fg[fg['년도'] == 2025]['기준일'].nunique()

    print(f'S-DoT: 2024({y2024_sdot}일) + 2025({y2025_sdot}일) = {y2024_sdot + y2025_sdot}일 vs 전체 {total_sdot_days}일')
    print(f'외국인: 2024({y2024_fg}일) + 2025({y2025_fg}일) = {y2024_fg + y2025_fg}일 vs 전체 {total_fg_days}일')

    # 방문자수 합산 검증
    total_visitors = sdot['방문자수'].sum()
    y2024_visitors = sdot[sdot['년도'] == 2024]['방문자수'].sum()
    y2025_visitors = sdot[sdot['년도'] == 2025]['방문자수'].sum()
    print(f'\nS-DoT 방문자 합산: 2024({y2024_visitors:,.0f}) + 2025({y2025_visitors:,.0f}) = {y2024_visitors + y2025_visitors:,.0f} vs 전체 {total_visitors:,.0f}')

    print(f'\n저장 경로: {OUTPUT_DIR}')
    print(f'생성 파일 수: {len(os.listdir(OUTPUT_DIR))}개')
    print('\n분석 완료!')


if __name__ == '__main__':
    main()
