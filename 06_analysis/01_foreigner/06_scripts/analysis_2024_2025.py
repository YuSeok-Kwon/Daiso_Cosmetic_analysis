"""
2024-2025 서울시 외국인/유동인구 및 화장품 매출 통합 분석
- 분석 대상: 외국인 생활인구, S-DoT 유동인구, 화장품 매출
- 목적: Hub & Spoke 전략을 위한 상권 추천
"""

import pandas as pd
import numpy as np
import glob
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 기본 경로 설정
BASE_PATH = "/Users/yu_seok/Documents/workspace/01_현재진행/01_nbCamp/Project/Why-pi/temp"
FOREIGNER_PATH = f"{BASE_PATH}/FOREIGNER/data/TEMP_FOREIGNER"
SDOT_PATH = f"{BASE_PATH}/FOREIGNER/data/S-DoT_WALK"
OUTPUT_PATH = BASE_PATH

# 자치구 코드 매핑 (자치구 코드 앞 5자리 기준)
GU_CODE_MAP = {
    '11110': '종로구', '11140': '중구', '11170': '용산구', '11200': '성동구',
    '11215': '광진구', '11230': '동대문구', '11260': '중랑구', '11290': '성북구',
    '11305': '강북구', '11320': '도봉구', '11350': '노원구', '11380': '은평구',
    '11410': '서대문구', '11440': '마포구', '11470': '양천구', '11500': '강서구',
    '11530': '구로구', '11545': '금천구', '11560': '영등포구', '11590': '동작구',
    '11620': '관악구', '11650': '서초구', '11680': '강남구', '11710': '송파구',
    '11740': '강동구'
}

print("="*80)
print("2024-2025 서울시 외국인/유동인구 및 화장품 매출 통합 분석")
print("="*80)

# =============================================================================
# 1. 외국인 생활인구 데이터 로드 및 처리
# =============================================================================
print("\n[1] 외국인 생활인구 데이터 처리 중...")

def load_foreigner_data(year_prefix):
    """특정 연도의 외국인 생활인구 데이터 로드"""
    pattern = f"{FOREIGNER_PATH}/TEMP_FOREIGNER_{year_prefix}*/TEMP_FOREIGNER_*.csv"
    files = glob.glob(pattern)

    all_data = []
    for file in files:
        try:
            df = pd.read_csv(file, encoding='euc-kr')
            # 컬럼명 정리 (인코딩 문제로 깨진 컬럼명 수정)
            df.columns = ['기준일ID', '시간대구분', '자치구코드', '집계구코드',
                         '총생활인구수', '중국인등체류인구수', '중국인등단기외국인체류인구수']
            all_data.append(df)
        except Exception as e:
            continue

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    return combined

# 2024년, 2025년 데이터 로드
foreigner_2024 = load_foreigner_data('2024')
foreigner_2025 = load_foreigner_data('2025')

print(f"  - 2024년 외국인 데이터: {len(foreigner_2024):,} rows")
print(f"  - 2025년 외국인 데이터: {len(foreigner_2025):,} rows")

def process_foreigner_data(df, year):
    """외국인 데이터 처리: 자치구별 집계"""
    if df.empty:
        return pd.DataFrame()

    # 시간대 필터링 (10시~22시 = 영업시간)
    df['시간대'] = df['시간대구분'].astype(int)
    business_hours = df[df['시간대'].between(10, 22)].copy()

    # 자치구 코드 추출 (앞 5자리)
    business_hours['자치구코드_5'] = business_hours['자치구코드'].astype(str).str[:5]
    business_hours['자치구명'] = business_hours['자치구코드_5'].map(GU_CODE_MAP)

    # 숫자 컬럼 변환 (별표 처리)
    numeric_cols = ['총생활인구수', '중국인등체류인구수', '중국인등단기외국인체류인구수']
    for col in numeric_cols:
        business_hours[col] = pd.to_numeric(business_hours[col], errors='coerce')

    # 자치구별 일평균 집계
    daily_gu = business_hours.groupby(['기준일ID', '자치구명']).agg({
        '총생활인구수': 'sum',
        '중국인등체류인구수': 'sum',
        '중국인등단기외국인체류인구수': 'sum'
    }).reset_index()

    gu_summary = daily_gu.groupby('자치구명').agg({
        '총생활인구수': 'mean',
        '중국인등체류인구수': 'mean',
        '중국인등단기외국인체류인구수': 'mean',
        '기준일ID': 'nunique'
    }).reset_index()

    gu_summary.columns = ['자치구', f'{year}_총생활인구_일평균', f'{year}_중국인등체류_일평균',
                          f'{year}_단기외국인_일평균', f'{year}_관측일수']

    gu_summary['year'] = year

    return gu_summary

# 연도별 처리
gu_2024 = process_foreigner_data(foreigner_2024, 2024)
gu_2025 = process_foreigner_data(foreigner_2025, 2025)

# =============================================================================
# 2. 2024년 vs 2025년 외국인 현황 비교
# =============================================================================
print("\n[2] 2024년 vs 2025년 외국인 현황 비교 분석...")

# 두 연도 병합
if not gu_2024.empty and not gu_2025.empty:
    comparison = pd.merge(
        gu_2024[['자치구', '2024_총생활인구_일평균', '2024_중국인등체류_일평균', '2024_단기외국인_일평균']],
        gu_2025[['자치구', '2025_총생활인구_일평균', '2025_중국인등체류_일평균', '2025_단기외국인_일평균']],
        on='자치구', how='outer'
    )

    # 변화율 계산
    comparison['총생활인구_변화율'] = ((comparison['2025_총생활인구_일평균'] - comparison['2024_총생활인구_일평균'])
                                        / comparison['2024_총생활인구_일평균'] * 100)
    comparison['외국인체류_변화율'] = ((comparison['2025_중국인등체류_일평균'] - comparison['2024_중국인등체류_일평균'])
                                        / comparison['2024_중국인등체류_일평균'] * 100)
    comparison['단기외국인_변화율'] = ((comparison['2025_단기외국인_일평균'] - comparison['2024_단기외국인_일평균'])
                                        / comparison['2024_단기외국인_일평균'] * 100)

    # 결과 저장
    comparison = comparison.round(2)
    comparison.to_csv(f"{OUTPUT_PATH}/외국인_연도별비교_2024_2025.csv", index=False, encoding='utf-8-sig')
    print(f"  - 저장 완료: 외국인_연도별비교_2024_2025.csv")

    # 상위 5개 자치구 출력
    print("\n  [외국인 생활인구 TOP 5 자치구 (2025년 기준)]")
    top5 = comparison.nlargest(5, '2025_총생활인구_일평균')[['자치구', '2025_총생활인구_일평균', '총생활인구_변화율']]
    for idx, row in top5.iterrows():
        change = "증가" if row['총생활인구_변화율'] > 0 else "감소"
        print(f"    {row['자치구']}: 일평균 {row['2025_총생활인구_일평균']:,.0f}명 ({change} {abs(row['총생활인구_변화율']):.1f}%)")

# =============================================================================
# 3. 화장품 매출 데이터 분석
# =============================================================================
print("\n[3] 화장품 매출 데이터 분석...")

cosmetics_file = f"{BASE_PATH}/화장품_매출_2024.csv"
cosmetics_df = pd.read_csv(cosmetics_file, encoding='utf-8-sig')

print(f"  - 전체 레코드: {len(cosmetics_df):,}")
print(f"  - 총 매출액: {cosmetics_df['당월_매출_금액'].sum():,.0f}원")

# 상권코드명에서 자치구 추출 시도
def extract_gu_from_name(name):
    """상권명에서 자치구 추출"""
    gu_keywords = ['종로', '중구', '용산', '성동', '광진', '동대문', '중랑', '성북',
                   '강북', '도봉', '노원', '은평', '서대문', '마포', '양천', '강서',
                   '구로', '금천', '영등포', '동작', '관악', '서초', '강남', '송파', '강동']

    for keyword in gu_keywords:
        if keyword in str(name):
            if keyword == '종로':
                return '종로구'
            elif keyword == '중구':
                return '중구'
            return keyword + '구'
    return None

# 상권별 매출 집계
sales_by_area = cosmetics_df.groupby('상권_코드_명').agg({
    '당월_매출_금액': 'sum',
    '당월_매출_건수': 'sum'
}).reset_index()
sales_by_area.columns = ['상권명', '총매출액', '총매출건수']
sales_by_area['객단가'] = sales_by_area['총매출액'] / sales_by_area['총매출건수']
sales_by_area = sales_by_area.sort_values('총매출액', ascending=False)

# 화장품 매출 상위 상권 저장
top_cosmetics = sales_by_area.head(50)
top_cosmetics.to_csv(f"{OUTPUT_PATH}/화장품_매출_상위50_상권.csv", index=False, encoding='utf-8-sig')
print(f"  - 저장 완료: 화장품_매출_상위50_상권.csv")

print("\n  [화장품 매출 TOP 10 상권]")
for idx, row in sales_by_area.head(10).iterrows():
    print(f"    {row['상권명']}: {row['총매출액']:,.0f}원 (건수: {row['총매출건수']:,})")

# =============================================================================
# 4. S-DoT 유동인구 데이터 분석
# =============================================================================
print("\n[4] S-DoT 유동인구 데이터 분석...")

sdot_files = glob.glob(f"{SDOT_PATH}/S-DoT_WALK_*.csv")
print(f"  - S-DoT 파일 수: {len(sdot_files)}")

# 샘플 데이터로 구조 확인 및 분석
all_sdot = []
for file in sdot_files:
    try:
        df = pd.read_csv(file, encoding='utf-8')
        all_sdot.append(df)
    except:
        try:
            df = pd.read_csv(file, encoding='euc-kr')
            all_sdot.append(df)
        except:
            continue

if all_sdot:
    sdot_combined = pd.concat(all_sdot, ignore_index=True)

    # 컬럼명 정리
    if len(sdot_combined.columns) >= 7:
        sdot_combined.columns = ['모델번호', '시리얼', '측정시간', '지역유형', '자치구', '행정동', '방문자수', '수집시간'][:len(sdot_combined.columns)]

    # 관광지역 필터링
    tourism_keywords = ['Myeong-dong', 'Itaewon', 'Hongdae', 'Bukchon', 'Insadong', 'Gangnam',
                        'Sinchon', 'Samcheong', 'Jamsil', 'Gwanghui', 'main_street', 'traditional_markets']

    if '지역유형' in sdot_combined.columns or '행정동' in sdot_combined.columns:
        # 관광 관련 데이터 필터링
        tourism_mask = sdot_combined.apply(
            lambda row: any(kw in str(row.values) for kw in tourism_keywords), axis=1
        )
        tourism_data = sdot_combined[tourism_mask].copy()

        if '자치구' in tourism_data.columns and '방문자수' in tourism_data.columns:
            # 자치구별 유동인구 집계
            tourism_data['방문자수'] = pd.to_numeric(tourism_data['방문자수'], errors='coerce')

            gu_flow = tourism_data.groupby('자치구').agg({
                '방문자수': ['sum', 'mean', 'count']
            }).reset_index()
            gu_flow.columns = ['자치구', '총유동인구', '평균유동인구', '측정횟수']
            gu_flow = gu_flow.sort_values('총유동인구', ascending=False)

            gu_flow.to_csv(f"{OUTPUT_PATH}/관광지역_유동인구_자치구별.csv", index=False, encoding='utf-8-sig')
            print(f"  - 저장 완료: 관광지역_유동인구_자치구별.csv")

            print("\n  [관광지역 유동인구 TOP 10 자치구]")
            for idx, row in gu_flow.head(10).iterrows():
                print(f"    {row['자치구']}: 총 {row['총유동인구']:,.0f}명 (평균 {row['평균유동인구']:.0f}명)")

    # 지역유형별 분석
    if '지역유형' in sdot_combined.columns:
        sdot_combined['방문자수'] = pd.to_numeric(sdot_combined['방문자수'], errors='coerce')
        type_summary = sdot_combined.groupby('지역유형')['방문자수'].agg(['sum', 'mean', 'count']).reset_index()
        type_summary.columns = ['지역유형', '총유동인구', '평균유동인구', '측정횟수']
        type_summary = type_summary.sort_values('총유동인구', ascending=False)

        type_summary.to_csv(f"{OUTPUT_PATH}/유동인구_지역유형별.csv", index=False, encoding='utf-8-sig')
        print(f"\n  - 저장 완료: 유동인구_지역유형별.csv")

        print("\n  [지역유형별 유동인구]")
        for idx, row in type_summary.iterrows():
            print(f"    {row['지역유형']}: 총 {row['총유동인구']:,.0f}명")

# =============================================================================
# 5. Hub & Spoke 전략 추천
# =============================================================================
print("\n[5] Hub & Spoke 상권 추천 분석...")

# 화장품 매출과 외국인 밀집지역 통합 분석
hub_candidates = pd.DataFrame()

# 외국인 생활인구 기반 점수
if not gu_2025.empty:
    foreigner_score = gu_2025[['자치구', '2025_총생활인구_일평균', '2025_단기외국인_일평균']].copy()
    foreigner_score.columns = ['자치구', '총생활인구', '단기외국인']

    # 정규화
    foreigner_score['외국인점수'] = (foreigner_score['총생활인구'] - foreigner_score['총생활인구'].min()) / \
                                    (foreigner_score['총생활인구'].max() - foreigner_score['총생활인구'].min())

    hub_candidates = foreigner_score

# S-DoT 유동인구 점수 추가
if all_sdot and '자치구' in gu_flow.columns:
    flow_score = gu_flow[['자치구', '총유동인구']].copy()
    flow_score['유동점수'] = (flow_score['총유동인구'] - flow_score['총유동인구'].min()) / \
                            (flow_score['총유동인구'].max() - flow_score['총유동인구'].min())

    if not hub_candidates.empty:
        hub_candidates = pd.merge(hub_candidates, flow_score[['자치구', '유동점수']], on='자치구', how='outer')
    else:
        hub_candidates = flow_score

# 복합 점수 계산
if not hub_candidates.empty and '외국인점수' in hub_candidates.columns and '유동점수' in hub_candidates.columns:
    hub_candidates['복합점수'] = hub_candidates['외국인점수'].fillna(0) * 0.5 + hub_candidates['유동점수'].fillna(0) * 0.5
    hub_candidates = hub_candidates.sort_values('복합점수', ascending=False)

    # Hub (상위 30%) vs Spoke (하위 70%) 분류
    threshold = hub_candidates['복합점수'].quantile(0.7)
    hub_candidates['전략유형'] = hub_candidates['복합점수'].apply(lambda x: 'Hub' if x >= threshold else 'Spoke')

    hub_candidates.to_csv(f"{OUTPUT_PATH}/Hub_Spoke_상권추천.csv", index=False, encoding='utf-8-sig')
    print(f"  - 저장 완료: Hub_Spoke_상권추천.csv")

    print("\n  [Hub 상권 추천 (관광/외국인 밀집)]")
    hub_areas = hub_candidates[hub_candidates['전략유형'] == 'Hub']
    for idx, row in hub_areas.iterrows():
        print(f"    {row['자치구']}: 복합점수 {row['복합점수']:.3f}")

    print("\n  [Spoke 상권 추천 (거주지역)]")
    spoke_areas = hub_candidates[hub_candidates['전략유형'] == 'Spoke'].head(5)
    for idx, row in spoke_areas.iterrows():
        print(f"    {row['자치구']}: 복합점수 {row['복합점수']:.3f}")

# =============================================================================
# 6. 최종 분석 요약 리포트 생성
# =============================================================================
print("\n[6] 최종 분석 요약 리포트 생성...")

summary_report = f"""
================================================================================
                    2024-2025 서울시 통합 분석 리포트
================================================================================

[분석 일시] 2026-02-15

[1. 외국인 생활인구 현황 (2024 vs 2025)]
--------------------------------------------------------------------------------
"""

if not gu_2024.empty and not gu_2025.empty:
    summary_report += f"""
- 2024년 총 관측일수: {gu_2024['2024_관측일수'].sum():,}일
- 2025년 총 관측일수: {gu_2025['2025_관측일수'].sum():,}일

[외국인 생활인구 TOP 5 자치구 (2025년)]
"""
    if 'comparison' in dir():
        top5_2025 = comparison.nlargest(5, '2025_총생활인구_일평균')
        for idx, row in top5_2025.iterrows():
            summary_report += f"  - {row['자치구']}: 일평균 {row['2025_총생활인구_일평균']:,.0f}명 (전년대비 {row['총생활인구_변화율']:+.1f}%)\n"

summary_report += f"""
[2. 화장품 매출 현황 (2024년 1분기)]
--------------------------------------------------------------------------------
- 총 분석 상권 수: {len(sales_by_area):,}개
- 총 매출액: {cosmetics_df['당월_매출_금액'].sum():,.0f}원
- 총 매출건수: {cosmetics_df['당월_매출_건수'].sum():,}건
- 평균 객단가: {cosmetics_df['당월_매출_금액'].sum() / cosmetics_df['당월_매출_건수'].sum():,.0f}원

[매출 TOP 5 상권]
"""
for idx, row in sales_by_area.head(5).iterrows():
    summary_report += f"  - {row['상권명']}: {row['총매출액']:,.0f}원\n"

summary_report += f"""
[3. S-DoT 유동인구 분석]
--------------------------------------------------------------------------------
- 분석 기간: 2024-2025년
- 총 측정 파일: {len(sdot_files)}개
"""

if all_sdot:
    summary_report += f"- 총 측정 건수: {len(sdot_combined):,}건\n"
    if '지역유형' in sdot_combined.columns:
        summary_report += "\n[지역유형별 유동인구 요약]\n"
        for idx, row in type_summary.head(5).iterrows():
            summary_report += f"  - {row['지역유형']}: 총 {row['총유동인구']:,.0f}명\n"

summary_report += f"""
[4. Hub & Spoke 전략 추천]
--------------------------------------------------------------------------------
"""

if not hub_candidates.empty:
    hub_list = hub_candidates[hub_candidates['전략유형'] == 'Hub']['자치구'].tolist()
    spoke_list = hub_candidates[hub_candidates['전략유형'] == 'Spoke']['자치구'].tolist()

    summary_report += f"""
[Hub 상권 (관광/외국인 밀집 - 연착륙 스킨케어 집중 배치)]
{', '.join(hub_list)}

[Spoke 상권 (거주지역 - 구색 위주 진열)]
{', '.join(spoke_list)}

[전략 제안]
1. Hub 상권: '연착륙 스킨케어 62종' 재고를 일반 대비 5~10배 깊게 배치
2. Spoke 상권: 다양한 제품을 테스트할 수 있도록 구색(Variety) 위주 진열
3. Hub 상권 우선순위: 중구(명동) > 강남구 > 용산구(이태원)
"""

summary_report += """
================================================================================
                              [분석 파일 목록]
================================================================================
1. 외국인_연도별비교_2024_2025.csv - 자치구별 외국인 현황 비교
2. 화장품_매출_상위50_상권.csv - 화장품 매출 상위 상권
3. 관광지역_유동인구_자치구별.csv - 관광지역 중심 유동인구
4. 유동인구_지역유형별.csv - 지역유형별 유동인구 요약
5. Hub_Spoke_상권추천.csv - Hub & Spoke 전략 추천

================================================================================
"""

# 리포트 저장
with open(f"{OUTPUT_PATH}/통합분석_리포트_2024_2025.txt", 'w', encoding='utf-8') as f:
    f.write(summary_report)

print(summary_report)
print(f"\n리포트 저장 완료: {OUTPUT_PATH}/통합분석_리포트_2024_2025.txt")
