#!/usr/bin/env python3
"""
서울시 상권분석서비스(추정매출-상권) API 다운로드 스크립트
2025년 1~3분기 화장품 데이터만 다운로드
"""

import requests
import pandas as pd
import time
import os

API_KEY = "6e674a78616b797537324668506566"
SERVICE_NAME = "VwsmTrdarSelngQq"
BASE_URL = f"http://openapi.seoul.go.kr:8088/{API_KEY}/json/{SERVICE_NAME}"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2025년 1~3분기만
QUARTERS = ["20251", "20252", "20253"]

# 화장품 업종 코드
COSMETICS_CODE = "CS300022"


def download_quarter_data(quarter_code):
    """특정 분기 데이터 다운로드"""

    print(f"\n[{quarter_code}] 다운로드 중...")

    # 1. 전체 건수 확인
    try:
        url = f"{BASE_URL}/1/1/{quarter_code}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        if SERVICE_NAME in data:
            total_count = data[SERVICE_NAME].get('list_total_count', 0)
            print(f"  총 {total_count:,}건")
        elif 'RESULT' in data:
            code = data['RESULT'].get('CODE', '')
            msg = data['RESULT'].get('MESSAGE', '')
            if 'INFO-200' in code:
                print(f"  데이터 없음 (아직 미공개)")
                return None
            print(f"  API 에러: {code} - {msg}")
            return None
        else:
            print(f"  응답 오류: {list(data.keys())}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"  연결 실패: {e}")
        return None

    if total_count == 0:
        return None

    # 2. 배치 다운로드
    all_data = []
    batch_size = 1000

    for start in range(1, total_count + 1, batch_size):
        end = min(start + batch_size - 1, total_count)

        try:
            url = f"{BASE_URL}/{start}/{end}/{quarter_code}"
            res = requests.get(url, timeout=60)
            res.raise_for_status()

            rows = res.json().get(SERVICE_NAME, {}).get('row', [])
            all_data.extend(rows)

            progress = (end / total_count) * 100
            print(f"  {start:,}~{end:,} ({progress:.0f}%)")

            time.sleep(0.3)

        except Exception as e:
            print(f"  {start}~{end} 실패: {e}")
            continue

    return all_data


def main():
    print("=" * 60)
    print("서울시 상권분석서비스(추정매출-상권) 다운로드")
    print("2025년 1~3분기 / 화장품만")
    print("=" * 60)

    all_data = []
    successful = []

    for quarter in QUARTERS:
        data = download_quarter_data(quarter)
        if data:
            all_data.extend(data)
            successful.append(quarter)

    print("\n" + "=" * 60)

    if all_data:
        df = pd.DataFrame(all_data)
        print(f"전체 다운로드: {len(df):,}건")

        # 화장품만 필터링
        col_name = None
        for col in ['SVC_INDUTY_CD', 'SVCINDUTY_CD', 'svc_induty_cd']:
            if col in df.columns:
                col_name = col
                break

        if col_name:
            df_cosmetics = df[df[col_name] == COSMETICS_CODE].copy()
            print(f"화장품 필터링: {len(df_cosmetics):,}건")

            if len(df_cosmetics) > 0:
                output = os.path.join(OUTPUT_DIR, "화장품_매출_2025_1_3분기.csv")
                df_cosmetics.to_csv(output, index=False, encoding='utf-8-sig')
                print(f"\n저장 완료: {output}")
                print(f"성공 분기: {', '.join(successful)}")
                print(f"컬럼: {', '.join(df_cosmetics.columns[:6])}...")
            else:
                print("화장품 데이터가 없습니다.")
        else:
            print(f"업종코드 컬럼을 찾을 수 없음. 컬럼: {list(df.columns)}")
            # 전체 저장
            output = os.path.join(OUTPUT_DIR, "추정매출_2025_1_3분기.csv")
            df.to_csv(output, index=False, encoding='utf-8-sig')
            print(f"전체 저장: {output}")
    else:
        print("다운로드된 데이터가 없습니다.")
        print("2025년 데이터가 아직 공개되지 않았을 수 있습니다.")

    print("=" * 60)


if __name__ == "__main__":
    main()
