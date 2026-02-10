"""
모든 aspect 파일의 감성 레이블 불일치 검사
"""

import pandas as pd
from pathlib import Path

aspect_dir = Path("/Users/yu_seok/Documents/workspace/nbCamp/Project/Why-pi/ABSA/data/raw/by_aspect")

# 각 aspect별 부정 키워드
aspect_keywords = {
    "품질": ['불량', '하자', '파손', '고장', '안열', '안나와', '터져', '깨져', '찢어', '박살'],
    "배송_포장": ['늦', '파손', '훼손', '찌그러', '터져', '깨져', '안옴', '분실'],
    "가격_가성비": ['비싸', '비쌈', '아까', '손해', '별로'],
    "사용감_성능": ['안좋', '별로', '안되', '안맞', '불편', '따가', '쓰림', '건조'],
    "디자인": ['별로', '촌스러', '안예쁘', '구리', '싸구려'],
    "재질_냄새": ['냄새', '악취', '비릿', '퀴퀴', '별로'],
    "용량_휴대": ['작', '적', '모자라', '부족'],
    "색상_발색": ['안나', '별로', '안예쁘', '다르', '이상'],
    "CS_응대": ['불친절', '무시', '답변없', '환불안', '교환안'],
    "재구매": ['안삼', '안살', '안함', '비추', '별로']
}


def check_aspect_file(filepath: Path) -> dict:
    """aspect 파일 불일치 검사"""
    df = pd.read_csv(filepath)
    aspect_name = filepath.stem

    # 해당 aspect 키워드 가져오기
    keywords = aspect_keywords.get(aspect_name, [])

    suspicious = []

    for idx, row in df.iterrows():
        rating = row['rating']
        sentiment = row['sentiment']
        text = str(row['text']).lower()
        score = row.get('sentiment_score', 0)

        issues = []

        # 케이스 1: rating 1점 + neutral
        if rating == 1 and sentiment == 'neutral':
            issues.append(f"rating 1 + neutral")

        # 케이스 2: rating 2점 + neutral + 부정 키워드
        if rating == 2 and sentiment == 'neutral':
            if any(kw in text for kw in keywords):
                issues.append(f"rating 2 + neutral + 부정키워드")

        # 케이스 3: 부정 키워드 + neutral (rating 1-3)
        if rating <= 3 and sentiment == 'neutral':
            found_kw = [kw for kw in keywords if kw in text]
            if found_kw:
                issues.append(f"부정키워드({found_kw[0]}) + neutral")

        # 케이스 4: score 불일치
        if score <= -0.5 and sentiment == 'neutral':
            issues.append(f"score {score} + neutral")

        if score >= 0.5 and sentiment == 'neutral':
            issues.append(f"score {score} + neutral")

        # 케이스 5: rating 4-5 + negative
        if rating >= 4 and sentiment == 'negative':
            issues.append(f"rating {rating} + negative")

        if issues:
            suspicious.append({
                'index': row.get('index', idx),
                'text': text[:50],
                'rating': rating,
                'sentiment': sentiment,
                'score': score,
                'issues': issues
            })

    return {
        'name': aspect_name,
        'total': len(df),
        'distribution': df['sentiment'].value_counts().to_dict(),
        'suspicious_count': len(suspicious),
        'suspicious': suspicious[:10]  # 상위 10개만
    }


def main():
    print("="*70)
    print("모든 Aspect 파일 감성 불일치 검사")
    print("="*70)

    results = []

    for filepath in sorted(aspect_dir.glob("*.csv")):
        if filepath.name.startswith("품질_correction"):
            continue
        result = check_aspect_file(filepath)
        results.append(result)

    # 요약 출력
    print("\n[요약]")
    print("-"*70)
    print(f"{'Aspect':<15} {'Total':>6} {'Pos':>5} {'Neu':>5} {'Neg':>5} {'의심':>6}")
    print("-"*70)

    total_suspicious = 0
    for r in results:
        dist = r['distribution']
        pos = dist.get('positive', 0)
        neu = dist.get('neutral', 0)
        neg = dist.get('negative', 0)
        susp = r['suspicious_count']
        total_suspicious += susp

        flag = "⚠️" if susp > 5 else ""
        print(f"{r['name']:<15} {r['total']:>6} {pos:>5} {neu:>5} {neg:>5} {susp:>6} {flag}")

    print("-"*70)
    print(f"{'총 의심 건수':<15} {total_suspicious:>39}")
    print()

    # 의심 건수가 많은 파일 상세 출력
    print("\n[의심 건수 > 5인 파일 상세]")
    for r in results:
        if r['suspicious_count'] > 5:
            print(f"\n{'='*70}")
            print(f"📁 {r['name']}.csv ({r['suspicious_count']}건)")
            print("="*70)
            for s in r['suspicious']:
                print(f"\n  [{s['index']}] rating={s['rating']}, {s['sentiment']} ({s['score']})")
                print(f"       {s['text']}...")
                print(f"       문제: {', '.join(s['issues'])}")


if __name__ == "__main__":
    main()
