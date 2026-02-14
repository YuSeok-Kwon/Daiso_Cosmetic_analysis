"""
Claude가 직접 리뷰를 라벨링하는 로직 (규칙 기반 + 키워드 매칭)
11개 Aspect: 배송/포장, 품질/퀄리티, 가격/가성비, 사용감/성능, 용량/휴대, 디자인, 재질/냄새, CS/응대, 재구매, 색상/발색, 미분류
"""
import pandas as pd
import re
import json
from pathlib import Path

# Aspect 키워드 사전
ASPECT_KEYWORDS = {
    '배송/포장': ['배송', '포장', '도착', '택배', '빠르', '느리', '박스', '파손', '뽁뽁이', '안전하게', '꼼꼼', '늦', '빨리'],
    '품질/퀄리티': ['품질', '퀄리티', '질', '싸구려', '고급', '저렴해보', '허접', '튼튼', '약해', '부실'],
    '가격/가성비': ['가격', '가성비', '싸', '저렴', '비싸', '값', '돈', '원', '할인', '세일', '득템', '이 가격에'],
    '사용감/성능': ['발림', '흡수', '지속', '커버', '효과', '촉촉', '건조', '당김', '따가', '자극', '순해', '끈적', '산뜻', '무거', '가벼', '밀림', '뭉침', '들뜸', '착붙', '밀착', '보습', '수분', '촉촉'],
    '용량/휴대': ['용량', '양', '많', '적', '작', '크', '휴대', '들고다니', '파우치', '여행', 'ml', 'g', '개입'],
    '디자인': ['디자인', '예쁘', '귀엽', '이쁘', '예뻐', '디자인', '모양', '생김', '멋', '세련'],
    '재질/냄새': ['냄새', '향', '향기', '냄시', '쿰쿰', '비릿', '플라스틱', '재질', '질감', '텍스처', '알코올', '무향', '화학'],
    'CS/응대': ['고객', '응대', '문의', '답변', '교환', '환불', '서비스', 'CS', '친절'],
    '재구매': ['재구매', '또 사', '또사', '쟁여', '정착', '계속 쓸', '앞으로', '다음에도', '재주문', '추가 구매', '더 살', '스톡'],
    '색상/발색': ['발색', '색', '컬러', '톤', '밝', '어두', '진하', '연하', '핑크', '레드', '베이지', '브라운', '코랄', '누드', '웜톤', '쿨톤', '화사', '생기']
}

# Sentiment 키워드 사전
POSITIVE_KEYWORDS = ['좋', '최고', '만족', '예쁘', '이쁘', '굿', '짱', '대박', '추천', '완벽', '사랑', '훌륭', '믿고', '꿀', '인생', '갓', 'good', 'best', '👍', '❤', '😍', '♡', '잘 샀', '득템', '착붙', '촉촉', '산뜻', '순해', '가벼', '빠르', '꼼꼼', '깔끔']
NEGATIVE_KEYWORDS = ['별로', '안 좋', '실망', '후회', '최악', '싫', '나쁘', '아쉽', '그냥', '그저', '안맞', '안 맞', '트러블', '뒤집', '따가', '자극', '건조', '당김', '끈적', '무거', '밀림', '뭉침', '들뜸', '늦', '느려', '파손', '빠짐', '깨짐', '없어', '안 와', '연하', '안 남', '없음', '글쎄', '음...', '흠...']
NEUTRAL_KEYWORDS = ['보통', '무난', '그럭저럭', '평범', '그냥저냥', '쓸만', '나쁘지 않', '괜찮']


def get_aspect_from_text(text):
    """텍스트에서 가장 관련 있는 aspect 추출"""
    text_lower = text.lower()
    aspect_scores = {}

    for aspect, keywords in ASPECT_KEYWORDS.items():
        score = 0
        matched_keywords = []
        for keyword in keywords:
            if keyword in text_lower:
                score += 1
                matched_keywords.append(keyword)
        if score > 0:
            aspect_scores[aspect] = (score, matched_keywords)

    if not aspect_scores:
        return [{'aspect': '미분류', 'confidence': 0.5, 'matched': []}]

    # 점수순 정렬
    sorted_aspects = sorted(aspect_scores.items(), key=lambda x: x[1][0], reverse=True)

    results = []
    for aspect, (score, matched) in sorted_aspects[:3]:  # 상위 3개까지
        confidence = min(0.9, 0.5 + score * 0.15)
        results.append({'aspect': aspect, 'confidence': confidence, 'matched': matched})

    return results


def get_sentiment_from_text(text, rating):
    """텍스트와 평점으로 sentiment 판단"""
    text_lower = text.lower()

    positive_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)
    negative_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)
    neutral_count = sum(1 for kw in NEUTRAL_KEYWORDS if kw in text_lower)

    # 평점 기반 가중치
    if rating >= 4:
        positive_count += 2
    elif rating <= 2:
        negative_count += 2
    elif rating == 3:
        neutral_count += 1

    if positive_count > negative_count and positive_count > neutral_count:
        sentiment = 'positive'
        score = min(1.0, 0.5 + positive_count * 0.1)
    elif negative_count > positive_count and negative_count > neutral_count:
        sentiment = 'negative'
        score = max(-1.0, -0.5 - negative_count * 0.1)
    else:
        sentiment = 'neutral'
        score = 0.0

    return sentiment, score


def label_single_review(row):
    """단일 리뷰 라벨링"""
    text = str(row.get('text', ''))
    rating = row.get('rating', 3)

    # Aspect 추출
    aspects = get_aspect_from_text(text)

    # Overall sentiment
    overall_sentiment, sentiment_score = get_sentiment_from_text(text, rating)

    # Aspect별 라벨 생성
    aspect_labels = []
    for asp in aspects:
        # Aspect별 sentiment는 전체와 동일하게 (간소화)
        aspect_sentiment, _ = get_sentiment_from_text(text, rating)

        aspect_labels.append({
            'aspect': asp['aspect'],
            'sentiment': aspect_sentiment,
            'confidence': round(asp['confidence'], 2),
            'reason': f"키워드 매칭: {', '.join(asp['matched'][:3])}" if asp['matched'] else "일반적 표현"
        })

    return {
        'sentiment': overall_sentiment,
        'sentiment_score': round(sentiment_score, 2),
        'aspect_labels': aspect_labels,
        'evidence': text[:100] + '...' if len(text) > 100 else text,
        'summary': text[:30] + '...' if len(text) > 30 else text
    }


def label_file(input_path, output_path):
    """파일 전체 라벨링"""
    df = pd.read_csv(input_path)
    print(f"라벨링 시작: {len(df)}개 리뷰")

    results = []
    for idx, row in df.iterrows():
        label = label_single_review(row)

        for asp_label in label['aspect_labels']:
            results.append({
                'review_idx': idx,
                'product_code': row.get('product_code', ''),
                'name': row.get('name', ''),
                'category_2': row.get('category_2', ''),
                'rating': row.get('rating', ''),
                'text': row.get('text', ''),
                'aspect': asp_label['aspect'],
                'aspect_sentiment': asp_label['sentiment'],
                'aspect_confidence': asp_label['confidence'],
                'aspect_reason': asp_label['reason'],
                'overall_sentiment': label['sentiment'],
                'sentiment_score': label['sentiment_score'],
                'summary': label['summary']
            })

        if (idx + 1) % 500 == 0:
            print(f"  진행: {idx + 1}/{len(df)}")

    # 저장
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"완료: {len(results)}개 라벨 -> {output_path}")

    return results_df


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python label_reviews_direct.py <input_file> [output_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.csv', '_labeled.csv')

    label_file(input_file, output_file)
