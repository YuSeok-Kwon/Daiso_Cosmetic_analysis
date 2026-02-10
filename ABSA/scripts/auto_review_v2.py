#!/usr/bin/env python3
"""
ABSA 재검토 스크립트 v2
- 품질: 물리적 결함/하자만 (배송파손 제외)
- sentiment_score 극단화
- 복합 감성 분리
"""

import json
import re
from pathlib import Path
from collections import defaultdict

# === Aspect 정의 ===
ASPECT_PATTERNS = {
    "사용감/성능": {
        "patterns": [
            r"발림", r"지속력", r"커버력", r"흡수", r"촉촉", r"건조하", r"건조해",
            r"보습", r"밀림", r"뭉침", r"자극", r"트러블", r"순하", r"민감", r"따가",
            r"가렵", r"부드럽", r"매끄럽", r"들뜸", r"밀착", r"착용감",
            r"피부결", r"화이트닝", r"미백", r"탄력", r"주름", r"모공", r"피지", r"유분",
            r"번들", r"산뜻", r"무거", r"가볍", r"쫀쫀", r"발림성",
            r"바르기", r"쓰기\s?좋", r"효과", r"잘먹", r"유지", r"오래가",
            r"리프팅", r"진정", r"잡티", r"케어", r"개선",
        ],
    },
    "재구매": {
        "patterns": [
            r"또\s?살", r"또\s?사[고려면]", r"다시\s?살", r"다시\s?사[고려면]",
            r"추천[합해드]", r"계속\s?쓸", r"단골", r"애정템", r"인생템", r"최애",
            r"쟁여", r"쟁이", r"재주문", r"꾸준히", r"계속\s?사",
            r"안\s?살\s?것", r"재구매\s?의사", r"재구매\s?[할합]",
        ],
    },
    "품질": {
        "patterns": [
            r"불량[이품]", r"변질", r"유통기한", r"정품", r"가품", r"짝퉁",
            r"결함", r"하자[가이]", r"누수", r"이상\s?없[이어]",
            r"정상\s?제품", r"깨짐\s?없", r"새는\s?거\s?없",
        ],
        "exclude": [r"배송", r"택배", r"포장", r"오배송", r"잘못\s?[온왔]", r"다른\s?상품"]
    },
    "가격/가성비": {
        "patterns": [
            r"가격", r"비싸", r"비쌈", r"저렴", r"가성비", r"세일", r"할인",
            r"\d+원", r"이\s?가격", r"가격대", r"혜자", r"득템", r"착한\s?가격",
        ],
    },
    "재질/냄새": {
        "patterns": [
            r"냄새", r"향[이가도은]", r"향기", r"텍스처", r"질감", r"무향",
            r"악취", r"고약", r"역겨", r"달콤", r"상쾌", r"은은한?\s?향",
        ],
    },
    "색상/발색": {
        "patterns": [
            r"색[이가도은상깔감]", r"컬러", r"톤[이가도은]", r"발색", r"쿨톤", r"웜톤",
            r"핑크", r"레드", r"오렌지", r"브라운", r"베이지", r"버건디", r"로즈",
            r"퍼플", r"블루", r"그린", r"골드", r"실버", r"글리터", r"펄",
            r"매트", r"무광", r"유광", r"화사", r"형광", r"파스텔", r"뮤트",
            r"착색", r"컬러감", r"피부톤",
        ],
    },
    "배송/포장": {
        "patterns": [
            r"배송", r"택배", r"포장[이가도은을]", r"박스", r"배달", r"도착",
            r"늦게", r"빨리", r"하루만에", r"당일", r"익일", r"배송비",
            r"오배송", r"누락", r"분실", r"완충재", r"에어캡",
            r"파손", r"깨[져짐졌]", r"터[져짐졌]", r"찢어", r"젖어", r"뚜껑.*풀",
        ],
    },
    "용량/휴대": {
        "patterns": [
            r"용량", r"양이\s?[적많]", r"양도\s?[적많]", r"휴대", r"여행용", r"미니",
            r"대용량", r"소용량", r"작[아다]", r"커서", r"\d+ml", r"\d+g",
            r"들고다니", r"휴대성", r"컴팩트", r"사이즈",
        ],
    },
    "디자인": {
        "patterns": [
            r"케이스[가이]", r"패키지[가이]", r"용기[가이]", r"디자인[이가]",
            r"외관", r"예[쁘쁨]", r"이[쁘쁨]", r"귀[엽여]", r"고급[스져]",
            r"심플", r"세련", r"튜브", r"펌프", r"브러쉬", r"뚜껑",
        ],
    },
    "CS/응대": {
        "patterns": [
            r"교환", r"환불", r"문의", r"응대", r"고객센터", r"상담",
            r"서비스", r"답변", r"처리", r"반품", r"클레임", r"불친절", r"친절",
        ],
    },
}

# === 감성 점수 패턴 ===
STRONG_NEGATIVE_PATTERNS = [
    r"최악", r"짜증", r"화[가나]", r"빡[치쳐]", r"열받", r"다시는",
    r"절대\s?안", r"후회", r"돈\s?아깝", r"버[려렸]", r"못\s?쓰",
    r"환불", r"교환\s?신청", r"오배송", r"파손", r"깨[져짐]",
]

WEAK_NEGATIVE_PATTERNS = [
    r"그냥\s?그", r"쏘쏘", r"별로", r"아쉬", r"글쎄", r"모르겠",
    r"그저\s?그", r"보통", r"평범", r"무난",
]

# 텍스트 기반 부정 패턴 (rating과 무관)
TEXT_NEGATIVE_PATTERNS = [
    r"안\s?[되빨나]", r"못\s?[쓰써]", r"잘\s?안", r"조금\s?들어",
    r"불량인[지가]", r"적[어다]", r"없[어다네]", r"안\s?좋",
]

STRONG_POSITIVE_PATTERNS = [
    r"최고", r"대박", r"완전\s?좋", r"인생템", r"최애", r"강추",
    r"완벽", r"짱짱", r"미쳤", r"혁명", r"감동",
]


def calculate_sentiment_score(text, sentiment):
    """sentiment_score 계산 - 더 극단적으로"""

    if sentiment == "positive":
        # 강한 긍정 체크
        for pattern in STRONG_POSITIVE_PATTERNS:
            if re.search(pattern, text):
                return 1.0
        return 0.8

    elif sentiment == "negative":
        # 강한 부정 체크
        for pattern in STRONG_NEGATIVE_PATTERNS:
            if re.search(pattern, text):
                return -1.0
        # 텍스트 기반 부정 패턴이 많으면 더 강한 부정
        negative_count = sum(1 for p in TEXT_NEGATIVE_PATTERNS if re.search(p, text))
        if negative_count >= 2:
            return -0.8
        return -0.7

    else:  # neutral
        # 약한 부정 체크 (그냥 그래요 등)
        for pattern in WEAK_NEGATIVE_PATTERNS:
            if re.search(pattern, text):
                return -0.3
        # 텍스트 기반 부정이 있으면 약한 부정
        for pattern in TEXT_NEGATIVE_PATTERNS:
            if re.search(pattern, text):
                return -0.2
        return 0.0


def check_is_delivery_damage(text):
    """배송 중 파손인지 확인"""
    delivery_keywords = [r"배송", r"택배", r"포장", r"박스", r"도착", r"받"]
    damage_keywords = [r"파손", r"깨[져짐졌]", r"터[져짐졌]", r"찢어", r"젖어", r"뚜껑.*풀"]

    has_delivery = any(re.search(p, text) for p in delivery_keywords)
    has_damage = any(re.search(p, text) for p in damage_keywords)

    return has_delivery and has_damage


def check_repurchase_tag_only(text):
    """'재구매' 태그만 있고 실제 재구매 의사가 없는지 확인"""
    # 맨 앞에 "재구매"로 시작하는 경우
    if re.match(r"^재구매\s", text):
        # 실제 재구매 의사 표현이 있는지 확인
        real_patterns = [r"또\s?살", r"다시\s?사", r"추천", r"계속\s?쓸", r"쟁여", r"재주문"]
        has_real_intent = any(re.search(p, text) for p in real_patterns)
        return not has_real_intent
    return False


def find_aspects(text):
    """텍스트에서 aspect 추출"""
    found = set()

    for aspect, rules in ASPECT_PATTERNS.items():
        patterns = rules["patterns"]
        exclude = rules.get("exclude", [])

        # exclude 패턴 체크
        if exclude:
            has_exclude = any(re.search(p, text) for p in exclude)
            if has_exclude and aspect == "품질":
                continue

        # include 패턴 체크
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                found.add(aspect)
                break

    # 특수 케이스: 배송 중 파손은 품질에서 제외
    if "품질" in found and check_is_delivery_damage(text):
        found.discard("품질")
        found.add("배송/포장")

    # 특수 케이스: 재구매 태그만 있는 경우 제외
    if "재구매" in found and check_repurchase_tag_only(text):
        found.discard("재구매")

    # 특수 케이스: 색상 맥락에서 "불량" 의문문은 품질에서 제외
    if "품질" in found:
        color_context = [r"색[이가]", r"컬러", r"발색", r"톤", r"누[래렇]", r"진[하해]", r"연[하해]"]
        has_color = any(re.search(p, text) for p in color_context)
        is_question = re.search(r"불량[품]?[을를이가]?\s?(받은|인가|인건|인지)", text)
        if has_color and is_question:
            found.discard("품질")

        # "불량식품" 등 비유적 표현은 품질에서 제외 (오타 포함)
        is_metaphor = re.search(r"불량식[품훔푸프폼]|불량\s?음식", text)
        if is_metaphor:
            found.discard("품질")

        # "가품"이 "거품" 오타인 경우 품질에서 제외
        if re.search(r"가품", text) and re.search(r"풍성|거품|거[푼폼]", text):
            found.discard("품질")

        # 순수 사용감 표현만 있고 실제 품질 불량 언급이 없는 경우 제외
        # "자극 없어요", "이상 없어요" 등 긍정적 표현은 품질이 아님
        no_issue_positive = re.search(r"(자극|문제|이상)\s?없", text)
        real_defect = re.search(r"불량[이품을가]|변질|결함|하자[가이]|깨[져짐]|터[져짐]", text)
        if no_issue_positive and not real_defect:
            found.discard("품질")

        # "하자면", "하자!" 등 구어체 표현은 품질 하자가 아님
        if re.search(r"하자면|하자[!~]|좀\s?하자", text) and not re.search(r"하자[가이]?\s?(있|제품|상품)", text):
            found.discard("품질")

        # "짝퉁이라 해서", "짝퉁 느낌", "짝퉁쓰는" 등 비유적 표현 제외
        if re.search(r"짝퉁(이라|쓰는|느낌|같)", text):
            found.discard("품질")

        # "정품대비", "정품 대비" 가격 비교 맥락 제외
        if re.search(r"정품\s?대비", text):
            found.discard("품질")

        # "XX 가품 면도기" 등 비교 대상으로서의 가품 제외
        if re.search(r"가품\s?(면도기|제품|브랜드)", text) or re.search(r"(브라운|필립스|다이슨)\s?가품", text):
            found.discard("품질")

        # 피부 부작용/반응만 있고 실제 불량 언급이 없는 경우 → 사용감/성능
        skin_reaction = re.search(r"빨갛|붉[은어]|부[풀어]|두드러기|뒤집어|따[끔가]|자극|트러블", text)
        explicit_defect = re.search(r"불량[이품을가]|변질|결함|하자[가이]", text)
        if skin_reaction and not explicit_defect:
            found.discard("품질")

        # "불량인줄 알았", "불량인가 했는데" 등 실제 불량이 아닌 경우 제외
        not_actually_defect = re.search(r"불량인줄\s?알|불량인가\s?했|불량이었는진\s?모르겠", text)
        if not_actually_defect:
            found.discard("품질")

        # "불량품받았나 생각들정도" - 비유적 표현
        metaphor_thinking = re.search(r"불량[품]?\s?(받았나|인가)\s?생각|불량[품]?인걸까.*궁금", text)
        if metaphor_thinking:
            found.discard("품질")

        # "가끔 파손", "가끔 불량" - 일반적 언급 (이번 제품 아님)
        general_mention = re.search(r"가끔\s?(파손|불량)", text)
        if general_mention:
            found.discard("품질")

        # "불량이 없더라도" - 현재 제품은 불량 아님
        no_defect_anyway = re.search(r"불량[이가]?\s?없더라도", text)
        if no_defect_anyway:
            found.discard("품질")

    return list(found)


def determine_sentiment(text, rating):
    """sentiment 결정"""
    # "불량이었는진 모르겠지만 잘 사용중" 같은 경우 → rating 우선
    not_sure_defect = re.search(r"불량이었는진\s?모르겠|불량인줄\s?알았", text)
    positive_ending = re.search(r"잘\s?사용|괜찮|좋아요|만족", text)
    if not_sure_defect and positive_ending and rating >= 4:
        return "positive"
    if not_sure_defect and rating >= 4:
        return "neutral"

    # 강한 부정 패턴
    for pattern in STRONG_NEGATIVE_PATTERNS:
        if re.search(pattern, text):
            return "negative"

    # 텍스트 기반 부정 패턴 (rating과 무관하게 부정으로 판단)
    negative_count = sum(1 for p in TEXT_NEGATIVE_PATTERNS if re.search(p, text))
    if negative_count >= 2:  # 부정 패턴이 2개 이상이면 부정
        return "negative"

    # 약한 부정 패턴
    for pattern in WEAK_NEGATIVE_PATTERNS:
        if re.search(pattern, text):
            if rating <= 3:
                return "negative"
            return "neutral"

    # 텍스트 기반 부정이 1개 있고 긍정 표현이 없으면 neutral
    if negative_count == 1:
        has_positive = any(re.search(p, text) for p in STRONG_POSITIVE_PATTERNS)
        if not has_positive:
            return "neutral"

    # rating 기반
    if rating >= 4:
        return "positive"
    elif rating <= 2:
        return "negative"
    else:
        return "neutral"


def review_record(record):
    """레코드 검토 및 수정"""
    text = record.get("text", "")
    rating = record.get("rating", 3)
    original_aspects = record.get("aspect_labels", [])

    if isinstance(original_aspects, str):
        original_aspects = [a.strip() for a in original_aspects.split(",")]

    # 1. 품질/효과 → 품질 변환
    original_aspects = [a.replace("품질/효과", "품질").replace("품질/불량", "품질") for a in original_aspects]

    # 2. Aspect 재검출
    detected = find_aspects(text)

    # 3. 기존 aspect 검증 및 병합 (증거 있는 것만)
    final_aspects = set()
    for asp in original_aspects:
        if asp in ASPECT_PATTERNS:
            # 품질은 명확한 키워드가 있어야만 유지
            if asp == "품질":
                quality_keywords = [r"불량", r"변질", r"유통기한", r"정품", r"가품", r"결함", r"하자"]
                has_quality_evidence = any(re.search(p, text) for p in quality_keywords)
                # 배송 관련 키워드가 있으면 제외
                delivery_keywords = [r"배송", r"택배", r"오배송", r"잘못\s?[온왔]", r"다른\s?상품"]
                has_delivery = any(re.search(p, text) for p in delivery_keywords)
                # 색상 관련 맥락에서 "불량" 언급은 제외 (색이 마음에 안 드는 것)
                color_context = [r"색[이가]", r"컬러", r"발색", r"톤", r"누[래렇]", r"진[하해]", r"연[하해]"]
                has_color_context = any(re.search(p, text) for p in color_context)
                # "불량품인가요", "불량인건가", "불량품을 받은건가" 등 의문문은 제외
                is_question = re.search(r"불량[품]?[을를이가]?\s?(받은|인가|인건|인지)", text)

                # "불량식품" 등 비유적 표현은 제외 (오타 포함)
                is_metaphor = re.search(r"불량식[품훔푸프폼]|불량\s?음식", text)

                # "가품"이 "거품" 오타인 경우 제외
                is_gaepum_typo = re.search(r"가품", text) and re.search(r"풍성|거품|거[푼폼]", text)

                # 순수 사용감 표현 ("자극 없어요" 등)은 품질이 아님
                no_issue_positive = re.search(r"(자극|문제|이상)\s?없", text)
                real_defect = re.search(r"불량[이품을가]|변질|결함|하자[가이]|깨[져짐]|터[져짐]", text)
                is_only_positive_usage = no_issue_positive and not real_defect

                # "하자면", "하자!" 등 구어체 표현
                is_haja_colloquial = re.search(r"하자면|하자[!~]|좀\s?하자", text) and not re.search(r"하자[가이]?\s?(있|제품|상품)", text)

                # "짝퉁이라 해서", "짝퉁 느낌" 등 비유적 표현
                is_jjaktoong_metaphor = re.search(r"짝퉁(이라|쓰는|느낌|같)", text)

                # "정품대비" 가격 비교
                is_jungpum_compare = re.search(r"정품\s?대비", text)

                # "XX 가품 면도기" 등 비교 대상
                is_gapum_compare = re.search(r"가품\s?(면도기|제품|브랜드)", text) or re.search(r"(브라운|필립스|다이슨)\s?가품", text)

                # 피부 부작용/반응만 있고 실제 불량 없음
                skin_reaction = re.search(r"빨갛|붉[은어]|부[풀어]|두드러기|뒤집어|따[끔가]|자극|트러블", text)
                explicit_defect = re.search(r"불량[이품을가]|변질|결함|하자[가이]", text)
                is_skin_reaction_only = skin_reaction and not explicit_defect

                # "불량인줄 알았", "불량인가 했는데" 등 실제 불량이 아닌 경우
                not_actually_defect = re.search(r"불량인줄\s?알|불량인가\s?했|불량이었는진\s?모르겠", text)

                # "불량품받았나 생각들정도" - 비유적 표현
                metaphor_thinking = re.search(r"불량[품]?\s?(받았나|인가)\s?생각|불량[품]?인걸까.*궁금", text)

                # "가끔 파손", "가끔 불량" - 일반적 언급
                general_mention = re.search(r"가끔\s?(파손|불량)", text)

                # "불량이 없더라도" - 현재 제품은 불량 아님
                no_defect_anyway = re.search(r"불량[이가]?\s?없더라도", text)

                if not has_quality_evidence or has_delivery or (has_color_context and is_question) or is_metaphor or is_gaepum_typo or is_only_positive_usage or is_haja_colloquial or is_jjaktoong_metaphor or is_jungpum_compare or is_gapum_compare or is_skin_reaction_only or not_actually_defect or metaphor_thinking or general_mention or no_defect_anyway:
                    continue
            # 재구매 태그만 있으면 제외
            if asp == "재구매" and check_repurchase_tag_only(text):
                continue
            final_aspects.add(asp)

    # 검출된 것 추가
    for asp in detected:
        final_aspects.add(asp)

    # 4. Sentiment 재결정
    sentiment = determine_sentiment(text, rating)

    # 5. Sentiment score 재계산
    sentiment_score = calculate_sentiment_score(text, sentiment)

    # 최종
    final_aspects = sorted(list(final_aspects))

    return final_aspects, sentiment, sentiment_score


def update_summary(aspects, sentiment):
    """summary 업데이트"""
    sentiment_text = {
        "positive": "긍정적",
        "neutral": "중립적",
        "negative": "부정적"
    }.get(sentiment, "중립적")

    if aspects:
        return f"{', '.join(aspects)}에 대해 {sentiment_text}"
    return f"제품에 대해 {sentiment_text}"


def main():
    # 원본 파일에서 시작
    input_file = Path("/Users/yu_seok/Documents/workspace/nbCamp/Project/Why-pi/ABSA/data/raw/chatgpt_labels_20k_validated_v2.jsonl")
    output_file = Path("/Users/yu_seok/Documents/workspace/nbCamp/Project/Why-pi/ABSA/data/raw/chatgpt_labels_20k_reviewed_v2.jsonl")

    print("=" * 60)
    print("ABSA 재검토 v2")
    print("=" * 60)

    stats = {
        "total": 0,
        "aspect_modified": 0,
        "sentiment_modified": 0,
        "score_modified": 0,
    }

    results = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            stats["total"] += 1

            original_aspects = record.get("aspect_labels", [])
            original_sentiment = record.get("sentiment", "neutral")
            original_score = record.get("sentiment_score", 0)

            # 검토
            new_aspects, new_sentiment, new_score = review_record(record)

            # 변경 통계
            if set(original_aspects) != set(new_aspects):
                stats["aspect_modified"] += 1
            if original_sentiment != new_sentiment:
                stats["sentiment_modified"] += 1
            if abs(original_score - new_score) > 0.1:
                stats["score_modified"] += 1

            # 업데이트
            record["aspect_labels"] = new_aspects
            record["sentiment"] = new_sentiment
            record["sentiment_score"] = new_score
            record["summary"] = update_summary(new_aspects, new_sentiment)

            results.append(record)

            if stats["total"] % 2000 == 0:
                print(f"처리 중... {stats['total']:,}건")

    # 저장
    with open(output_file, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)
    print(f"총 처리: {stats['total']:,}건")
    print(f"Aspect 수정: {stats['aspect_modified']:,}건")
    print(f"Sentiment 수정: {stats['sentiment_modified']:,}건")
    print(f"Score 수정: {stats['score_modified']:,}건")


if __name__ == "__main__":
    main()
