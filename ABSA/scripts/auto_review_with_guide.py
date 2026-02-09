#!/usr/bin/env python3
"""
aspect_evaluation_guide.md 규칙을 적용한 자동 검수 스크립트 v2
더 정밀한 키워드 매칭
"""

import json
import re
from pathlib import Path
from collections import defaultdict

# === 가이드 기반 키워드 정의 (정규식 패턴 사용) ===

ASPECT_PATTERNS = {
    "사용감/성능": {
        "patterns": [
            r"발림", r"지속력", r"커버력", r"흡수", r"촉촉", r"건조하", r"건조해", r"효과",
            r"보습", r"밀림", r"뭉침", r"자극", r"트러블", r"순하", r"민감", r"따가",
            r"가렵", r"자극없", r"부드럽", r"매끄럽", r"들뜸", r"밀착", r"착용감",
            r"피부결", r"화이트닝", r"미백", r"탄력", r"주름", r"모공", r"피지", r"유분",
            r"번들", r"산뜻", r"무거", r"가볍", r"쫀쫀", r"찐득", r"발림성", r"지속력",
            r"커버력", r"보습력", r"흡수력", r"밀착력", r"쿠션감", r"바르기", r"펴발",
            r"쓰기 좋", r"사용하기", r"바를때", r"효과없", r"효과 없", r"잘먹", r"먹힘",
            r"유지", r"오래가", r"순함", r"자극적", r"가려움", r"리프팅", r"진정",
            r"잡티", r"케어", r"관리", r"개선", r"차오름", r"촉감",
        ],
        "exclude_patterns": []
    },
    "재구매": {
        "patterns": [
            r"재구매", r"또\s?살", r"또\s?사[고려면]", r"다시\s?살", r"다시\s?사[고려면]",
            r"추천", r"계속\s?쓸", r"단골", r"애정템", r"인생템", r"최애", r"쟁여", r"쟁이",
            r"재주문", r"꾸준히", r"계속\s?사", r"안\s?살", r"재구매\s?의사",
        ],
        "exclude_patterns": []
    },
    "품질/효과": {
        "patterns": [
            r"불량", r"깨[짐져졌]", r"변질", r"유통기한", r"정품", r"가품", r"짝퉁",
            r"[세새]다", r"터[짐져졌]", r"찢어", r"파손", r"망가", r"고장", r"결함",
            r"하자", r"누수", r"샘", r"갈라[짐져졌]", r"부러", r"금이", r"금감",
            r"이상하", r"잘못\s?[온왔]",
        ],
        "exclude_patterns": [r"효과없", r"효과\s?없", r"효과가\s?없"]
    },
    "가격/가성비": {
        "patterns": [
            r"가격", r"비싸", r"비쌈", r"저렴", r"가성비", r"세일", r"할인",
            r"\d+원", r"이\s?가격", r"가격대", r"혜자", r"득템", r"착한\s?가격",
            r"돈\s?아깝", r"값어치", r"이\s?돈에",
        ],
        "exclude_patterns": []
    },
    "재질/냄새": {
        "patterns": [
            r"냄새", r"향[이가도은]", r"향기", r"텍스처", r"질감", r"무향", r"향수",
            r"악취", r"고약", r"역겨", r"비린", r"구수", r"달콤", r"상쾌", r"프루티",
            r"플로럴", r"시트러스", r"우디", r"머스크", r"은은한?\s?향", r"진한\s?향",
            r"향이\s?좋", r"향이\s?별로", r"냄새가\s?나", r"냄새나",
        ],
        "exclude_patterns": [r"발림", r"발림성"]
    },
    "색상/발색": {
        "patterns": [
            r"색[이가도은상깔감]", r"컬러", r"톤[이가도은]", r"발색", r"쿨톤", r"웜톤",
            r"MLBB", r"mlbb", r"누드", r"코랄", r"피치", r"핑크", r"레드", r"오렌지",
            r"브라운", r"베이지", r"버건디", r"로즈", r"와인", r"체리", r"퍼플",
            r"바이올렛", r"블루", r"그린", r"옐로", r"골드", r"실버", r"글리터", r"펄",
            r"쉬머", r"매트", r"무광", r"유광", r"광택", r"투명", r"선명", r"진한\s?색",
            r"연한\s?색", r"어두[워운]", r"밝[은게다]", r"화사", r"형광", r"비비드",
            r"파스텔", r"뮤트", r"딥[한해]", r"라이트", r"다크", r"칙칙", r"노[랗란]",
            r"붉[은게]", r"뻘[겋건]", r"새빨", r"살구", r"자연스러운\s?색", r"피부톤",
            r"웜블랙", r"쿨블랙", r"쿨브라운", r"웜브라운", r"무드", r"뉴트럴",
            r"예[쁜쁨].*색", r"이[쁜쁨].*색", r"색.*예[쁘쁨]", r"색.*이[쁘쁨]",
            r"봄웜", r"여름쿨", r"가을웜", r"겨울쿨", r"봄라이트", r"여쿨",
            r"착색", r"물빠짐", r"물들", r"컬러감", r"색조", r"색 이[쁘뻐]",
        ],
        "exclude_patterns": [r"패키지\s?색", r"박스\s?색", r"케이스\s?색"]
    },
    "배송/포장": {
        "patterns": [
            r"배송", r"택배", r"포장[이가도은을]", r"박스", r"배달", r"도착",
            r"늦게\s?[오와왔]", r"빨리\s?[오와왔]", r"하루만에", r"당일", r"익일",
            r"배송비", r"무료배송", r"퀵", r"로켓", r"새벽배송", r"오배송",
            r"누락", r"분실", r"완충재", r"에어캡", r"뽁뽁이", r"안전하게\s?포장",
            r"꼼꼼하게\s?포장", r"포장\s?꼼꼼",
        ],
        "exclude_patterns": [r"패키지가\s?예[쁘쁨]", r"패키지\s?디자인", r"용기\s?디자인"]
    },
    "용량/휴대": {
        "patterns": [
            r"용량", r"양이\s?[적많]", r"양도\s?[적많]", r"양\s?[적많]", r"휴대",
            r"여행용", r"미니", r"풀사이즈", r"대용량", r"소용량", r"작[아다]", r"커서",
            r"\d+ml", r"\d+g", r"파우치에\s?들어", r"주머니에", r"가방에",
            r"들고다니", r"휴대성", r"컴팩트", r"사이즈[가이]", r"내용물\s?[양적]",
            r"양\s?적", r"양\s?많", r"넉넉", r"작은\s?사이즈", r"크기[가이]",
            r"조금\s?들어", r"많이\s?들어", r"적게\s?들어", r"개수", r"\d+장",
            r"\d+매", r"들고\s?다니기",
        ],
        "exclude_patterns": []
    },
    "디자인": {
        "patterns": [
            r"케이스[가이]", r"패키지[가이]", r"용기[가이]", r"디자인[이가]",
            r"외관[이가]", r"생김[새이]", r"모양[이가]", r"예[쁘쁨][고다니]",
            r"이[쁘쁨][고다니]", r"귀[엽여]", r"고급[스스럽져진]",
            r"심플", r"감성", r"인테리어", r"세련", r"깔끔한\s?디자인",
            r"튜브", r"펌프", r"브러쉬", r"어플리케이터", r"뚜껑", r"캡[이가]",
            r"저렴해\s?보", r"케이스\s?예[쁘쁨]", r"패키지\s?예[쁘쁨]",
        ],
        "exclude_patterns": [r"색[이가]", r"컬러[가이]", r"발색"]
    },
    "CS/응대": {
        "patterns": [
            r"교환", r"환불", r"문의", r"응대", r"고객센터", r"상담", r"직원[이분]",
            r"서비스[가이]", r"답변", r"처리", r"반품", r"클레임", r"불친절", r"친절",
            r"빠른\s?응대", r"느린\s?응대", r"연락[이이]", r"전화[가를]",
            r"채팅", r"카톡", r"톡상담", r"메일", r"이메일",
        ],
        "exclude_patterns": []
    },
}


def check_pattern_match(text, patterns, exclude_patterns=None):
    """정규식 패턴 매칭 확인"""
    if exclude_patterns is None:
        exclude_patterns = []

    text_lower = text.lower()

    # exclude 패턴이 있으면 False
    for pattern in exclude_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return False

    # include 패턴 확인
    for pattern in patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True

    return False


def find_aspects_from_text(text):
    """텍스트에서 aspect 추출"""
    found_aspects = set()

    for aspect, rules in ASPECT_PATTERNS.items():
        if check_pattern_match(text, rules["patterns"], rules.get("exclude_patterns", [])):
            found_aspects.add(aspect)

    return list(found_aspects)


def validate_aspects(text, current_aspects):
    """현재 aspect 검증"""
    validated = set()

    for aspect in current_aspects:
        if aspect in ASPECT_PATTERNS:
            rules = ASPECT_PATTERNS[aspect]
            if check_pattern_match(text, rules["patterns"], rules.get("exclude_patterns", [])):
                validated.add(aspect)
        else:
            # 알 수 없는 aspect는 유지
            validated.add(aspect)

    return list(validated)


def fix_misclassifications(text, current_aspects):
    """오분류 패턴 수정 (가이드 5.1 기준)"""
    fixed_aspects = set(current_aspects)

    # 1. 사용감 → 색상: "발색"이 있으면 색상/발색 추가
    if re.search(r"발색", text, re.IGNORECASE):
        fixed_aspects.add("색상/발색")

    # 2. 디자인에서 색상 분리: "색이 예뻐요" 등
    if re.search(r"색[이가도].*예[쁘쁨뻐]|예[쁘쁨뻐].*색", text, re.IGNORECASE):
        fixed_aspects.add("색상/발색")
        if "디자인" in fixed_aspects and not re.search(r"패키지|케이스|용기|디자인", text, re.IGNORECASE):
            fixed_aspects.discard("디자인")

    # 3. 품질 → 사용감: "효과없음"은 사용감
    if re.search(r"효과[가이]?\s?(없|없어|미비)", text, re.IGNORECASE):
        fixed_aspects.add("사용감/성능")
        if "품질/효과" in fixed_aspects and not re.search(r"불량|깨[짐져]|파손|변질", text, re.IGNORECASE):
            fixed_aspects.discard("품질/효과")

    # 4. 사용감 → 재질: "향이 좋아요" 등
    if re.search(r"향[이가도은]|냄새", text, re.IGNORECASE):
        fixed_aspects.add("재질/냄새")

    # 5. 배송 → 디자인: "패키지가 예뻐요"
    if re.search(r"패키지[가이]?\s?예[쁘쁨뻐]|용기[가이]?\s?예[쁘쁨뻐]", text, re.IGNORECASE):
        fixed_aspects.add("디자인")
        if "배송/포장" in fixed_aspects and not re.search(r"배송|택배|도착", text, re.IGNORECASE):
            fixed_aspects.discard("배송/포장")

    return list(fixed_aspects)


def check_boundary_cases(text, current_aspects):
    """경계 케이스 처리 (가이드 5.2 기준)"""
    new_aspects = set(current_aspects)

    # "예쁘다" - 제품 색상 vs 패키지
    if re.search(r"예[쁘쁨뻐]|이[쁘쁨뻐]", text, re.IGNORECASE):
        if re.search(r"색[이가도]|컬러|발색|톤", text, re.IGNORECASE):
            new_aspects.add("색상/발색")
        if re.search(r"패키지|케이스|용기|디자인", text, re.IGNORECASE):
            new_aspects.add("디자인")

    # "양이 적다/많다" - 내용물 용량
    if re.search(r"양[이도가]\s?[적많]|용량", text, re.IGNORECASE):
        new_aspects.add("용량/휴대")

    # "냄새" - 변질 vs 제품향
    if re.search(r"냄새|향[이가도]", text, re.IGNORECASE):
        if re.search(r"변질|썩|상[한해]|이상한\s?냄새", text, re.IGNORECASE):
            new_aspects.add("품질/효과")
        else:
            new_aspects.add("재질/냄새")

    return list(new_aspects)


def review_record(record):
    """단일 레코드 검토 및 수정"""
    text = record.get("text", "")
    current_aspects = record.get("aspect_labels", [])

    if isinstance(current_aspects, str):
        current_aspects = [a.strip() for a in current_aspects.split(",")]

    # 빈 리스트면 그대로 반환 (aspect 없는 리뷰)
    original_aspects = list(current_aspects)

    # 1. 텍스트에서 aspect 추출
    detected_aspects = find_aspects_from_text(text)

    # 2. 현재 aspect 검증 (증거 없는 것 제거)
    validated_aspects = validate_aspects(text, current_aspects)

    # 3. 오분류 수정
    fixed_aspects = fix_misclassifications(text, validated_aspects)

    # 4. 누락된 aspect 추가 (검출된 것 중)
    for asp in detected_aspects:
        if asp not in fixed_aspects:
            fixed_aspects.append(asp)

    # 5. 경계 케이스 처리
    final_aspects = check_boundary_cases(text, fixed_aspects)

    # 6. 최종 검증 - 증거 있는 것만 유지
    final_validated = []
    for asp in final_aspects:
        if asp in ASPECT_PATTERNS:
            rules = ASPECT_PATTERNS[asp]
            if check_pattern_match(text, rules["patterns"], rules.get("exclude_patterns", [])):
                final_validated.append(asp)
        else:
            final_validated.append(asp)

    # 중복 제거 및 정렬
    final_validated = sorted(list(set(final_validated)))

    # 변경 여부 확인
    original_set = set(original_aspects)
    new_set = set(final_validated)
    modified = original_set != new_set

    return final_validated, modified


def update_summary(record, new_aspects):
    """summary 업데이트"""
    sentiment = record.get("sentiment", "neutral")
    sentiment_text = {
        "positive": "긍정적",
        "neutral": "중립적",
        "negative": "부정적"
    }.get(sentiment, "중립적")

    if new_aspects:
        aspects_str = ", ".join(new_aspects)
        return f"{aspects_str}에 대해 {sentiment_text}"
    return record.get("summary", "")


def main():
    # 입력 파일
    input_file = Path("/Users/yu_seok/Documents/workspace/nbCamp/Project/Why-pi/ABSA/data/raw/chatgpt_labels_20k_validated_v2.jsonl")
    output_file = Path("/Users/yu_seok/Documents/workspace/nbCamp/Project/Why-pi/ABSA/data/raw/chatgpt_labels_20k_reviewed_v2.jsonl")

    print("=" * 60)
    print("ABSA 자동 검수 v2 (aspect_evaluation_guide.md 규칙 적용)")
    print("=" * 60)

    # 통계
    stats = {
        "total": 0,
        "modified": 0,
        "aspect_changes": defaultdict(lambda: {"added": 0, "removed": 0}),
    }

    results = []

    # 파일 읽기 및 처리
    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            stats["total"] += 1

            # 원본 aspects
            original_aspects = record.get("aspect_labels", [])
            if isinstance(original_aspects, str):
                original_aspects = [a.strip() for a in original_aspects.split(",")]
            original_set = set(original_aspects)

            # 검토 및 수정
            new_aspects, modified = review_record(record)
            new_set = set(new_aspects)

            if modified:
                stats["modified"] += 1

                # 변경 통계
                added = new_set - original_set
                removed = original_set - new_set

                for asp in added:
                    stats["aspect_changes"][asp]["added"] += 1
                for asp in removed:
                    stats["aspect_changes"][asp]["removed"] += 1

                # 레코드 업데이트
                record["aspect_labels"] = new_aspects
                record["summary"] = update_summary(record, new_aspects)

            results.append(record)

            # 진행 상황 출력
            if stats["total"] % 2000 == 0:
                print(f"처리 중... {stats['total']:,}건 (수정: {stats['modified']:,}건)")

    # 결과 저장
    with open(output_file, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 최종 통계 출력
    print("\n" + "=" * 60)
    print("검수 완료!")
    print("=" * 60)
    print(f"총 처리: {stats['total']:,}건")
    print(f"수정됨: {stats['modified']:,}건 ({stats['modified']/stats['total']*100:.1f}%)")
    print(f"\n출력 파일: {output_file}")

    print("\n[Aspect별 변경 통계]")
    for aspect in sorted(stats["aspect_changes"].keys()):
        changes = stats["aspect_changes"][aspect]
        print(f"  {aspect}: +{changes['added']:,} / -{changes['removed']:,}")

    return stats


if __name__ == "__main__":
    main()
