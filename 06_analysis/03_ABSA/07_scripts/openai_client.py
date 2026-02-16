"""
OpenAI API client for ABSA labeling with rate limiting, caching, and cost tracking.
"""
import json
import time
import os
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
import hashlib

try:
    from openai import OpenAI, APIError, RateLimitError, APIConnectionError
except ImportError:
    raise ImportError("Please install openai: pip install openai>=1.12.0")


@dataclass
class LabelingResult:
    """Result from ChatGPT labeling"""
    sentiment: str
    sentiment_score: float
    aspect_labels: List[str]
    evidence: str
    summary: str
    model: str
    tokens_input: int
    tokens_output: int
    cost: float


class OpenAIClient:
    """
    Singleton OpenAI client with rate limiting, caching, and cost tracking.
    """
    _instance = None

    # Pricing (USD per 1K tokens)
    PRICING = {
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},  # Judge 모델
        "gpt-4.1": {"input": 0.001, "output": 0.004}
    }

    # Fixed aspect labels (검수 결과 반영 v4 - 미분류 추가)
    ASPECT_LABELS = [
        "배송/포장",
        "품질/퀄리티",
        "가격/가성비",
        "사용감/성능",
        "용량/휴대",
        "디자인",
        "재질/냄새",
        "CS/응대",
        "재구매",
        "색상/발색",
        "미분류"
    ]

    # Aspect 정의 및 키워드 (검수 결과 반영 v3 - 구분 기준 강화)
    ASPECT_DEFINITIONS = {
        "배송/포장": {
            "정의": "배송 속도, 택배 상태, 포장 품질 (배송 중 파손 포함)",
            "키워드": ["배송", "택배", "포장", "박스", "배송 중 파손", "배달", "도착"],
            "예시": "배송 빨라요, 포장이 꼼꼼해요, 배송 중 깨짐"
        },
        "품질/퀄리티": {
            "정의": "제품 자체의 물리적·제조적 결함만 해당. 사용 경험/효과는 '사용감/성능'으로 분류",
            "키워드": ["불량", "변질", "유통기한", "정품", "하자", "깨짐", "금감", "찍힘", "부러짐"],
            "예시": "받자마자 깨져있음, 유통기한 임박, 정품인지 의심",
            "주의": "좋아요/괜찮아요/효과있어요 등 사용 후 만족 표현은 '사용감/성능'임"
        },
        "가격/가성비": {
            "정의": "가격의 적정성 및 가격 대비 만족도",
            "키워드": ["싸다", "비싸다", "저렴", "세일", "할인", "가성비", "가격", "천원", "이 가격에"],
            "예시": "이 가격에 이 퀄리티 대박, 가성비 좋아요"
        },
        "사용감/성능": {
            "정의": "제품 사용 시 느끼는 경험, 효과, 만족도 (화장품의 핵심 기능)",
            "키워드": ["발림", "지속력", "커버력", "보습", "촉촉", "좋아요", "별로", "만족", "순하다", "자극없다", "효과"],
            "예시": "발림성 좋아요, 보습력 최고, 커버력 별로, 순해서 좋아요",
            "주의": "'좋다/괜찮다/만족' 같은 일반적 호평은 품질/퀄리티가 아닌 '사용감/성능'임"
        },
        "용량/휴대": {
            "정의": "제품의 용량 크기 및 휴대 편의성",
            "키워드": ["용량", "양", "크기", "사이즈", "휴대", "여행용", "미니", "작다", "크다", "적다", "많다"],
            "예시": "양이 적어요, 휴대하기 좋아요, 여행용으로 딱"
        },
        "디자인": {
            "정의": "패키징 외관, 용기 디자인, 용기 구조적 문제 (펌프, 뚜껑 등)",
            "키워드": ["케이스", "패키지", "용기", "구조", "예쁘다", "이쁘다", "펌프", "뚜껑", "디자인", "귀엽다"],
            "예시": "케이스가 예뻐요, 펌프가 고장났어요, 뚜껑이 잘 안 닫혀요"
        },
        "재질/냄새": {
            "정의": "제품의 물리적 제형(텍스처), 질감, 향기/냄새",
            "키워드": ["냄새", "향", "텍스처", "질감", "끈적", "묽다", "되다", "제형", "크리미", "젤", "오일", "향기"],
            "예시": "향이 좋아요, 텍스처가 가벼워요, 끈적임이 있어요, 냄새가 이상해요"
        },
        "CS/응대": {
            "정의": "교환, 환불, 고객센터, 매장 직원 응대, 재고 문의 관련 경험",
            "키워드": ["교환", "환불", "문의", "응대", "고객센터", "직원", "매장", "재고", "품절", "구하기 힘들다"],
            "예시": "교환 해줬어요, 직원분이 친절해요, 재고 없어서 못 샀어요"
        },
        "재구매": {
            "정의": "재구매 의사 또는 타인 추천 의사 표현",
            "키워드": ["또 살", "재구매", "추천", "단골", "쟁여둠", "정착", "계속 쓸", "집어들"],
            "예시": "또 살 거예요, 강추합니다, 쟁여뒀어요, 정착템"
        },
        "색상/발색": {
            "정의": "색상 구현력, 발색 정도, 피부톤 적합성",
            "키워드": ["색", "컬러", "톤", "발색", "착색", "쿨톤", "웜톤", "뜨다", "어둡다", "밝다", "예쁜 색"],
            "예시": "발색이 예뻐요, 쿨톤한테 안 맞아요, 색이 떠요"
        }
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")

            self.client = OpenAI(api_key=api_key)
            # Cache file path relative to this script's location
            self.cache_file = Path(__file__).parent / "data" / "cache" / "chatgpt_cache.json"
            self.cache = self._load_cache()
            self.total_cost = 0.0
            self.total_requests = 0

            # Rate limiting
            self.rate_limit_rpm = 60
            self.rate_limit_tpm = 90000
            self.request_times = []
            self.token_counts = []

            self.initialized = True

    def _load_cache(self) -> Dict:
        """Load cache from disk"""
        if self.cache_file.exists():
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        """Save cache to disk"""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _get_cache_key(self, review_text: str, product_code: int, rating: int, model: str, name: str = "", category_1: str = "", category_2: str = "") -> str:
        """Generate cache key"""
        content = f"{review_text}|{product_code}|{rating}|{model}|{name}|{category_1}|{category_2}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _wait_for_rate_limit(self, estimated_tokens: int):
        """Wait if necessary to respect rate limits"""
        current_time = time.time()

        # Remove old request times (older than 1 minute)
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        self.token_counts = self.token_counts[-len(self.request_times):]

        # Check RPM limit
        if len(self.request_times) >= self.rate_limit_rpm:
            sleep_time = 60 - (current_time - self.request_times[0]) + 1
            if sleep_time > 0:
                print(f"Rate limit (RPM): sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)

        # Check TPM limit
        total_tokens = sum(self.token_counts) + estimated_tokens
        if total_tokens >= self.rate_limit_tpm:
            sleep_time = 60 - (current_time - self.request_times[0]) + 1
            if sleep_time > 0:
                print(f"Rate limit (TPM): sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                self.request_times = []
                self.token_counts = []

        # Record this request
        self.request_times.append(current_time)
        self.token_counts.append(estimated_tokens)

    def _build_absa_prompt(self, review_text: str, product_code: int, rating: int, name: str = "", category_1: str = "", category_2: str = "") -> str:
        """Build prompt for ABSA labeling (다이소 뷰티 특화 v3)"""
        # 제품 정보 구성 (category_2만 사용)
        if name:
            if category_2:
                product_info = f"{name} (카테고리: {category_2})"
            else:
                product_info = name
        else:
            product_info = f"제품코드 {product_code}"

        prompt = f"""당신은 한국어 화장품 리뷰를 분석하는 '다이소 뷰티 특화' ABSA 전문가입니다.
초저가 화장품 시장과 고객의 다이소 소비 패턴(가성비, 듀프, 품절 대란 등)의 문맥을 파악해야 합니다.

**[리뷰 정보]**
- 제품: {product_info}
- 평점: {rating}/5
- 리뷰: "{review_text}"

**[핵심 분석 규칙]**
1. **확실한 Aspect만 추출**: confidence ≥ 0.7인 Aspect만 추출하세요. 억지로 추출하지 마세요.
2. **미분류 처리**: 확실한 Aspect가 없으면 "미분류" Aspect + neutral sentiment로 반환
3. **혼합 감성 분리**: "내용물은 좋은데 용기가 샌다"처럼 역접(근데, 하지만)이 있으면 각 Aspect를 분리 추출
4. **별점-텍스트 불일치**: "구하기 힘들어서 짜증 (5점)"은 품절 대란으로 인한 아쉬움 → 재구매 positive
5. **재구매 시그널**: "쟁여둠, 또 삼, 정착, 보이면 무조건" → 재구매 positive (confidence ≥ 0.9)
6. **미사용 리뷰**: 사용 전/기대감만 있으면 성능/품질/재질 Aspect 추출 금지

**[⚠️ 평점(Rating)을 활용한 Sentiment 판단]**
- 평점은 sentiment 판단의 **보조 지표**로 활용하세요 (텍스트 내용이 최우선)
- **평점 기준**:
  - 1~2점: negative 가능성 높음
  - 3점: neutral 가능성 높음
  - 4~5점: positive 가능성 높음
- **적용 원칙**:
  - 텍스트가 모호하거나 짧을 때 평점을 참고하여 sentiment 결정
  - 예: "그냥 그래요" (3점) → neutral / "그냥 그래요" (1점) → negative
  - 예: "괜찮아요" (5점) → positive / "괜찮아요" (2점) → neutral~negative
- **주의**: 텍스트가 명확한 감성을 표현하면 평점과 불일치해도 텍스트 우선
  - 예: "진짜 별로에요 ㅠㅠ" (5점) → 텍스트 기준 negative (평점 오입력 가능성)

**[⚠️ 중요: Aspect 혼동 방지 규칙]**

**1. 색상/발색 vs 사용감/성능 (가장 많이 혼동됨!)**
- **색상/발색**: "발색, 색, 컬러, 톤, 쿨톤, 웜톤, 예쁜색, 이쁜색, 착색" 키워드가 있으면 무조건 색상/발색
  - O: "발색이 진해요", "색이 예뻐요", "발색이 별로", "색이 탁해요", "발색 안됨"
  - O: "광고 그렇게 이쁜거같지않은데" → 색상에 대한 불만 → 색상/발색
- **사용감/성능**: 발림성, 지속력, 커버력, 보습력, 효과 등 '사용 경험'
  - O: "발림 좋아요", "커버력 별로", "보습력 최고", "효과 있어요"
- **핵심**: "발색" ≠ "발림". 발색=색상, 발림=사용감

**2. 배송/포장 vs 품질/퀄리티 (배송 중 문제 구분)**
- **배송/포장**: "왔어요, 도착, 받았는데, 배송" + 파손/누락/오배송
  - O: "부러져서 왔어요" → 배송 중 파손 → 배송/포장
  - O: "빈 상자만 왔음" → 배송 누락 → 배송/포장
  - O: "매트로 주문했는데 촉촉으로 옴" → 오배송 → 배송/포장
  - O: "깨지지 않고 왔어요" → 배송 상태 만족 → 배송/포장
- **품질/퀄리티**: 제품 자체의 제조 결함 (배송과 무관)
  - O: "처음 열었는데 불량", "유통기한 임박", "정품 의심"

**3. 품질/퀄리티 vs 사용감/성능**
- **품질/퀄리티**: 물리적 결함 + "퀄리티" 키워드 포함
  - O: "퀄리티 좋아요", "퀄리티 별로", "퀄이 좋다" → 품질/퀄리티
  - O: "불량", "깨짐", "변질", "유통기한" → 품질/퀄리티
- **사용감/성능**: 사용 후 느끼는 경험과 효과
  - O: "좋아요", "별로에요", "만족", "효과 없음" (퀄리티 언급 없이) → 사용감/성능

**4. 재구매 (다른 이슈와 함께 나올 때)**
- **재구매 의사가 명확하면 반드시 재구매 Aspect 포함**
  - O: "또 살거예요", "재구매할게요", "쟁여둠", "정착", "계속 쓸" → 재구매 추출 필수
  - O: "전에 써보고 좋아서 재구매" → 재구매 positive (불량 이슈는 별도 Aspect로)
  - O: "계속 쓰던 제품" → 재구매 neutral/positive

**[카테고리별 판단 규칙]**

[스킨케어]
- 리들샷/스피큘: "따갑다, 찌릿" → 사용감/성능 neutral (정상 반응)
- 기초케어: "순하다, 자극없다" → 사용감/성능 positive / "트러블, 좁쌀" → 사용감/성능 negative
- 보습크림: "끈적임" → 재질/냄새 neutral / "촉촉" → 사용감/성능 positive

[메이크업]
- 립제품: "볼에 발라요" (본래 용도 실패) → 사용감/성능 negative
- 베이스: "다크닝, 잿빛" → 색상/발색 negative / "톤업" → 색상/발색 positive
- 립틴트: "기승전핑크" → 색상/발색 negative

[다이소 특화]
- 듀프 비교: "올리브영 XX랑 똑같다" → 가격/가성비 positive
- 소용량: "여행용 딱" → 용량/휴대 positive / "금방 다 씀" → 용량/휴대 negative
- 용기 결함: "펌핑 고장, 샌다, 뚜껑 안 닫혀" → 디자인 negative

[CS/응대 판단 기준]
- 직원 친절/불친절 → CS/응대
- 교환/환불 경험 → CS/응대
- 매장 재고 부족/품절 → CS/응대 (구하기 힘들다, 없어서 못 샀다)

[재질/냄새 판단 기준]
- 제품의 물리적 텍스처: 묽다, 되다, 크리미, 젤타입 → 재질/냄새
- 향/냄새 관련: 향이 좋다, 냄새가 이상하다 → 재질/냄새
- 끈적임: 토너/에센스의 끈적임 → 재질/냄새 / 크림류는 사용감/성능과 함께 판단

**[Aspect 목록 (11개)]**
1. 배송/포장: 배송 속도, 택배 포장, 배송 중 파손
2. 품질/퀄리티: 제품 물리적 결함만 (깨짐, 부러짐, 변질, 유통기한, 불량), "퀄리티" 키워드
3. 가격/가성비: 가격 적정성, 가성비, 듀프 비교
4. 사용감/성능: 사용 경험, 효과, 만족도 (좋아요/별로/만족/최고 포함)
5. 용량/휴대: 용량 크기, 휴대 편의성
6. 디자인: 패키징 외관, 용기 구조/결함 (펌프, 뚜껑)
7. 재질/냄새: 텍스처, 질감, 향기/냄새
8. CS/응대: 교환/환불, 직원 응대, 매장 재고/품절
9. 재구매: 재구매 의사, 추천 (쟁여둠, 정착, 또 살)
10. 색상/발색: 색상 구현, 발색, 피부톤 적합성 (발색, 색, 컬러 키워드)
11. 미분류: 위 10개에 해당하지 않거나 confidence < 0.7인 경우 → sentiment는 neutral

**[출력 형식]**
{{
  "sentiment": "positive|neutral|negative",
  "sentiment_score": -1.0 ~ 1.0,
  "aspect_labels": [
    {{"aspect": "Aspect명", "sentiment": "positive|neutral|negative", "confidence": 0.0~1.0, "reason": "근거 문장"}}
  ],
  "evidence": "핵심 근거 원문 인용",
  "summary": "30자 이내 요약"
}}

반드시 유효한 JSON만 반환하세요. 추가 설명 없이 순수 JSON만 출력하세요."""

        return prompt

    def _validate_and_fix_result(self, result: Dict) -> Dict:
        """Validate and fix the result (검수 결과 반영 v2)"""
        # Check required fields
        required_fields = ["sentiment", "sentiment_score", "aspect_labels", "evidence", "summary"]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")

        # Validate sentiment
        if result["sentiment"] not in ["positive", "neutral", "negative"]:
            # Try to infer from score
            score = result["sentiment_score"]
            if score > 0.2:
                result["sentiment"] = "positive"
            elif score < -0.2:
                result["sentiment"] = "negative"
            else:
                result["sentiment"] = "neutral"

        # Validate sentiment_score
        result["sentiment_score"] = max(-1.0, min(1.0, float(result["sentiment_score"])))

        # Validate aspect_labels (새로운 형식: [{aspect, sentiment, confidence, reason}])
        if not isinstance(result["aspect_labels"], list):
            result["aspect_labels"] = []

        validated_aspects = []
        for item in result["aspect_labels"]:
            if isinstance(item, dict):
                # 새 형식: {aspect, sentiment, confidence, reason}
                aspect = item.get("aspect", "")
                confidence = max(0.0, min(1.0, float(item.get("confidence", 0.8))))

                # confidence < 0.7이면 제외 (미분류는 예외)
                if confidence < 0.7 and aspect != "미분류":
                    continue

                if aspect in self.ASPECT_LABELS:
                    validated_item = {
                        "aspect": aspect,
                        "sentiment": item.get("sentiment", "neutral"),
                        "confidence": confidence,
                        "reason": item.get("reason", "")
                    }
                    if validated_item["sentiment"] not in ["positive", "neutral", "negative"]:
                        validated_item["sentiment"] = "neutral"
                    validated_aspects.append(validated_item)
            elif isinstance(item, str):
                # 이전 형식 호환: 문자열만 있는 경우
                if item in self.ASPECT_LABELS:
                    validated_aspects.append({
                        "aspect": item,
                        "sentiment": result["sentiment"],
                        "confidence": 0.8,
                        "reason": ""
                    })

        # 확실한 Aspect가 없으면 미분류 추가
        if not validated_aspects:
            validated_aspects.append({
                "aspect": "미분류",
                "sentiment": "neutral",
                "confidence": 1.0,
                "reason": "확실한 Aspect 없음"
            })

        result["aspect_labels"] = validated_aspects

        # Validate evidence and summary
        if not isinstance(result["evidence"], str) or not result["evidence"].strip():
            result["evidence"] = "N/A"
        if not isinstance(result["summary"], str) or not result["summary"].strip():
            result["summary"] = "N/A"

        return result

    def label_review(
        self,
        review_text: str,
        product_code: int,
        rating: int,
        name: str = "",
        category_1: str = "",
        category_2: str = "",
        model: str = "gpt-4o-mini",
        use_cache: bool = True
    ) -> LabelingResult:
        """
        Label a review with sentiment and aspects.

        Args:
            review_text: Review text to label
            product_code: Product code
            rating: Rating (1-5)
            name: Product name (for context-aware analysis)
            category_1: Primary category (e.g., 스킨케어)
            category_2: Secondary category (e.g., 에센스/세럼)
            model: OpenAI model to use
            use_cache: Whether to use cache

        Returns:
            LabelingResult with sentiment, aspects, and metadata
        """
        # Check cache
        cache_key = self._get_cache_key(review_text, product_code, rating, model, name, category_1, category_2)
        if use_cache and cache_key in self.cache:
            cached = self.cache[cache_key]
            return LabelingResult(**cached)

        # Build prompt
        prompt = self._build_absa_prompt(review_text, product_code, rating, name, category_1, category_2)

        # Estimate tokens (rough estimate)
        estimated_tokens = len(prompt) // 2 + 200

        # Rate limiting
        self._wait_for_rate_limit(estimated_tokens)

        # Call API with retry logic
        max_retries = 3
        backoff = 2.0

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=500,
                    response_format={"type": "json_object"}
                )

                # Parse response
                result_text = response.choices[0].message.content
                result = json.loads(result_text)
                result = self._validate_and_fix_result(result)

                # Calculate cost
                tokens_input = response.usage.prompt_tokens
                tokens_output = response.usage.completion_tokens
                cost = (
                    tokens_input * self.PRICING[model]["input"] / 1000 +
                    tokens_output * self.PRICING[model]["output"] / 1000
                )

                # Create result object
                labeling_result = LabelingResult(
                    sentiment=result["sentiment"],
                    sentiment_score=result["sentiment_score"],
                    aspect_labels=result["aspect_labels"],
                    evidence=result["evidence"],
                    summary=result["summary"],
                    model=model,
                    tokens_input=tokens_input,
                    tokens_output=tokens_output,
                    cost=cost
                )

                # Update stats
                self.total_cost += cost
                self.total_requests += 1

                # Cache result
                if use_cache:
                    self.cache[cache_key] = asdict(labeling_result)
                    if self.total_requests % 10 == 0:  # Save every 10 requests
                        self._save_cache()

                return labeling_result

            except RateLimitError as e:
                if attempt < max_retries - 1:
                    sleep_time = backoff ** (attempt + 1)
                    print(f"Rate limit hit, retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    raise
            except (APIError, APIConnectionError, json.JSONDecodeError) as e:
                if attempt < max_retries - 1:
                    sleep_time = backoff ** attempt
                    print(f"API error ({e}), retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    raise

        raise RuntimeError("Max retries exceeded")

    def get_total_cost(self) -> float:
        """Get total cost so far"""
        return self.total_cost

    def get_total_requests(self) -> int:
        """Get total number of requests"""
        return self.total_requests

    def save_cache(self):
        """Manually save cache"""
        self._save_cache()

    def judge_review(
        self,
        text: str,
        rating: int,
        original_label: Dict,
        risk_reasons: List[str],
        model: str = "gpt-4.1-mini",
        use_cache: bool = True
    ) -> Dict:
        """
        Judge a labeling result for validation.

        Args:
            text: Original review text
            rating: Rating (1-5)
            original_label: Original labeling result
            risk_reasons: List of risk reasons from sampling
            model: OpenAI model to use for judging
            use_cache: Whether to use cache

        Returns:
            Dict with judgment, issues, fixed_label, confidence, reason
        """
        # Build judge prompt (검수 결과 반영 v2)
        system_prompt = """당신은 ABSA(Aspect-Based Sentiment Analysis) 검수 전문가입니다.
반드시 JSON 형식으로만 응답하세요.

[규칙]
1. 기존 라벨을 기준으로 오류만 판단하세요
2. 새로 라벨링하지 마세요
3. 확실한 오류만 수정하세요
4. 사용 전 리뷰는 성능/품질 aspect 제외
5. 배송 중 파손은 "배송/포장"으로 분류 (품질/불량 아님)

[Aspect 목록 - 10개]
배송/포장, 품질/불량, 가격/가성비, 사용감/성능, 용량/휴대, 디자인, 재질/냄새, CS/응대, 재구매, 색상/발색

[주요 오류 패턴]
- Aspect 혼동: 품질/불량 ↔ 사용감/성능 (품질은 결함/하자, 성능은 사용 경험)
- Sentiment 오류: 문맥상 부정인데 긍정으로 판단
- 불필요 추출: "재구매" 키워드만으로 재구매 aspect 추출 (실제 의사 표현이 아닐 때)
- 누락: 명확히 언급된 aspect를 놓침

[Sentiment]
positive, neutral, negative

[출력 형식 - JSON]
{
  "judgment": "ok" | "fix" | "uncertain",
  "issues": ["issue_type1", ...],
  "fixed_label": {...},
  "confidence": 0.0~1.0,
  "reason": "판단 근거"
}

judgment가 "ok"면 issues와 fixed_label은 빈 배열/객체로.
judgment가 "fix"면 issues와 fixed_label 필수.
judgment가 "uncertain"면 issues에 "ambiguous" 포함."""

        user_prompt = f"""
[원문]
{text}

[평점]
{rating}점/5점

[기존 라벨]
- sentiment: {original_label.get('sentiment', 'N/A')}
- sentiment_score: {original_label.get('sentiment_score', 'N/A')}
- aspect_labels: {original_label.get('aspect_labels', [])}
- evidence: {original_label.get('evidence', 'N/A')}

[검수 이유]
{', '.join(risk_reasons)}

위 라벨이 올바른지 검수해주세요."""

        # Check cache
        cache_content = f"judge|{text}|{rating}|{json.dumps(original_label, ensure_ascii=False)}|{model}"
        cache_key = hashlib.md5(cache_content.encode('utf-8')).hexdigest()
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]

        # Estimate tokens
        estimated_tokens = (len(system_prompt) + len(user_prompt)) // 2 + 300

        # Rate limiting
        self._wait_for_rate_limit(estimated_tokens)

        # Call API with retry logic
        max_retries = 3
        backoff = 2.0

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=800,
                    response_format={"type": "json_object"}
                )

                # Parse response
                result_text = response.choices[0].message.content
                result = json.loads(result_text)

                # Validate result
                result = self._validate_judge_result(result)

                # Calculate cost
                tokens_input = response.usage.prompt_tokens
                tokens_output = response.usage.completion_tokens
                cost = (
                    tokens_input * self.PRICING[model]["input"] / 1000 +
                    tokens_output * self.PRICING[model]["output"] / 1000
                )

                # Add metadata
                result["model"] = model
                result["tokens"] = tokens_input + tokens_output
                result["cost"] = cost

                # Update stats
                self.total_cost += cost
                self.total_requests += 1

                # Cache result
                if use_cache:
                    self.cache[cache_key] = result
                    if self.total_requests % 10 == 0:
                        self._save_cache()

                return result

            except RateLimitError as e:
                if attempt < max_retries - 1:
                    sleep_time = backoff ** (attempt + 1)
                    print(f"Rate limit hit, retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    raise
            except (APIError, APIConnectionError, json.JSONDecodeError) as e:
                if attempt < max_retries - 1:
                    sleep_time = backoff ** attempt
                    print(f"API error ({e}), retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    raise

        raise RuntimeError("Max retries exceeded")

    def _validate_judge_result(self, result: Dict) -> Dict:
        """Validate and fix judge result"""
        # Check judgment field
        if result.get("judgment") not in ["ok", "fix", "uncertain"]:
            result["judgment"] = "uncertain"

        # Ensure issues is a list
        if not isinstance(result.get("issues"), list):
            result["issues"] = []

        # Ensure fixed_label is a dict
        if not isinstance(result.get("fixed_label"), dict):
            result["fixed_label"] = {}

        # Ensure confidence is a float
        try:
            result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
        except (ValueError, TypeError):
            result["confidence"] = 0.5

        # Ensure reason is a string
        if not isinstance(result.get("reason"), str):
            result["reason"] = ""

        # Validate fixed_label if judgment is "fix"
        if result["judgment"] == "fix" and result["fixed_label"]:
            # Validate sentiment
            if result["fixed_label"].get("sentiment") not in ["positive", "neutral", "negative"]:
                if "sentiment" in result["fixed_label"]:
                    del result["fixed_label"]["sentiment"]

            # Validate aspect_labels
            if "aspect_labels" in result["fixed_label"]:
                valid_aspects = [a for a in result["fixed_label"]["aspect_labels"]
                               if a in self.ASPECT_LABELS]
                result["fixed_label"]["aspect_labels"] = valid_aspects

        return result
