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
        "gpt-4.1": {"input": 0.002, "output": 0.008}
    }

    # Fixed aspect labels
    ASPECT_LABELS = [
        "배송/포장",
        "품질/불량",
        "가격/가성비",
        "사용감/성능",
        "사이즈/호환",
        "디자인",
        "재질/냄새",
        "CS/응대",
        "재구매"
    ]

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

    def _get_cache_key(self, review_text: str, product_code: int, rating: int, model: str) -> str:
        """Generate cache key"""
        content = f"{review_text}|{product_code}|{rating}|{model}"
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

    def _build_absa_prompt(self, review_text: str, product_code: int, rating: int) -> str:
        """Build prompt for ABSA labeling"""
        aspect_list = "\n".join([f"{i+1}. {label}" for i, label in enumerate(self.ASPECT_LABELS)])

        prompt = f"""당신은 한국어 쇼핑몰 리뷰를 분석하는 감성 분석 전문가입니다.

다음 리뷰를 분석하여 JSON 형식으로 결과를 반환하세요.

**리뷰 정보:**
- 제품 코드: {product_code}
- 평점: {rating}/5
- 리뷰 내용: "{review_text}"

**분석 항목:**

1. **sentiment**: 감성 분류 (positive/neutral/negative)
   - 리뷰 텍스트 내용을 기반으로 판단 (평점은 참고만)
   - positive: 긍정적 표현이 지배적
   - neutral: 중립적이거나 객관적 사실만 나열
   - negative: 부정적 표현이 지배적

2. **sentiment_score**: 감성 점수 (-1.0 ~ 1.0)
   - -1.0: 매우 부정적
   - 0.0: 중립
   - 1.0: 매우 긍정적

3. **aspect_labels**: 해당하는 모든 측면 라벨 (배열)
   다음 9개 라벨 중 리뷰에서 언급된 모든 항목을 선택:
{aspect_list}

4. **evidence**: 판단 근거 문장 (리뷰에서 발췌)
   - 가장 핵심적인 문장 1-2개
   - 원문 그대로 인용

5. **summary**: 1문장 요약
   - "X, Y에 대해 긍정적/부정적" 형식
   - 30자 이내

**출력 형식 (JSON):**
```json
{{
  "sentiment": "positive",
  "sentiment_score": 0.8,
  "aspect_labels": ["배송/포장", "품질/불량"],
  "evidence": "배송이 빠르고 품질도 좋아요",
  "summary": "배송, 품질에 대해 긍정적"
}}
```

반드시 유효한 JSON만 반환하세요. 추가 설명은 포함하지 마세요."""

        return prompt

    def _validate_and_fix_result(self, result: Dict) -> Dict:
        """Validate and fix the result"""
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

        # Validate aspect_labels
        if not isinstance(result["aspect_labels"], list):
            result["aspect_labels"] = []
        result["aspect_labels"] = [label for label in result["aspect_labels"]
                                   if label in self.ASPECT_LABELS]

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
        model: str = "gpt-4o-mini",
        use_cache: bool = True
    ) -> LabelingResult:
        """
        Label a review with sentiment and aspects.

        Args:
            review_text: Review text to label
            product_code: Product code
            rating: Rating (1-5)
            model: OpenAI model to use
            use_cache: Whether to use cache

        Returns:
            LabelingResult with sentiment, aspects, and metadata
        """
        # Check cache
        cache_key = self._get_cache_key(review_text, product_code, rating, model)
        if use_cache and cache_key in self.cache:
            cached = self.cache[cache_key]
            return LabelingResult(**cached)

        # Build prompt
        prompt = self._build_absa_prompt(review_text, product_code, rating)

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
        # Build judge prompt
        system_prompt = """당신은 ABSA(Aspect-Based Sentiment Analysis) 검수 전문가입니다.
반드시 JSON 형식으로만 응답하세요.

[규칙]
1. 기존 라벨을 기준으로 오류만 판단하세요
2. 새로 라벨링하지 마세요
3. 확실한 오류만 수정하세요

[Aspect 목록]
배송/포장, 품질/불량, 가격/가성비, 사용감/성능, 사이즈/호환, 디자인, 재질/냄새, CS/응대, 재구매

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
