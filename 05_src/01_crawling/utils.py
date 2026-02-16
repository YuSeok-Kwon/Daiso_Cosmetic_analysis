"""
크롤링 유틸리티 함수
"""
import time
import random
import os
import logging
from datetime import datetime
from config import CRAWLING_CONFIG

def setup_logger(name, log_file=None):
    """로거 설정"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 포맷 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 파일 핸들러
    if log_file:
        log_dir = CRAWLING_CONFIG['log_dir']
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, log_file),
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def random_delay(min_sec=None, max_sec=None):
    """랜덤 지연 (사람처럼 행동)"""
    if min_sec is None:
        min_sec = CRAWLING_CONFIG['min_delay']
    if max_sec is None:
        max_sec = CRAWLING_CONFIG['max_delay']

    delay = random.uniform(min_sec, max_sec)
    time.sleep(delay)

def scroll_page(driver, scroll_pause=0.5, max_scrolls=10):
    """페이지 스크롤하여 동적 콘텐츠 로딩"""
    last_height = driver.execute_script("return document.body.scrollHeight")
    scrolls = 0

    while scrolls < max_scrolls:
        # 스크롤 다운
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause)

        # 새로운 높이 계산
        new_height = driver.execute_script("return document.body.scrollHeight")

        if new_height == last_height:
            break

        last_height = new_height
        scrolls += 1

    # 맨 위로 스크롤
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(0.5)

def safe_find_element(driver, by, value, default=''):
    """안전한 요소 찾기"""
    try:
        element = driver.find_element(by, value)
        return element.text.strip()
    except:
        return default

def safe_find_attribute(driver, by, value, attribute, default=''):
    """안전한 속성 가져오기"""
    try:
        element = driver.find_element(by, value)
        return element.get_attribute(attribute)
    except:
        return default

def extract_price(price_text):
    """가격 텍스트에서 숫자 추출"""
    if not price_text:
        return None

    # 숫자와 콤마만 추출
    import re
    numbers = re.sub(r'[^\d,]', '', price_text)
    numbers = numbers.replace(',', '')

    try:
        return int(numbers)
    except:
        return None

def extract_rating(rating_text):
    """평점 텍스트에서 숫자 추출"""
    if not rating_text:
        return None

    import re
    match = re.search(r'(\d+\.?\d*)', rating_text)

    try:
        return float(match.group(1)) if match else None
    except:
        return None

def extract_review_count(review_text):
    """리뷰 수 추출"""
    if not review_text:
        return 0

    import re
    numbers = re.sub(r'[^\d,]', '', review_text)
    numbers = numbers.replace(',', '')

    try:
        return int(numbers)
    except:
        return 0

def save_to_csv(data, filename):
    """데이터를 CSV로 저장"""
    import pandas as pd

    data_dir = CRAWLING_CONFIG['data_dir']
    os.makedirs(data_dir, exist_ok=True)

    filepath = os.path.join(data_dir, filename)

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False, encoding='utf-8-sig')

    return filepath

def get_timestamp():
    """현재 타임스탬프"""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def get_date_string():
    """날짜 문자열"""
    return datetime.now().strftime('%Y%m%d')
