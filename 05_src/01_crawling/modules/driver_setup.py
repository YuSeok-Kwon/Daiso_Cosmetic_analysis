"""
Selenium 드라이버 설정
"""
import undetected_chromedriver as uc
from selenium.webdriver.chrome.options import Options
from config import CRAWLING_CONFIG, USER_AGENTS
import random

def create_driver(headless=None):
    """
    봇 탐지 회피가 적용된 Chrome 드라이버 생성

    Args:
        headless: 헤드리스 모드 여부 (None일 경우 config 사용)

    Returns:
        undetected_chromedriver 인스턴스
    """

    if headless is None:
        headless = CRAWLING_CONFIG['headless']

    # Chrome 옵션 설정
    options = uc.ChromeOptions()

    # 헤드리스 모드 (필요시)
    if headless:
        options.add_argument('--headless=new')

    # 기본 옵션
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')

    # User-Agent 랜덤 설정
    user_agent = random.choice(USER_AGENTS)
    options.add_argument(f'user-agent={user_agent}')

    # 언어 설정
    options.add_argument('--lang=ko-KR')

    # 창 크기
    options.add_argument('--window-size=1920,1080')

    # 자동화 탐지 방지 (주석 처리 - 호환성 문제)
    # options.add_experimental_option("excludeSwitches", ["enable-automation"])
    # options.add_experimental_option('useAutomationExtension', False)

    # 알림 비활성화
    prefs = {
        "profile.default_content_setting_values.notifications": 2,
        "profile.default_content_settings.popups": 0,
    }
    options.add_experimental_option("prefs", prefs)

    try:
        # undetected-chromedriver 생성
        driver = uc.Chrome(
            options=options,
            version_main=144,  # Chrome 버전 명시
            driver_executable_path=None,  # 자동으로 다운로드
        )

        # 타임아웃 설정
        driver.set_page_load_timeout(CRAWLING_CONFIG['page_load_timeout'])
        driver.implicitly_wait(CRAWLING_CONFIG['implicit_wait'])

        # WebDriver 속성 숨기기 (추가 보호)
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {
            "userAgent": user_agent
        })
        driver.execute_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )

        return driver

    except Exception as e:
        print(f"드라이버 생성 실패: {e}")
        raise

def quit_driver(driver):
    """드라이버 안전하게 종료"""
    if driver:
        try:
            driver.quit()
        except:
            pass
