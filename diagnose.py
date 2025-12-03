import time
from selenium import webdriver
from selenium.webdriver.common.by import By

# --- 진단 대상 URL ---
# 문제가 되는 카테고리 페이지 중 하나
TARGET_URL = "https://cyberpunk.fandom.com/wiki/Category:Cyberpunk_2077_characters"

print("🔍 진단 모드 시작: 웹사이트 구조를 분석합니다...")

options = webdriver.ChromeOptions()
# 봇 탐지 회피를 위한 기본 설정
options.add_argument("--disable-blink-features=AutomationControlled") 
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

driver = webdriver.Chrome(options=options)

try:
    print(f"1. 페이지 접속 시도: {TARGET_URL}")
    driver.get(TARGET_URL)
    
    # 충분한 로딩 시간 대기 (10초)
    print("2. 페이지 로딩 대기 중 (10초)...")
    time.sleep(5)
    
    # 스크롤 시도 (하단 내용 로딩 유도)
    print("3. 스크롤 다운 시도...")
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(5)

    # --- [진단 1] 스크린샷 저장 ---
    print("4. 현재 화면 스크린샷 저장 중 (debug_screenshot.png)...")
    driver.save_screenshot("debug_screenshot.png")
    
    # --- [진단 2] HTML 소스 저장 ---
    print("5. 전체 HTML 소스코드 저장 중 (debug_source.html)...")
    with open("debug_source.html", "w", encoding="utf-8") as f:
        f.write(driver.page_source)

    # --- [진단 3] 링크 개수 파악 ---
    print("6. 페이지 내의 모든 링크(<a> 태그) 분석 중...")
    all_links = driver.find_elements(By.TAG_NAME, "a")
    
    wiki_links = [link.get_attribute('href') for link in all_links if link.get_attribute('href') and "/wiki/" in link.get_attribute('href')]
    
    print(f"   - 발견된 전체 <a> 태그 수: {len(all_links)}개")
    print(f"   - 그 중 '/wiki/'가 포함된 링크 수: {len(wiki_links)}개")
    
    print("\n[샘플 링크 10개 출력]")
    for i, link in enumerate(wiki_links[:10]):
        print(f"   {i+1}. {link}")

    print("\n✅ 진단 완료. 폴더에 생성된 'debug_screenshot.png'와 'debug_source.html'을 확인하세요.")

except Exception as e:
    print(f"❌ 진단 중 오류 발생: {e}")

finally:
    driver.quit()