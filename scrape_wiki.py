import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
import urllib3

# SSL 경고 무시
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 1. 핵심 문서 URL 리스트 (52개) ---
# 탐색 과정 없이 바로 접속할 '알짜배기' 주소들입니다.
target_urls = [
    # 주인공 및 핵심 인물
    "https://cyberpunk.fandom.com/wiki/V_(character)",
    "https://cyberpunk.fandom.com/wiki/Johnny_Silverhand",
    "https://cyberpunk.fandom.com/wiki/Jackie_Welles",
    "https://cyberpunk.fandom.com/wiki/Judy_Alvarez",
    "https://cyberpunk.fandom.com/wiki/Panam_Palmer",
    "https://cyberpunk.fandom.com/wiki/Goro_Takemura",
    "https://cyberpunk.fandom.com/wiki/Adam_Smasher",
    "https://cyberpunk.fandom.com/wiki/Alt_Cunningham",
    "https://cyberpunk.fandom.com/wiki/Rogue_Amendiares",
    "https://cyberpunk.fandom.com/wiki/Kerry_Eurodyne",
    "https://cyberpunk.fandom.com/wiki/River_Ward",
    "https://cyberpunk.fandom.com/wiki/Evelyn_Parker",
    "https://cyberpunk.fandom.com/wiki/Dexter_DeShawn",
    "https://cyberpunk.fandom.com/wiki/Solomon_Reed",
    "https://cyberpunk.fandom.com/wiki/Song_So_Mi",
    "https://cyberpunk.fandom.com/wiki/Rosalind_Myers",
    "https://cyberpunk.fandom.com/wiki/Kurt_Hansen",
    
    # 아라사카 가문
    "https://cyberpunk.fandom.com/wiki/Saburo_Arasaka",
    "https://cyberpunk.fandom.com/wiki/Yorinobu_Arasaka",
    "https://cyberpunk.fandom.com/wiki/Hanako_Arasaka",

    # 주요 기업 (Corporations)
    "https://cyberpunk.fandom.com/wiki/Arasaka",
    "https://cyberpunk.fandom.com/wiki/Militech",
    "https://cyberpunk.fandom.com/wiki/Kang_Tao",
    "https://cyberpunk.fandom.com/wiki/Biotechnica",
    "https://cyberpunk.fandom.com/wiki/Trauma_Team_International",
    "https://cyberpunk.fandom.com/wiki/Zetatech",
    "https://cyberpunk.fandom.com/wiki/Night_Corp",

    # 갱단 및 세력 (Gangs & Factions)
    "https://cyberpunk.fandom.com/wiki/Maelstrom",
    "https://cyberpunk.fandom.com/wiki/Valentinos",
    "https://cyberpunk.fandom.com/wiki/Voodoo_Boys",
    "https://cyberpunk.fandom.com/wiki/Animals_(Gang)",
    "https://cyberpunk.fandom.com/wiki/Tyger_Claws",
    "https://cyberpunk.fandom.com/wiki/6th_Street",
    "https://cyberpunk.fandom.com/wiki/The_Mox",
    "https://cyberpunk.fandom.com/wiki/Scavengers",
    "https://cyberpunk.fandom.com/wiki/Wraiths",
    "https://cyberpunk.fandom.com/wiki/Aldecaldos",
    "https://cyberpunk.fandom.com/wiki/Barghest",

    # 주요 지역 (Locations)
    "https://cyberpunk.fandom.com/wiki/Night_City",
    "https://cyberpunk.fandom.com/wiki/Watson",
    "https://cyberpunk.fandom.com/wiki/Westbrook",
    "https://cyberpunk.fandom.com/wiki/City_Center",
    "https://cyberpunk.fandom.com/wiki/Heywood",
    "https://cyberpunk.fandom.com/wiki/Santo_Domingo",
    "https://cyberpunk.fandom.com/wiki/Pacifica",
    "https://cyberpunk.fandom.com/wiki/Dogtown",
    "https://cyberpunk.fandom.com/wiki/Afterlife",
    "https://cyberpunk.fandom.com/wiki/Konpeki_Plaza",

    # 핵심 설정 (Lore & Tech)
    "https://cyberpunk.fandom.com/wiki/Cyberware",
    "https://cyberpunk.fandom.com/wiki/Cyberpsychosis",
    "https://cyberpunk.fandom.com/wiki/Netrunner",
    "https://cyberpunk.fandom.com/wiki/Braindance",
    "https://cyberpunk.fandom.com/wiki/Blackwall",
    "https://cyberpunk.fandom.com/wiki/Relic"
]

output_file = "cyberpunk_lore.txt"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

print(f"🚀 총 {len(target_urls)}개의 핵심 문서 수집을 시작합니다...")

with open(output_file, "w", encoding="utf-8") as f:
    # 실패한 URL을 기록할 리스트
    failed_urls = []
    
    for i, url in enumerate(tqdm(target_urls, desc="진행 중")):
        try:
            # verify=False로 SSL 에러 우회
            response = requests.get(url, headers=headers, verify=False, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                
                # 제목 추출
                title_tag = soup.find("h1", {"id": "firstHeading"})
                title = title_tag.text.strip() if title_tag else "제목 없음"
                
                # 본문 추출 (잡다한 요소 제거)
                content_div = soup.find("div", {"class": "mw-parser-output"})
                if content_div:
                    for garbage in content_div.find_all(["div", "aside", "table", "figure"], class_=["toc", "infobox", "rail-module", "thumb"]):
                        garbage.decompose()
                        
                    paragraphs = content_div.find_all("p", recursive=False)
                    full_text = ""
                    for p in paragraphs:
                        text = p.get_text(strip=True)
                        if text:
                            full_text += text + "\n"
                    
                    if full_text:
                        f.write(f"[문서 제목: {title}]\n")
                        f.write(f"출처: {url}\n")
                        f.write(full_text)
                        f.write("\n----\n\n")
            else:
                failed_urls.append(url)
                
            time.sleep(0.5) # 서버 부하 방지

        except Exception as e:
            print(f"\n⚠️ 오류 발생 ({url}): {e}")
            failed_urls.append(url)

print(f"\n🎉 수집 완료! '{output_file}' 파일이 생성되었습니다.")
if failed_urls:
    print(f"⚠️ {len(failed_urls)}개의 문서는 수집에 실패했습니다.")