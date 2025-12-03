# 12주차 과제: Cyberpunk 2077 RAG Agent

이 프로젝트는 '사이버펑크 2077'의 **인게임 샤드(Shard)** 데이터와 **위키(Lore)** 데이터를 결합하여 구축된 전문 지식 RAG 에이전트입니다. Streamlit을 사용하여 사이버펑크 테마의 UI를 구현했습니다.

## 1. 주요 기능

* **이중 데이터 소스:** 1,200개 이상의 인게임 텍스트(Shard)와 50개 이상의 핵심 설정 문서(Lore)를 통합하여 답변합니다.
* **지능형 검색 (Query Translation):** 사용자가 한국어로 질문하면, 내부적으로 영어로 번역하여 원문 데이터베이스를 정밀 검색합니다. (예: "이블린 일정" -> "Evelyn's Schedule" 검색)
* **출처 추적:** 답변 생성에 사용된 데이터가 '인게임 샤드'인지 '위키 설정'인지 출처를 명시합니다.
* **사이버펑크 UI:** 몰입감을 위한 커스텀 CSS 디자인이 적용되었습니다.

## 2. 제출 파일 구성

* `app.py`: 메인 애플리케이션 (UI, RAG 로직, 번역기 포함)
* `requirements.txt`: 필수 라이브러리 목록
* `cyberpunk_shards.txt`: 인게임 데이터 원본
* `cyberpunk_lore.txt`: 위키 데이터 원본
* `cyberpunk_chroma_db/`: 벡터 데이터베이스 폴더

## 3. 실행 방법 (Local)

1.  **라이브러리 설치:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **API 키 설정:**
    * `.streamlit/secrets.toml` 파일을 생성하여 `OPENAI_API_KEY`를 입력하거나,
    * 터미널 환경 변수로 설정해야 합니다.

3.  **실행:**
    ```bash
    streamlit run app.py
    ```