# 12주차 과제: Cyberpunk 2077 RAG Agent UI Implementation

이 프로젝트는 Streamlit을 사용하여 구현된 **데이터 기반 LLM RAG 에이전트**입니다.
'사이버펑크 2077' 게임 내의 샤드(Shard) 텍스트 데이터를 기반으로 사용자의 질문에 답변하며, 게임의 분위기를 살린 커스텀 UI가 적용되어 있습니다.

## 1. 제출 파일 구성

* `app.py`: 메인 애플리케이션 소스 코드 (Streamlit UI + RAG 로직)
* `requirements.txt`: 프로젝트 실행에 필요한 Python 라이브러리 목록
* `cyberpunk_shards.txt`: RAG 시스템의 기반이 되는 원본 텍스트 데이터 (1228개 샤드)
* `cyberpunk_chroma_db/`: 미리 임베딩된 벡터 데이터베이스 폴더 (실행 속도 단축을 위해 포함)

## 2. 실행 환경 설정

이 프로그램은 Python 3.10 이상 환경에서 실행하는 것을 권장합니다.

### 2-1. 라이브러리 설치
터미널에서 프로젝트 폴더로 이동한 후, 다음 명령어를 실행하여 의존성 라이브러리를 설치합니다.

pip install -r requirements.txt

### 2-2. OpenAI API 키 설정 (필수)
소스 코드(`app.py`)에는 보안을 위해 API 키가 포함되어 있지 않습니다.
로컬 환경에서 실행하려면 Streamlit의 `secrets.toml` 파일을 설정하거나 환경 변수를 사용해야 합니다.

**방법 A: .streamlit 폴더 생성 (권장)**
1. 프로젝트 루트 폴더에 `.streamlit` 이라는 이름의 폴더를 만듭니다.
2. 그 안에 `secrets.toml` 파일을 생성하고 아래 내용을 입력합니다.

[OPENAI_API_KEY]
OPENAI_API_KEY = "여기에_발급받은_SK_키를_입력하세요"

**방법 B: 코드 직접 수정 (간편)**
`app.py` 파일을 열고 20번째 줄 근처의 키 설정 부분을 다음과 같이 직접 수정하여 실행할 수도 있습니다.

# (수정 전) os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
# (수정 후) os.environ["OPENAI_API_KEY"] = "sk-..." (직접 입력)

## 3. 실행 방법

터미널에서 다음 명령어를 입력하여 웹 UI를 실행합니다.

streamlit run app.py

명령어 실행 후 자동으로 웹 브라우저가 열리며 챗봇과 대화할 수 있습니다.

## 4. 주요 기능

* **사이버펑크 테마 UI**: 게임 내 단말기 느낌을 주는 CSS 스타일링 적용
* **RAG 기반 답변**: `cyberpunk_shards.txt`에 포함된 내용만을 근거로 답변 (환각 방지 프롬프트 적용)
* **Multi-turn 대화**: 이전 대화 내용을 기억하여 문맥에 맞는 답변 제공