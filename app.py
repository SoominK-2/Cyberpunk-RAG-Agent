# --- 1. SQLite 패치 ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import os
import random
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 2. 페이지 설정 ---
st.set_page_config(
    page_title="NIGHT CITY ARCHIVES",
    page_icon="💾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링 (사이드바 400px 고정)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&display=swap');
    .stApp { background-color: #050505; font-family: 'Rajdhani', sans-serif; }
    h1 { color: #FCEE0A !important; text-transform: uppercase; text-shadow: 2px 2px 0px #00F0FF; }
    
    [data-testid="stSidebar"] { 
        min-width: 400px !important; 
        max-width: 450px !important; 
    }

    /* 모바일 환경에서 사이드바 완전히 숨기기 */
    @media (max-width: 768px) {
        [data-testid="stSidebar"] {
            display: none !important; /* 사이드바 본체 숨김 */
        }
        [data-testid="stSidebarUser"] {
            display: none !important;
        }
        [data-testid="stSidebarControls"] {
            display: none !important; /* 햄버거 메뉴 버튼 숨김 */
        }
        section.main {
             padding-left: 1rem !important; /* 모바일에서 좌우 패딩 최소화 */
             padding-right: 1rem !important;
        }
    }   
    .stButton button { width: 100%; border: 1px solid #FCEE0A; color: #FCEE0A; background-color: #000; text-align: left; }
    .stButton button:hover { border-color: #00F0FF; color: #00F0FF; }
    .stChatMessage { background-color: #1a1a1a; border: 1px solid #333; border-radius: 0px !important; }
    div[data-testid="stChatMessage"]:nth-child(odd) { border-left: 5px solid #FCEE0A; }
    div[data-testid="stChatMessage"]:nth-child(even) { border-right: 5px solid #00F0FF; background-color: #0a0a0a; }
    .stChatInput input { background-color: #111 !important; color: #FCEE0A !important; border: 2px solid #FCEE0A !important; }
    .stSpinner > div { border-top-color: #FCEE0A !important; }
</style>
""", unsafe_allow_html=True)

# --- 3. 사이드바 ---
with st.sidebar:
    st.title("📂 넷러너 가이드")
    st.markdown("---")
    st.info("💡 **Tip:** 아래 질문을 클릭하면 자동으로 입력됩니다.")
    
    questions = {
        "👥 V와 조니의 관계?": "V와 조니 실버핸드는 서로 어떤 관계이고 어떻게 변해가?",
        "🏢 아라사카의 숨겨진 목적": "아라사카 기업이 렐릭(Relic)을 만든 진짜 목적이 뭐야?",
        "🦾 사이버사이코시스 원인": "사이버사이코시스는 왜 생기는 거고 증상은 어때?",
        "📅 이블린 파커의 행적": "이블린 파커의 스케줄과 그녀에게 무슨 일이 있었는지 알려줘",
        "🏙️ 나이트 시티 구역별 특징": "나이트 시티의 주요 구역들과 각각의 분위기를 설명해줘",
        "🎸 사무라이 밴드 멤버": "전설적인 밴드 '사무라이'의 멤버들은 누구누구야?"
    }
    
    for label, prompt in questions.items():
        if st.button(label):
            st.session_state["prompt_input"] = prompt

# --- 4. 메인 로직 ---
st.title("🔌 NIGHT CITY ARCHIVES")
st.caption("ACCESSING SECURE DATASLATE... // WELCOME, EDGERUNNER.")

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
RAG_MODEL = "gpt-4o-mini"
CHROMA_DIR = "/tmp/chroma_db"

# 헬퍼 함수: Streamlit 메시지를 간결한 대화 기록 문자열로 변환
def get_chat_history_string(messages):
    history = []
    # 초기 환영 메시지는 건너뛰고, 최대 4개의 이전 턴만 사용
    for msg in messages[-5:-1]:
        if msg["role"] == "user":
            history.append(f"User: {msg['content']}")
        elif msg["role"] == "assistant":
            history.append(f"Fixer: {msg['content']}")
    return "\n".join(history)

@st.cache_resource
def load_database():
    try:
        # 데이터 로드 및 벡터 DB 설정 (이전과 동일)
        all_docs = []
        if os.path.exists("cyberpunk_shards.txt"):
            docs1 = TextLoader("cyberpunk_shards.txt", encoding="utf-8").load()
            for d in docs1: d.metadata["source"] = "인게임 샤드"
            all_docs.extend(docs1)
        if os.path.exists("cyberpunk_lore.txt"):
            docs2 = TextLoader("cyberpunk_lore.txt", encoding="utf-8").load()
            for d in docs2: d.metadata["source"] = "위키 설정(Lore)"
            all_docs.extend(docs2)
        if not all_docs:
            return None, None, None, None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_docs)
        embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
        db = Chroma.from_documents(splits, embed_model, persist_directory=CHROMA_DIR)
        retriever = db.as_retriever(search_kwargs={"k": 25})
        llm = ChatOpenAI(model_name=RAG_MODEL, temperature=0.3)
        
        # 1. 쿼리 재구성을 위한 프롬프트 (Condensing) - 영어로 독립적인 검색어 생성
        condense_template = """
        Given the following conversation history and a new question, combine them into a single, standalone English search query for a Cyberpunk 2077 database.
        If the new question is a follow-up, use the history to clarify the intent.
        If the new question is standalone, just translate it to English.
        Output ONLY the standalone English query text.

        Chat History:
        {chat_history}

        New Question: {question}

        Standalone English Query:
        """
        condense_prompt = ChatPromptTemplate.from_template(condense_template)
        
        # 2. 쿼리 재구성 체인
        condense_chain = condense_prompt | llm | StrOutputParser()

        # 3. RAG 답변 생성용 프롬프트 (Final Answer)
        template = """
        당신은 '사이버펑크 2077' 세계관의 냉소적이고 유능한 정보 브로커(Fixer)입니다.
        
        [지시사항]
        1. **언어**: 답변은 **오직 한국어**로만 작성하며, 다른 언어(영어, 일본어 등)를 절대 섞어 쓰지 마시오.
        2. 말투: "~입니다/습니다" 절대 금지. "~야", "~군", "~하더군", "~일걸" 같은 반말 사용.
        3. 태도: 너무 딱딱하게 설명하지 말고, 의뢰인에게 정보를 브리핑하듯 자연스럽게 이야기해.
        4. 고유명사 (한국어 공식 번역 준수): 
           - Panam -> **팬앰**
           - Hanako -> **하나코**
           - Yorinobu -> 요리노부
           - Saburo -> 사부로
           - Relic -> 렐릭
           - Evelyn -> **이블린**
           - Arasaka -> 아라사카
           - Militech -> 밀리테크
           - Johnny -> 조니
           - V -> V(브이)
           기타 인물명, 지명 등은 최대한 한국어 표기를 사용해.
        5. 근거: 반드시 아래 제공된 Context(정보)들을 종합해서 답해. 
        6. 모름: 정보가 없으면 "내 정보망엔 없는 건인데. 다른 걸 물어봐."라고 짧게 끊어. 헛소리 금지.

        [대화 기록 (참고용)]
        {chat_history}
        
        Context (검색된 정보):
        {context}
        
        Question (사용자의 원래 질문):
        {question}
        
        Answer (정보 브로커 스타일):
        """
        final_rag_prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            # 검색된 문서들의 내용을 하나의 문자열로 합칩니다.
            return "\n\n".join(doc.page_content for doc in docs)

        # 4. 최종 RAG 체인 (검색된 Context와 Chat History, Original Question을 모두 사용)
        rag_chain = (
            RunnablePassthrough.assign(context=(lambda x: x["standalone_query"]) | retriever | format_docs)
            | final_rag_prompt
            | llm
            | StrOutputParser()
        )
        
        return rag_chain, condense_chain, retriever, llm

    except Exception as e:
        st.error(f"시스템 오류: {e}")
        return None, None, None, None

rag_chain, condense_chain, retriever, llm = load_database()

# --- 5. 채팅 인터페이스 ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "원하는 정보를 말해봐. 가격은... 나중에 청구하지."}]

# 1. 대화 기록 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("🔍 데이터 출처 확인"):
                for src in msg["sources"]:
                    st.caption(src)

# 2. 사용자 입력 처리
if user_input := st.chat_input("데이터 검색...") or st.session_state.get("prompt_input"):
    if st.session_state.get("prompt_input"):
        del st.session_state["prompt_input"]

    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if rag_chain and condense_chain:
            status_placeholder = st.empty()
            
            try:
                # ⭐️ 멀티 턴 핵심 로직: 대화 기록 + 현재 질문으로 독립적인 검색어 생성 ⭐️
                chat_history_str = get_chat_history_string(st.session_state.messages)

                loading_texts = [
                    "📡 암호 해독 중...", "💾 데이터뱅크 접속...", 
                    "⚡ 넷러닝 프로토콜 시작...", "🔍 샤드 데이터 스캔 중...", "🕶️ 정보망 가동..."
                ]
                
                with status_placeholder.status(random.choice(loading_texts), expanded=True) as status:
                    
                    # 1. 독립적인 검색 쿼리 생성 (영어)
                    status.write("이전 대화 맥락을 기반으로 검색어 재구성 중...")
                    standalone_query = condense_chain.invoke({
                        "chat_history": chat_history_str,
                        "question": user_input
                    }).strip()
                    
                    status.write(f"최종 검색 쿼리: **{standalone_query}**")
                    status.write("데이터베이스 검색 및 답변 생성 중...")
                    
                    # 2. RAG 실행 (Standalone Query를 검색에 사용)
                    result = rag_chain.invoke({
                        "standalone_query": standalone_query, # Context 검색에 사용됨
                        "question": user_input, # 최종 답변 생성 프롬프트에 사용됨
                        "chat_history": chat_history_str # 최종 답변 생성 프롬프트에 사용됨
                    })
                    
                    # 3. 검색된 출처 문서 추출 (Standalone Query로 검색)
                    source_docs = retriever.invoke(standalone_query)
                    unique_sources = []
                    for doc in source_docs:
                        clean_content = doc.page_content.replace("\n", " ").replace("\r", " ")
                        src_text = f"[{doc.metadata.get('source', 'Unknown')}] {clean_content[:50]}..."
                        if src_text not in unique_sources:
                            unique_sources.append(src_text)
                    
                    status.update(label="✅ 데이터 확보 완료", state="complete", expanded=False)

                st.markdown(result)
                
                if unique_sources:
                    with st.expander("🔍 데이터 출처 확인"):
                        for src in unique_sources:
                            st.caption(src)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": result, 
                    "sources": unique_sources
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"처리 중 오류 발생: {e}")
        else:
            st.error("시스템 오프라인.")