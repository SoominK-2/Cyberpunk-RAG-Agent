import streamlit as st
import os
import sys
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. 페이지 설정 (가장 먼저 실행되어야 함) ---
st.set_page_config(
    page_title="Cyberpunk 2077 Wiki AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. 환경 변수 및 초기 설정 ---
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
RAG_MODEL = "gpt-4o-mini"
DATA_FILE = "cyberpunk_shards.txt"
CHROMA_DIR = "./cyberpunk_chroma_db"

# --- 3. 커스텀 CSS (사이버펑크 테마 디자인) ---
st.markdown("""
<style>
    /* 전체 배경 및 폰트 */
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&display=swap');
    
    .stApp {
        background-color: #050505;
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* 헤더 스타일 */
    h1 {
        color: #FCEE0A !important;
        text-transform: uppercase;
        text-shadow: 2px 2px 0px #00F0FF;
        font-weight: 800 !important;
        letter-spacing: 2px;
    }
    
    /* 캡션 스타일 */
    .stCaption {
        color: #00F0FF !important;
        font-size: 1.1em !important;
        border-left: 3px solid #FCEE0A;
        padding-left: 10px;
    }

    /* 채팅 메시지 컨테이너 */
    .stChatMessage {
        background-color: #1a1a1a;
        border: 1px solid #333;
        border-radius: 0px !important; /* 각진 테두리 */
        margin-bottom: 10px;
    }

    /* 유저 메시지 (오른쪽 정렬 느낌) */
    div[data-testid="stChatMessage"]:nth-child(odd) {
        border-left: 5px solid #FCEE0A;
    }

    /* AI 메시지 (왼쪽 정렬 느낌) */
    div[data-testid="stChatMessage"]:nth-child(even) {
        border-right: 5px solid #00F0FF;
        background-color: #0a0a0a;
    }

    /* 입력창 스타일 */
    .stChatInput input {
        background-color: #111 !important;
        color: #FCEE0A !important;
        border: 2px solid #FCEE0A !important;
        border-radius: 0px !important;
    }
    
    /* 로딩 스피너 색상 */
    .stSpinner > div {
        border-top-color: #FCEE0A !important;
    }
    
    /* 하단 Streamlit 마크 숨기기 (선택사항) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

# 헤더 표시
st.title("🔌 NIGHT CITY ARCHIVES")
st.caption("ACCESSING SECURE DATASLATE... // WELCOME, EDGERUNNER.")

# --- 4. 데이터 로드 및 RAG 체인 구축 (캐시 처리) ---
@st.cache_resource
def load_database():
    try:
        loader = TextLoader(DATA_FILE, encoding="utf-8")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
        db = Chroma.from_documents(
            documents=docs, 
            embedding=embed_model, 
            persist_directory=CHROMA_DIR
        )
        retriever = db.as_retriever()
        
        llm = ChatOpenAI(model_name=RAG_MODEL)
        template = """
        당신은 '사이버펑크 2077' 세계관의 정통한 정보 브로커(Fixer)입니다.
        말투는 냉소적이지만 정보는 정확하게 전달하세요. (예: "~라고 하더군.", "~야.")
        제공된 데이터(Context)에 있는 내용만 답하고, 모르는 내용은 "그건 내 정보망에 없는 내용이야."라고 딱 잘라 말하세요.
        
        Context:
        {context}
        
        Question:
        {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain

    except Exception as e:
        st.error(f"⚠️ CRITICAL ERROR: DATABASE CORRUPTED. {e}")
        return None

rag_chain = load_database()

# --- 5. 채팅 UI 및 Multi-Turn 구현 ---
if rag_chain:
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # 초기 환영 메시지 추가
        st.session_state.messages.append({"role": "assistant", "content": "원하는 정보를 말해봐. 가격은... 나중에 청구하지."})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt_text := st.chat_input("데이터를 검색할 키워드 입력..."):
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        with st.chat_message("user"):
            st.markdown(prompt_text)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("📡 DECRYPTING SHARD DATA..."):
                full_response = rag_chain.invoke(prompt_text)
                message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})