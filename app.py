# --- 1. SQLite 패치 (Streamlit Cloud 오류 방지) ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 2. 페이지 설정 & 사이드바 스타일 수정 ---
st.set_page_config(
    page_title="NIGHT CITY ARCHIVES",
    page_icon="💾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS (사이버펑크 테마 + 사이드바 너비 조정)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&display=swap');
    .stApp { background-color: #050505; font-family: 'Rajdhani', sans-serif; }
    
    /* 헤더 스타일 */
    h1 { color: #FCEE0A !important; text-transform: uppercase; text-shadow: 2px 2px 0px #00F0FF; }
    
    /* 사이드바 너비 확장 (줄바꿈 방지) */
    [data-testid="stSidebar"] {
        min-width: 400px !important;
        max-width: 500px !important;
    }
    
    /* 버튼 스타일 */
    .stButton button {
        width: 100%;
        border: 1px solid #FCEE0A;
        color: #FCEE0A;
        background-color: #000;
        text-align: left;
    }
    .stButton button:hover {
        border-color: #00F0FF;
        color: #00F0FF;
    }

    /* 메시지 박스 스타일 */
    .stChatMessage { background-color: #1a1a1a; border: 1px solid #333; border-radius: 0px !important; }
    div[data-testid="stChatMessage"]:nth-child(odd) { border-left: 5px solid #FCEE0A; }
    div[data-testid="stChatMessage"]:nth-child(even) { border-right: 5px solid #00F0FF; background-color: #0a0a0a; }
    .stChatInput input { background-color: #111 !important; color: #FCEE0A !important; border: 2px solid #FCEE0A !important; }
    .stSpinner > div { border-top-color: #FCEE0A !important; }
</style>
""", unsafe_allow_html=True)

# --- 3. 사이드바 (추천 질문) ---
with st.sidebar:
    st.title("📂 넷러너 가이드")
    st.markdown("---")
    st.info("💡 **Tip:** 아래 질문을 클릭하면 자동으로 입력됩니다.")
    
    # 질문 목록 (짧고 간결하게 수정하여 줄바꿈 최소화)
    questions = {
        "V와 아라사카의 관계?": "아라사카와 V의 관계에 대해 상세히 말해줘",
        "조니 실버핸드는 누구?": "조니 실버핸드의 과거와 정체에 대해 알려줘",
        "사이버사이코시스란?": "사이버사이코시스의 원인과 증상은 뭐야?",
        "나이트 시티 주요 구역": "나이트 시티의 주요 구역과 특징을 설명해줘",
        "렐릭(Relic)이란?": "렐릭(Relic)이 무엇이고 왜 중요한지 알려줘"
    }
    
    for label, prompt in questions.items():
        if st.button(label):
            st.session_state["prompt_input"] = prompt

# --- 4. 메인 로직 ---
st.title("🔌 NIGHT CITY ARCHIVES")
st.caption("ACCESSING SECURE DATASLATE... // WELCOME, EDGERUNNER.")

# 환경 변수 설정
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
RAG_MODEL = "gpt-4o-mini"
CHROMA_DIR = "./cyberpunk_chroma_db"

@st.cache_resource
def load_database():
    try:
        all_docs = []
        files_loaded = []

        # 1. 샤드 데이터 로드
        if os.path.exists("cyberpunk_shards.txt"):
            loader1 = TextLoader("cyberpunk_shards.txt", encoding="utf-8")
            docs1 = loader1.load()
            for d in docs1: d.metadata["source"] = "인게임 샤드"
            all_docs.extend(docs1)
            files_loaded.append("Shards")
        
        # 2. Lore 데이터 로드 (파일이 없으면 패스)
        if os.path.exists("cyberpunk_lore.txt"):
            loader2 = TextLoader("cyberpunk_lore.txt", encoding="utf-8")
            docs2 = loader2.load()
            for d in docs2: d.metadata["source"] = "위키 설정(Lore)"
            all_docs.extend(docs2)
            files_loaded.append("Lore")

        if not all_docs:
            st.error("❌ 데이터 파일이 없습니다. (cyberpunk_shards.txt 확인 필요)")
            return None

        # 데이터 로드 성공 메시지 (디버깅용, 나중에 주석 처리 가능)
        st.success(f"✅ 시스템 가동: {', '.join(files_loaded)} 데이터 로드 완료 ({len(all_docs)}개 문서)")

        # 3. 텍스트 분할 및 임베딩
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_docs)
        embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # 4. DB 생성
        db = Chroma.from_documents(splits, embed_model, persist_directory=CHROMA_DIR)
        retriever = db.as_retriever()
        
        # 5. LLM & Chain
        llm = ChatOpenAI(model_name=RAG_MODEL)
        
        # 수동 체인 구성 (LCEL)
        template = """
        당신은 '사이버펑크 2077' 세계관의 정통한 정보 브로커입니다.
        반드시 아래 제공된 Context(정보)만을 기반으로 답변하세요.
        
        Context:
        {context}
        
        Question:
        {question}
        
        Answer (한글로, 출처가 있다면 언급하며):
        """
        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # *중요* retriever 객체도 같이 반환해서 나중에 출처 검색에 씀
        return rag_chain, retriever

    except Exception as e:
        st.error(f"⚠️ 데이터베이스 초기화 오류:\n{e}")
        return None, None

# 로드 실행
rag_chain, retriever = load_database()

# --- 5. 채팅 인터페이스 ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "원하는 정보를 말해봐. 가격은... 나중에 청구하지."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("🔍 데이터 출처 확인"):
                for src in msg["sources"]:
                    st.caption(src)

# 입력 처리
user_input = st.chat_input("데이터 검색...") or st.session_state.get("prompt_input")

if user_input:
    # 버튼 입력값 초기화
    if "prompt_input" in st.session_state:
        del st.session_state["prompt_input"]

    # 1. 사용자 메시지 표시
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. AI 답변 생성
    with st.chat_message("assistant"):
        if rag_chain:
            with st.spinner("📡 NEURAL LINK ESTABLISHED..."):
                # 답변 생성
                response = rag_chain.invoke(user_input)
                st.markdown(response)
                
                # 출처 찾기 (Retriever 별도 호출)
                source_docs = retriever.invoke(user_input)
                unique_sources = []
                for doc in source_docs:
                    src_text = f"[{doc.metadata.get('source', 'Unknown')}] {doc.page_content[:50]}..."
                    if src_text not in unique_sources:
                        unique_sources.append(src_text)
                
                with st.expander("🔍 데이터 출처 확인"):
                    for src in unique_sources:
                        st.caption(src)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response, 
                    "sources": unique_sources
                })
        else:
            st.error("⛔ 데이터베이스 연결 실패. 관리자에게 문의하세요.")