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

# --- 2. 페이지 설정 ---
st.set_page_config(
    page_title="NIGHT CITY ARCHIVES",
    page_icon="💾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&display=swap');
    .stApp { background-color: #050505; font-family: 'Rajdhani', sans-serif; }
    h1 { color: #FCEE0A !important; text-transform: uppercase; text-shadow: 2px 2px 0px #00F0FF; }
    [data-testid="stSidebar"] { min-width: 400px !important; max-width: 500px !important; }
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
    
    # 번역 결과 표시 영역 (플레이스홀더)
    trans_debug_area = st.empty()
    
    st.info("💡 **Tip:** 아래 질문을 클릭하면 자동으로 입력됩니다.")
    
    questions = {
        "V와 아라사카의 관계?": "아라사카와 V의 관계에 대해 상세히 말해줘",
        "조니 실버핸드는 누구?": "조니 실버핸드의 과거와 정체에 대해 알려줘",
        "이블린 파커의 일정": "이블린 파커의 스케줄 샤드 내용은 뭐야?",
        "나이트 시티 주요 구역": "나이트 시티의 주요 구역과 특징을 설명해줘",
        "렐릭(Relic)이란?": "렐릭(Relic)이 무엇이고 왜 중요한지 알려줘"
    }
    
    for label, prompt in questions.items():
        if st.button(label):
            st.session_state["prompt_input"] = prompt

# --- 4. 메인 로직 ---
st.title("🔌 NIGHT CITY ARCHIVES")
st.caption("ACCESSING SECURE DATASLATE... // WELCOME, EDGERUNNER.")

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
RAG_MODEL = "gpt-4o-mini"
# 서버 권한 문제 해결을 위해 /tmp 사용
CHROMA_DIR = "/tmp/chroma_db"

@st.cache_resource
def load_database():
    try:
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
            return None, None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_docs)
        embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
        
        db = Chroma.from_documents(splits, embed_model, persist_directory=CHROMA_DIR)
        
        # ⭐️⭐️⭐️ [수정됨] 검색 범위를 20개로 대폭 늘림 ⭐️⭐️⭐️
        retriever = db.as_retriever(search_kwargs={"k": 20})
        
        llm = ChatOpenAI(model_name=RAG_MODEL)
        
        template = """
        당신은 '사이버펑크 2077' 세계관의 정통한 정보 브로커입니다.
        반드시 아래 제공된 Context(정보)만을 기반으로 답변하세요.
        Context에 없는 내용은 "내 정보망에 없는 내용이야."라고 답하세요.
        
        Context:
        {context}
        
        Question:
        {question}
        
        Answer (한국어로):
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
        
        return rag_chain, retriever

    except Exception as e:
        st.error(f"시스템 오류: {e}")
        return None, None

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

if user_input := st.chat_input("데이터 검색...") or st.session_state.get("prompt_input"):
    if st.session_state.get("prompt_input"):
        del st.session_state["prompt_input"]

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if rag_chain:
            with st.spinner("📡 TRANSLATING & SEARCHING..."):
                try:
                    # 1. 질문 번역 (한글 -> 영어)
                    llm_trans = ChatOpenAI(model_name=RAG_MODEL)
                    trans_prompt = ChatPromptTemplate.from_template(
                        "Translate the following Korean text to English for a Cyberpunk 2077 database search. Output ONLY the translated text.\nText: {text}"
                    )
                    trans_chain = trans_prompt | llm_trans | StrOutputParser()
                    english_query = trans_chain.invoke({"text": user_input}).strip()
                    
                    # ⭐️⭐️⭐️ [수정됨] 사이드바에 번역 결과 표시 ⭐️⭐️⭐️
                    trans_debug_area.success(f"🇺🇸 번역된 검색어: **{english_query}**")
                    
                    # 2. RAG 실행 (영어 질문으로 검색)
                    response = rag_chain.invoke(english_query)
                    st.markdown(response)
                    
                    # 3. 출처 확인
                    source_docs = retriever.invoke(english_query)
                    unique_sources = []
                    for doc in source_docs:
                        src_text = f"[{doc.metadata.get('source', 'Unknown')}] {doc.page_content[:50].replace(chr(10), ' ')}..."
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
                except Exception as e:
                    st.error(f"처리 중 오류 발생: {e}")
        else:
            st.error("시스템 오프라인.")