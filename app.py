# --- 1. SQLite 패치 ---
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
        retriever = db.as_retriever(search_kwargs={"k": 25})
        llm = ChatOpenAI(model_name=RAG_MODEL)
        
        # ⭐️ 프롬프트 수정: 말투 강화 및 번역 규칙 주입 ⭐️
        template = """
        당신은 '사이버펑크 2077' 세계관의 냉소적이고 유능한 정보 브로커(Fixer)입니다.
        
        [지시사항]
        1. 말투: "~입니다/습니다" 같은 존댓말 절대 금지. "~야", "~군", "~하더군" 같은 반말이나 하대하는 말투를 사용해.
        2. 고유명사 번역 규칙: 
           - 영어로 된 고유명사는 사이버펑크 2077 한국어 공식 번역명을 따라야 해.
           - Relic -> 렐릭, Evelyn -> 이블린, Arasaka -> 아라사카, Militech -> 밀리테크, Maelstrom -> 멜스트롬, Johnny -> 조니, V -> V(브이).
           - 그 외의 영어 이름도 발음나는 대로 자연스러운 한국어로 표기해.
        3. 근거: 반드시 아래 제공된 Context(정보)들을 종합해서 답해. 
        4. 모름: 정보가 없으면 "내 정보망엔 없는 건인데. 다른 걸 물어봐."라고 짧게 끊어.
        
        Context:
        {context}
        
        Question:
        {question}
        
        Answer (정보 브로커 스타일):
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

# --- 5. 채팅 인터페이스 (버그 수정됨) ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "원하는 정보를 말해봐. 가격은... 나중에 청구하지."}]

# 1. 이전 대화 기록 출력 (여기가 순수하게 '기록'만 보여주는 곳)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 저장된 출처가 있을 때만 표시
        if msg.get("sources"):
            with st.expander("🔍 데이터 출처 확인"):
                for src in msg["sources"]:
                    st.caption(src)

# 2. 사용자 입력 처리
if user_input := st.chat_input("데이터 검색...") or st.session_state.get("prompt_input"):
    if st.session_state.get("prompt_input"):
        del st.session_state["prompt_input"]

    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # ⭐️ 핵심: 사용자 메시지는 '화면 표시' 후 세션에 저장 (순서 중요)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # AI 답변 생성 과정
    with st.chat_message("assistant"):
        if rag_chain:
            status_placeholder = st.empty()
            
            try:
                # 상태창 (번역 및 검색 과정 표시)
                with status_placeholder.status("📡 암호 해독 중...", expanded=True) as status:
                    status.write("질문 번역 중...")
                    llm_trans = ChatOpenAI(model_name=RAG_MODEL)
                    trans_prompt = ChatPromptTemplate.from_template(
                        "Translate the following Korean text to English for a Cyberpunk 2077 database search. Output ONLY the translated text.\nText: {text}"
                    )
                    trans_chain = trans_prompt | llm_trans | StrOutputParser()
                    english_query = trans_chain.invoke({"text": user_input}).strip()
                    
                    status.write(f"검색어 변환: **{english_query}**")
                    status.write("데이터베이스 검색 중...")
                    
                    # RAG 실행
                    response = rag_chain.invoke(english_query)
                    
                    # 출처 확인
                    source_docs = retriever.invoke(english_query)
                    unique_sources = []
                    for doc in source_docs:
                        # Lore 데이터인지 Shard 데이터인지에 따라 표시 방식 최적화
                        src_type = doc.metadata.get('source', 'Unknown')
                        content_snippet = doc.page_content[:50].replace(chr(10), ' ')
                        src_text = f"[{src_type}] {content_snippet}..."
                        
                        if src_text not in unique_sources:
                            unique_sources.append(src_text)
                    
                    status.update(label="✅ 데이터 확보 완료", state="complete", expanded=False)

                # ⭐️ 답변 출력 (화면에만 먼저 그림)
                st.markdown(response)
                
                # ⭐️ 출처 출력 (화면에만 먼저 그림)
                if unique_sources:
                    with st.expander("🔍 데이터 출처 확인"):
                        for src in unique_sources:
                            st.caption(src)
                
                # ⭐️ 모든 과정이 끝난 후, 세션에 '한 번만' 저장 (중복 버그 해결의 핵심)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response, 
                    "sources": unique_sources
                })
                
            except Exception as e:
                st.error(f"처리 중 오류 발생: {e}")
        else:
            st.error("시스템 오프라인.")