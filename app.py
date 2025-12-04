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
    
# 다양한 유형의 질문으로 구성 (관계, 설정, 사건, 인물 등)
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
        
        # 프롬프트 수정: 고유명사 리스트 보강 및 환각 방지 강화
        template = """
        당신은 '사이버펑크 2077' 세계관의 냉소적이고 유능한 정보 브로커(Fixer)입니다.
        
        [지시사항]
        1. 말투: "~입니다/습니다" 금지. "~야", "~군", "~하더군" 같은 반말 사용.
        2. 고유명사 (한국어 공식 번역 준수): 
           - Panam -> **팬앰** (절대 '팬암' 아님)
           - Hanako -> **하나코** (절대 '한코' 아님)
           - Yorinobu -> 요리노부
           - Saburo -> 사부로
           - Relic -> 렐릭
           - Evelyn -> 이블린 (절대 '이브린' 아님)
           - Arasaka -> 아라사카
           - Militech -> 밀리테크
           - Johnny -> 조니
           - V -> V(브이)
           - 기타 고유명사도 한국어 공식 번역 준수.
        3. 태도: 너무 딱딱하게 설명하지 말고, 의뢰인에게 정보를 브리핑하듯 자연스럽게 이야기해.
        4. 근거: 반드시 아래 제공된 Context(정보)들을 종합해서 답해. 
        5. 엄격한 제한: Context에 없는 내용(날씨, 후속작 소식, 네 생각 등)은 절대 지어내지 마. "그건 내 정보망(데이터)에 없는 내용이야."라고 딱 잘라 거절해.
        
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
        if rag_chain:
            # 랜덤 로딩 메시지
            loading_texts = [
                "📡 암호 해독 중...",
                "💾 데이터뱅크 접속...",
                "⚡ 넷러닝 프로토콜 시작...",
                "🔍 샤드 데이터 스캔 중...",
                "🕶️ 정보망 가동..."
            ]
            status_placeholder = st.empty()
            
            try:
                # 랜덤 텍스트 선택
                with status_placeholder.status(random.choice(loading_texts), expanded=True) as status:
                    status.write("질문 번역 중...")
                    llm_trans = ChatOpenAI(model_name=RAG_MODEL)
                    trans_prompt = ChatPromptTemplate.from_template(
                        "Translate the following Korean text to English for a Cyberpunk 2077 database search. Output ONLY the translated text.\nText: {text}"
                    )
                    trans_chain = trans_prompt | llm_trans | StrOutputParser()
                    english_query = trans_chain.invoke({"text": user_input}).strip()
                    
                    status.write(f"검색어 변환: **{english_query}**")
                    status.write("데이터베이스 검색 중...")
                    
                    response = rag_chain.invoke(english_query)
                    
                    source_docs = retriever.invoke(english_query)
                    unique_sources = []
                    for doc in source_docs:
                        clean_content = doc.page_content.replace("\n", " ").replace("\r", " ")
                        src_text = f"[{doc.metadata.get('source', 'Unknown')}] {clean_content[:50]}..."
                        if src_text not in unique_sources:
                            unique_sources.append(src_text)
                    
                    status.update(label="✅ 데이터 확보 완료", state="complete", expanded=False)

                st.markdown(response)
                
                if unique_sources:
                    with st.expander("🔍 데이터 출처 확인"):
                        for src in unique_sources:
                            st.caption(src)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response, 
                    "sources": unique_sources
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"처리 중 오류 발생: {e}")
        else:
            st.error("시스템 오프라인.")