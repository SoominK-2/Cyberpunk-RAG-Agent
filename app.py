import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 1. 페이지 설정 ---
st.set_page_config(
    page_title="NIGHT CITY ARCHIVES",
    page_icon="💾",
    layout="wide",
    initial_sidebar_state="expanded" # 사이드바 기본 열림
)

# --- 2. 사이드바 (사용자 가이드 & 추천 질문) ---
with st.sidebar:
    st.title("넷러너 가이드")
    st.markdown("---")
    
    st.subheader("이용 팁")
    st.info(
        """
        이 에이전트는 **게임 내 샤드(Shard)**와 **위키 데이터**를 
        기반으로 답변합니다.
        
        - **가능:** 특정 인물, 사건, 샤드 내용 요약
        - **불가능:** 실시간 뉴스, 게임 공략, 개인적인 잡담
        """
    )
    
    st.subheader("추천 질문")
    example_questions = [
        "아라사카와 V의 관계에 대해 말해줘",
        "사이버사이코시스란 뭐야?",
        "조니 실버핸드는 누구야?",
        "'학생의 일기' 샤드 내용은?",
        "나이트 시티의 주요 기업들은?"
    ]
    
    for ex in example_questions:
        if st.button(ex):
            # 버튼 클릭 시 입력창에 자동 입력 효과 (session_state 활용)
            st.session_state.prompt_input = ex

# --- 3. 환경 변수 및 설정 ---
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
RAG_MODEL = "gpt-4o-mini"
CHROMA_DIR = "./cyberpunk_chroma_db"

# 커스텀 CSS (이전과 동일)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&display=swap');
    .stApp { background-color: #050505; font-family: 'Rajdhani', sans-serif; }
    h1 { color: #FCEE0A !important; text-transform: uppercase; text-shadow: 2px 2px 0px #00F0FF; }
    .stCaption { color: #00F0FF !important; border-left: 3px solid #FCEE0A; padding-left: 10px; }
    .stChatMessage { background-color: #1a1a1a; border: 1px solid #333; border-radius: 0px !important; }
    div[data-testid="stChatMessage"]:nth-child(odd) { border-left: 5px solid #FCEE0A; }
    div[data-testid="stChatMessage"]:nth-child(even) { border-right: 5px solid #00F0FF; background-color: #0a0a0a; }
    .stChatInput input { background-color: #111 !important; color: #FCEE0A !important; border: 2px solid #FCEE0A !important; }
    .stSpinner > div { border-top-color: #FCEE0A !important; }
</style>
""", unsafe_allow_html=True)

st.title("🔌 NIGHT CITY ARCHIVES")
st.caption("ACCESSING SECURE DATASLATE... // WELCOME, EDGERUNNER.")

# --- 4. 데이터 로드 및 체인 구축 (출처 기능 추가) ---
@st.cache_resource
def load_database():
    try:
        all_docs = []
        
        # (1) 샤드 데이터 로드
        if os.path.exists("cyberpunk_shards.txt"):
            loader1 = TextLoader("cyberpunk_shards.txt", encoding="utf-8")
            docs1 = loader1.load()
            for d in docs1: d.metadata["source"] = "인게임 샤드 데이터"
            all_docs.extend(docs1)

        # (2) 위키(Lore) 데이터 로드
        if os.path.exists("cyberpunk_lore.txt"):
            loader2 = TextLoader("cyberpunk_lore.txt", encoding="utf-8")
            docs2 = loader2.load()
            for d in docs2: d.metadata["source"] = "위키(Lore) 데이터"
            all_docs.extend(docs2)
            
        if not all_docs:
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_docs)

        embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
        db = Chroma.from_documents(splits, embed_model, persist_directory=CHROMA_DIR)
        retriever = db.as_retriever()
        
        llm = ChatOpenAI(model_name=RAG_MODEL)
        
        # 시스템 프롬프트
        system_prompt = (
            "당신은 '사이버펑크 2077' 세계관 전문가입니다. "
            "아래 제공된 Context만을 기반으로 답변하세요. "
            "만약 Context에 정보가 없다면 '해당 내용은 내 데이터베이스에 없습니다.'라고 답하세요. "
            "\n\n"
            "Context:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # 출처를 반환할 수 있는 체인 생성 (create_retrieval_chain 사용)
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        return rag_chain

    except Exception as e:
        st.error(f"⚠️ 시스템 오류: {e}")
        return None

rag_chain = load_database()

# --- 5. 채팅 UI 및 로직 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "원하는 정보를 말해봐. 가격은... 나중에 청구하지."})

# 이전 대화 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # 저장된 출처가 있다면 표시
        if "sources" in message:
            with st.expander("🔍 참고한 데이터 출처"):
                for src in message["sources"]:
                    st.text(f"- {src}")

# 사용자 입력 처리
# 사이드바 버튼을 눌렀다면 그 값을, 아니면 일반 입력을 받음
if user_input := st.chat_input("데이터 검색...") or st.session_state.get("prompt_input"):
    # 버튼 클릭값 초기화 (재실행 방지)
    if st.session_state.get("prompt_input"):
        del st.session_state["prompt_input"]

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("📡 CONNECTING TO NET..."):
            if rag_chain:
                # 체인 실행 (입력 키는 'input'이어야 함)
                result = rag_chain.invoke({"input": user_input})
                
                response_text = result["answer"]
                source_docs = result["context"]
                
                # 출처 정리 (중복 제거)
                sources = []
                for doc in source_docs:
                    # 메타데이터나 내용의 일부를 출처로 표시
                    src_info = f"[{doc.metadata.get('source', '알 수 없음')}] {doc.page_content[:30]}..."
                    if src_info not in sources:
                        sources.append(src_info)

                st.markdown(response_text)
                
                # 출처 아코디언 표시
                if sources:
                    with st.expander("🔍 참고한 데이터 출처"):
                        for src in sources:
                            st.text(f"- {src}")
                
                # 세션에 답변과 출처 저장
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_text,
                    "sources": sources
                })
            else:
                st.error("데이터베이스 로드 실패")