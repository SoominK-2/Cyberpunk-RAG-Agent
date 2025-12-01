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

# --- 1. 환경 변수 및 초기 설정 ---
# 🚨 주의: API 키를 여기에 직접 입력하거나 환경 변수로 설정해야 합니다.
# 챗봇 노트북에서 사용한 키를 재사용합니다.
# 실제 제출 시에는 이 부분은 사용자에게 맡기는 것이 좋습니다.
# app.py 파일의 키 설정 부분을 이렇게 변경합니다.
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
RAG_MODEL = "gpt-4o-mini"
DATA_FILE = "cyberpunk_shards.txt"
CHROMA_DIR = "./cyberpunk_chroma_db"

# Streamlit 앱 제목 설정
st.title("사이버펑크 2077 세계관 백과사전 AI")
st.caption("제공된 샤드 데이터만을 기반으로 답변하는 RAG 챗봇입니다.")

# --- 2. 데이터 로드 및 RAG 체인 구축 (캐시 처리) ---

# @st.cache_resource: 앱이 시작될 때 이 함수를 한 번만 실행하고 결과를 캐시합니다.
@st.cache_resource
def load_database():
    try:
        # 1. 텍스트 로드
        loader = TextLoader(DATA_FILE, encoding="utf-8")
        documents = loader.load()

        # 2. 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        # 3. 임베딩 모델 및 벡터 DB 생성
        embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
        db = Chroma.from_documents(
            documents=docs, 
            embedding=embed_model, 
            persist_directory=CHROMA_DIR
        )
        retriever = db.as_retriever()
        
        # 4. LLM 및 프롬프트 정의 (RAG_Chatbot.ipynb의 셀 4 내용 재사용)
        llm = ChatOpenAI(model_name=RAG_MODEL)
        template = """
        당신은 '사이버펑크 2077' 세계관 전문가입니다.
        제공된 Context(샤드 내용)만을 바탕으로 사용자의 질문에 답변해 주세요.
        만약 Context에 질문과 관련된 내용이 없다면, "죄송합니다. 제가 아는 샤드 내용 중에는 해당 정보가 없습니다."라고 답변하세요.
        
        Context:
        {context}
        
        Question:
        {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        # 5. RAG 체인 구성 (RunnablePassthrough 사용)
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain

    except Exception as e:
        # 파일이 없거나 API 키 오류 발생 시
        st.error(f"데이터 로드 또는 DB 구축 중 오류 발생: {e}")
        st.caption(f"'{DATA_FILE}' 파일과 OpenAI API 키 설정을 확인해 주세요.")
        return None

# 데이터베이스 로드 및 RAG 체인 초기화
rag_chain = load_database()

# --- 3. 채팅 UI 및 Multi-Turn 구현 ---

if rag_chain:
    # 챗 기록이 없으면 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 이전 채팅 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 처리
    if prompt_text := st.chat_input("사이버펑크 세계관에 대해 질문해보세요."):
        
        # 1. 사용자 질문을 기록 및 표시
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        with st.chat_message("user"):
            st.markdown(prompt_text)

        # 2. LLM 호출 및 답변 생성
        with st.chat_message("assistant"):
            with st.spinner("Night City의 지식을 검색 중입니다..."):
                # RAG 체인 호출 (Multi-turn은 Streamlit의 messages history로 처리합니다.)
                # RAG 체인이 질문(prompt_text)을 받아서 답변을 생성합니다.
                full_response = rag_chain.invoke(prompt_text)
                st.markdown(full_response)
        
        # 3. 답변을 기록
        st.session_state.messages.append({"role": "assistant", "content": full_response})