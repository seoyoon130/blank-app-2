import os
import streamlit as st
import tempfile

from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor

# --------------------------------------------------------------------
# 1. Web Search Tool
# --------------------------------------------------------------------
def search_web():
    return TavilySearchResults(k=6, name="web_search")


# --------------------------------------------------------------------
# 2. PDF Tool
# --------------------------------------------------------------------
def load_pdf_files(uploaded_files):
    all_documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        all_documents.extend(documents)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(all_documents)

    vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())
    retriever = vector.as_retriever(search_kwargs={"k": 5})

    retriever_tool = create_retriever_tool(
        retriever,
        name="pdf_search",
        description="This tool gives you direct access to the uploaded PDF documents. "
                    "Always use this tool first when the question might be answered from the PDFs."
    )
    return retriever_tool


# --------------------------------------------------------------------
# 3. Agent + Prompt 구성
# --------------------------------------------------------------------
def build_agent(tools):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        "당신은 똑똑한 어시스턴트입니다. 당신은 세 가지 도구를 사용할 수 있습니다:\n"
        "- `csv_repl`: CSV 데이터 분석 및 시각화 전용 도구입니다. DataFrame `df`를 사용하여 파이썬 코드를 실행하며, 실제 존재하는 컬럼명을 그대로 사용해야 합니다.\n"
        "- `pdf_search`: PDF 문서의 내용을 검색하는 도구입니다. 질문이 PDF 내용과 관련 있다면 **반드시 가장 먼저** `pdf_search`를 사용해 보세요. "
        "만약 관련된 답변을 찾을 수 없으면 그때 다른 도구를 고려하세요.\n"
        "- `web_search`: CSV나 PDF와 무관한 일반 지식 질문, 최신 정보가 필요한 질문일 경우 사용합니다.\n\n"
        "도구 선택 우선순위 규칙:\n"
        "1. 질문이 PDF 문서와 관련 → `pdf_search`를 가장 먼저 시도. "
        "만약 관련 답을 못 찾으면 다른 도구(`csv_repl` 또는 `web_search`)를 사용할 수 있습니다.\n"
        "2. 질문에 '데이터'라는 표현이 있거나, CSV 분석/시각화가 필요하다면 `csv_repl`을 사용하세요.\n"
        "3. 위 두 가지가 모두 아니면 `web_search`를 사용하세요.\n\n"
        "`csv_repl`을 사용할 때는 실행한 파이썬 코드의 결과를 그대로 출력하고, 추가 설명이나 가공은 하지 마세요."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

    return agent_executor


# --------------------------------------------------------------------
# 4. Agent 실행 함수
# --------------------------------------------------------------------
def ask_agent(agent_executor, question: str):
    result = agent_executor.invoke({"input": question})
    answer = result["output"]

    # intermediate_steps에서 마지막만 가져오기
    if result.get("intermediate_steps"):
        last_action, _ = result["intermediate_steps"][-1]
        answer += f"\n\n출처:\n- Tool: {last_action.tool}, Query: {last_action.tool_input}"

    return f"답변:\n{answer}"


# --------------------------------------------------------------------
# 5. Streamlit 메인
# --------------------------------------------------------------------
def main():
    st.set_page_config(page_title="부산트립봇", layout="wide", page_icon=":ocean:")

    # ------------------------------
    # ✅ 배경 이미지 + 글씨 오버레이
    # ------------------------------
    st.markdown("""
        <style>
        .hero-container {
            position: relative;
            text-align: center;
        }
        .hero-image {
            width: 100%;
            border-radius: 10px;
        }
        .hero-text {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-size: 64px;
            font-weight: 900;
            text-shadow: 3px 3px 10px rgba(0,0,0,0.7);
        }
        </style>
        <div class="hero-container">
            <img src="data/busan.png" class="hero-image">
            <div class="hero-text">부산트립봇 🌊</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('---')

    # ------------------------------
    # ✅ PDF 업로드를 대화창 위로 이동
    # ------------------------------
    openai_api = st.text_input("🔑 OPENAI API 키", type="password")
    tavily_api = st.text_input("🔍 TAVILY API 키", type="password")
    pdf_docs = st.file_uploader("📂 PDF 파일 업로드", accept_multiple_files=True)

    st.markdown("---")

    # ------------------------------
    # 챗봇 본체
    # ------------------------------
    if openai_api and tavily_api:
        os.environ['OPENAI_API_KEY'] = openai_api
        os.environ['TAVILY_API_KEY'] = tavily_api

        tools = [search_web()]
        if pdf_docs:
            tools.append(load_pdf_files(pdf_docs))

        agent_executor = build_agent(tools)

        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        user_input = st.chat_input("✉️ 질문을 입력하세요!")

        if user_input:
            response = ask_agent(agent_executor, user_input)
            st.session_state["messages"].append({"role": "user", "content": user_input})
            st.session_state["messages"].append({"role": "assistant", "content": response})

        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

    else:
        st.warning("API 키를 입력하세요.")


if __name__ == "__main__":
    main()
