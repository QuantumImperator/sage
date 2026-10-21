
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai

from sage.ingestion import extract_pdf_pages, chunk_pages
from sage.retrieval import HybridRetriever
from sage.answering import answer_question


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

st.set_page_config(
    page_title="S.A.G.E. | NCERT Learning Intelligence",
    page_icon="📘",
    layout="centered",
)


@st.cache_resource
def load_retriever():
    pdf_path = ROOT / "src/sage/data/sources/jesc110.pdf"
    pages = extract_pdf_pages(str(pdf_path))
    chunks = chunk_pages(pages)
    return HybridRetriever(chunks)


@st.cache_resource
def load_client(api_key):
    return genai.Client(api_key=api_key)


st.title("S.A.G.E.")
st.caption("Source-grounded Assessment and Graph Evaluation")
st.markdown(
    "Ask questions about your NCERT Science chapter. "
    "Answers are grounded in retrieved textbook passages."
)

with st.sidebar:
    st.subheader("Knowledge source")
    st.write("NCERT Class 10 Science")
    st.caption("Chapter: The Human Eye and the Colourful World")
    st.divider()
    st.write("Retrieval: Hybrid lexical + semantic search")
    st.write("Answer generation: Gemini")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("View textbook evidence"):
                for source in message["sources"]:
                    st.markdown(
                        f"**[{source['evidence_id']}]** "
                        f"{source['source']} — PDF page {source['page']}"
                    )
                    st.write(source["content"])
                    st.divider()


question = st.chat_input("Ask a question about the chapter...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            model = os.getenv("GEMINI_MODEL")

            if not api_key or not model:
                raise RuntimeError(
                    "Configure GEMINI_API_KEY and GEMINI_MODEL in .env"
                )

            with st.spinner("Finding textbook evidence and preparing an answer..."):
                retriever = load_retriever()
                client = load_client(api_key)

                result = answer_question(
                    question=question,
                    retriever=retriever,
                    client=client,
                    model=model,
                )

            st.markdown(result["answer"])
            status=result.get("status","success")
            
            if status=="generation_unavailable":
                st.warning(
                    "gemini is temporarily unavailable. "
                    "retrieved textbook evidence is shown below."
                )
            elif status=="empty_generation":
                st.warning("the model returned an empty reponse")
            elif status=="insufficient_evidence":
                st.info("no sufficient supporting evidence was found. ")


            with st.expander("View textbook evidence", expanded=True):
                for source in result["sources"]:
                    st.markdown(
                        f"**[{source['evidence_id']}]** "
                        f"{source['source']} — PDF page {source['page']}"
                    )
                    st.write(source["content"])
                    st.divider()

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                }
            )

        except Exception as error:
            st.error(
                "S.A.G.E. could not complete this request. "
                "Check the configuration and terminal logs."
            )
            print(f"Request failed: {type(error).__name__}: {error}")

