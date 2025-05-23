"""
Chainlit Paul-Graham-Essay RAG Bot
──────────────────────────────────
"""

import os, asyncio
from pathlib import Path
from operator import itemgetter

import chainlit as cl
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableConfig

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_huggingface import (
    HuggingFaceEndpoint,
    HuggingFaceEndpointEmbeddings,
)

# ── 1. ENV ────────────────────────────────────────────────────────────
load_dotenv()                          # reads .env
HF_LLM_ENDPOINT   = os.environ["HF_LLM_ENDPOINT"]
HF_EMBED_ENDPOINT = os.environ["HF_EMBED_ENDPOINT"]
HF_TOKEN          = os.environ["HF_TOKEN"]

# ── 2. LOAD & INDEX DOCUMENTS ─────────────────────────────────────────
DATA_FILE   = Path(__file__).parent / "data" / "paul_graham_essays.txt"

docs        = TextLoader(DATA_FILE).load()
chunks      = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=30
             ).split_documents(docs)

embeddings  = HuggingFaceEndpointEmbeddings(
    model=HF_EMBED_ENDPOINT,
    task="feature-extraction",
    huggingfacehub_api_token=HF_TOKEN,
)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever   = vectorstore.as_retriever()

# ── 3. PROMPT & LLM ───────────────────────────────────────────────────
RAG_TEMPLATE = """<|start_header_id|>system<|end_header_id|>
You answer the QUESTION only from the CONTEXT. If you don't know, say so.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
QUESTION: {query}

CONTEXT:
{context}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""
prompt      = PromptTemplate.from_template(RAG_TEMPLATE)

llm         = HuggingFaceEndpoint(
    endpoint_url=HF_LLM_ENDPOINT,
    task="text-generation",
    max_new_tokens=512,
    temperature=0.01,
    huggingfacehub_api_token=HF_TOKEN,
)

rag_chain = (
    {"context": itemgetter("query") | retriever, "query": itemgetter("query")}
    | prompt
    | llm
)

# ── 4. CHAINLIT CALLBACKS ─────────────────────────────────────────────
@cl.author_rename
def rename(author: str) -> str:
    return {"Assistant": "Paul Graham Essay Bot"}.get(author, author)


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("chain", rag_chain)
    await cl.Message(
        content="👋 Hi!  Ask me anything about Paul Graham’s essays."
    ).send()


@cl.on_message
async def on_user_message(message: cl.Message):
    chain = cl.user_session.get("chain")
    msg   = cl.Message(content="")

    async for chunk in chain.astream(
        {"query": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
