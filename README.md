# 🤖 Hybrid RAG AI Agent

## Overview

A production-ready AI Agent that combines **Retrieval-Augmented
Generation (RAG)** with **real-time web search** using LangGraph
orchestration.

Built to demonstrate: - Hybrid document + web retrieval - Intelligent
routing - Source-cited answers - Confidence scoring - Restricted
execution mode - Transparent agent trace

------------------------------------------------------------------------

## 🚀 Key Highlights

-   **Hybrid Retrieval** -- Dynamically fuses internal PDF knowledge
    (Pinecone vector DB) with live web search (Tavily).
-   **LangGraph Orchestration** -- Structured routing across RAG, web,
    and fusion nodes.
-   **Restricted Mode** -- Enforces internal-only responses with
    external access disabled.
-   **Source-Cited Answers** -- Clean citations: `[RAG-x]`, `[WEB-x]`.
-   **Confidence Scoring** -- Heuristic scoring based on retrieval
    coverage and corroboration.
-   **Transparent Trace** -- Step-by-step workflow visibility.

------------------------------------------------------------------------

## 🏗 Architecture

**Frontend:** Streamlit\
**Backend:** FastAPI\
**Agent Framework:** LangGraph\
**LLM:** Groq (Llama 3)\
**Embeddings:** sentence-transformers/all-MiniLM-L6-v2\
**Vector Store:** Pinecone\
**Search:** Tavily API

------------------------------------------------------------------------

## 🧠 What This Demonstrates

-   Production-grade RAG system design
-   Hybrid retrieval and context fusion
-   Hallucination mitigation through grounding
-   Clean modular architecture
-   Extensible tool-based agent framework

------------------------------------------------------------------------

## 🔧 Tech Stack

Python · FastAPI · Streamlit · LangGraph · LangChain · Pinecone · Groq ·
Tavily

------------------------------------------------------------------------

## 📌 Use Cases

-   Enterprise knowledge assistants
-   Document intelligence systems
-   AI copilots with controlled external access
-   Retrieval-backed Q&A platforms

------------------------------------------------------------------------

## 📎 Author

Built as part of a RAG & Multimodal AI Developer Assessment.

------------------------------------------------------------------------

**Status:** Submission Ready
