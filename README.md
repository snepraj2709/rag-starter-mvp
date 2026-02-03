# RAG Starter MVP (SEBI Annual Report 2024-25)

## Overview
This project is a simple Retrieval-Augmented Generation (RAG) chatbot for finance analysts. It lets users ask questions about the SEBI Annual Report 2024-25 via a Streamlit UI, using OpenAI for embeddings/chat and Pinecone as the vector database.

## Live URL
- https://rag-starter-mvp-production.up.railway.app

## Tools & Libraries
- Python (app + ingestion)
- Streamlit (web UI)
- LangChain (RAG pipeline, retrieval chains)
- OpenAI (embeddings + chat model)
- Pinecone (vector database)
- python-dotenv (environment variable loading)
- pypdf (PDF parsing during ingestion)
- Railway (deployment hosting)

## Constraints & Configuration
- Required environment variables:
  - `OPENAI_API_KEY`
  - `PINECONE_API_KEY`
  - `INDEX_NAME`
- The Pinecone index must already be populated with the SEBI Annual Report content.
  - Use `ingestion.py` to load and chunk `data/annual-report-2024-25.pdf` and push embeddings to Pinecone.
- The Streamlit app runs from `chatbot/streamlit_chatbot.py` and uses a conversational retrieval chain in `chatbot/statefull_bot.py`.

