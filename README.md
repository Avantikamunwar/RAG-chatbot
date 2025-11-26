A lightweight Retrieval-Augmented Generation (RAG) chatbot built using:

FastAPI (Backend API)

Ollama (Local LLM inference)

Nomic Embed Text (Local embedding model)

Pinecone (Vector database)

HTML/JS Frontend (Simple chat UI)

This project allows users to upload documents, build a vector index, and ask questions answered using retrieved context + LLM output.

🚀 Features

✅ Local LLM using Ollama
✅ Local embeddings using nomic-embed-text
✅ Vector search using Pinecone
✅ Simple RAG pipeline (embed → store → retrieve → generate)
✅ REST API using FastAPI
✅ Basic frontend UI for chatting
✅ Fully offline inference (except Pinecone)