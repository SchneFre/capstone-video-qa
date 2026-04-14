# Video Q&A with AI (Streamlit + Whisper + LangChain + Pinecone)

An AI-powered application that lets you **upload videos and ask questions about their content** using either text or voice.  
The system transcribes video audio, creates embeddings, stores them in a vector database, and uses an LLM to answer questions intelligently.

---

## Features

- Upload video files (`mp4`, `mov`, `mkv`)
- Automatic speech-to-text using OpenAI Whisper
- Transcript chunking with LangChain
- Embeddings with OpenAI
- Vector storage using Pinecone
- Conversational Q&A with GPT-3.5
- Ask questions via microphone or text input
- Auto-generated video summary
- Clean Streamlit UI with architecture view

---

## System Architecture

1. User uploads video  
2. FFmpeg extracts audio  
3. Whisper transcribes audio → text  
4. LangChain splits text into chunks  
5. OpenAI embeddings generated  
6. Stored in Pinecone vector database  
7. User asks a question  
8. Retriever fetches relevant chunks  
9. GPT model generates answer  

---

## Tech Stack

- **Frontend:** Streamlit  
- **Speech-to-Text:** OpenAI Whisper  
- **LLM:** OpenAI GPT-3.5 Turbo  
- **Embeddings:** OpenAI Embeddings  
- **Vector DB:** Pinecone  
- **Orchestration:** LangChain  
- **Media Processing:** FFmpeg  

---
# Test 
- streamlit run .\deployment\app.py
- .env file is required for OPENAI_API_KEY & PINECONE_API_KEY