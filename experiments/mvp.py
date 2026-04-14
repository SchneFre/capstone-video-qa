import os
import subprocess
import tempfile
import streamlit as st
import whisper
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone, ServerlessSpec


# ==========================
# Load environment variables
# ==========================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = "video-qa-index"
PINECONE_ENV = "us-east-1"


# ==========================
# Streamlit config
# ==========================

st.set_page_config(page_title="Video Q&A", layout="wide")

st.title("🎬 Video Q&A with AI")
st.write("Upload a video and ask questions about its content.")


# ==========================
# Functions
# ==========================

def convert_video_to_audio(video_path, audio_path, ffmpeg_dir=r"C:\FFmpeg\ffmpeg-8.0.1-essentials_build\bin"):
    os.environ["PATH"] += os.pathsep + ffmpeg_dir
    ffmpeg_exe = os.path.join(ffmpeg_dir, "ffmpeg.exe")

    subprocess.run([
        ffmpeg_exe,
        "-y",
        "-i", video_path,
        "-ac", "1",
        "-ar", "16000",
        audio_path
    ], check=True)


def transcribe_audio(audio_file):

    model = whisper.load_model("tiny")

    result = model.transcribe(audio_file)

    return result["text"]


def create_vector_store(chunks):

    pc = Pinecone(api_key=PINECONE_API_KEY)

    if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
        )

    index = pc.Index(INDEX_NAME)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    vector_db = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        text_key="text"
    )

    vector_db.add_texts(chunks)

    return vector_db


def build_qa_chain(vector_db):

    retriever = vector_db.as_retriever(search_type="similarity")

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(
            model_name="gpt-3.5-turbo",
            openai_api_key=OPENAI_API_KEY
        ),
        chain_type="stuff",
        retriever=retriever
    )

    return qa_chain


# ==========================
# Upload video
# ==========================

uploaded_file = st.file_uploader("Upload a video file", type=["mp4","mov","mkv"])


if uploaded_file:

    if "processed" not in st.session_state:

        with st.spinner("Processing video..."):

            # Save video temporarily
            temp_video = tempfile.NamedTemporaryFile(delete=False)
            temp_video.write(uploaded_file.read())

            video_path = temp_video.name
            audio_path = video_path + ".wav"

            # 1️⃣ Extract audio
            convert_video_to_audio(video_path, audio_path)

            # 2️⃣ Transcribe
            transcript = transcribe_audio(audio_path)

            st.success("Transcription complete")

            # 3️⃣ Split transcript
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150
            )

            chunks = splitter.split_text(transcript)

            # 4️⃣ Create vector DB
            vector_db = create_vector_store(chunks)

            # 5️⃣ Create QA chain
            qa_chain = build_qa_chain(vector_db)

            st.session_state.qa_chain = qa_chain
            st.session_state.chat_history = []
            st.session_state.processed = True

            st.success("Video ready for questions!")


# ==========================
# Chat Interface
# ==========================

if "processed" in st.session_state:

    st.divider()
    st.subheader("💬 Ask Questions About the Video")

    user_input = st.chat_input("Ask something about the video...")

    if user_input:

        qa_chain = st.session_state.qa_chain

        answer = qa_chain.run(user_input)

        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("ai", answer))


    for role, message in st.session_state.chat_history:

        if role == "user":
            st.chat_message("user").write(message)
        else:
            st.chat_message("assistant").write(message)