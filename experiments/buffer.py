import os
import subprocess
import tempfile
import streamlit as st
import whisper
from dotenv import load_dotenv
import time
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


# ==========================
# SIDEBAR NAVIGATION
# ==========================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Chatbot", "Architecture"]
)


# ==========================
# ARCHITECTURE PAGE
# ==========================

if page == "Architecture":

    st.title("System Architecture")

    st.subheader("Core Architecture Diagram")

    col1, col2 = st.columns([2, 1])  # adjust ratio as needed

    with col1:
        st.image("architecture.png", width=1000)

    with col2:
        st.subheader("Process Flow")
        st.markdown("""
        **Pipeline:**

        1. User uploads video  
        2. Video → Audio (FFmpeg)  
        3. Audio → Text (Whisper)  
        4. Text → Chunks (LangChain)  
        5. Chunks → Embeddings (OpenAI)  
        6. Stored in Pinecone  
        7. User asks question  
        8. Retriever finds relevant chunks  
        9. LLM generates answer  
        """)

    st.stop()


# ==========================
# CHATBOT PAGE
# ==========================

st.title("Video Q&A with AI")
st.write("Upload a video and ask questions using text or your microphone.")


# ==========================
# Load Whisper model
# ==========================

model = whisper.load_model("tiny.en")


# ==========================
# Functions
# ==========================

def log_time(step_name, start_time):
    elapsed = time.time() - start_time
    st.write(f"{step_name}: {elapsed:.2f} sec")


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
    result = model.transcribe(audio_file)
    return result["text"]


def create_vector_store(chunks):

    pc = Pinecone(api_key=PINECONE_API_KEY)

    if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:

        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_ENV
            )
        )

    index = pc.Index(INDEX_NAME)
    
    index.delete(delete_all=True)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    vector_db = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        text_key="text"
    )

    vector_db.add_texts(chunks)

    return vector_db


def build_qa_chain(vector_db):

    retriever = vector_db.as_retriever(search_kwargs={"k": 2})

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(
            model_name="gpt-3.5-turbo",
            openai_api_key=OPENAI_API_KEY,
            temperature=0
        ),
        chain_type="stuff",
        retriever=retriever
    )

    return qa_chain


def summarize_video(transcript):

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=OPENAI_API_KEY,
        temperature=0
    )

    short_text = transcript[:4000]

    return llm.predict(f"Summarize this video:\n\n{short_text}")


# ==========================
# Upload video
# ==========================

uploaded_file = st.file_uploader(
    "Upload a video file",
    type=["mp4", "mov", "mkv"]
)

if uploaded_file:

    if "current_video" not in st.session_state:
        st.session_state.current_video = None

    if uploaded_file.name != st.session_state.current_video:

        st.session_state.current_video = uploaded_file.name
        st.session_state.processed = False
        st.session_state.chat_history = []
        st.session_state.qa_chain = None

        st.rerun()

    if not st.session_state.get("processed", False):

        with st.spinner("Processing video..."):

            temp_video = tempfile.NamedTemporaryFile(delete=False)
            temp_video.write(uploaded_file.read())

            video_path = temp_video.name
            audio_path = video_path + ".wav"

            convert_video_to_audio(video_path, audio_path)

            # start = time.time()
            transcript = transcribe_audio(audio_path)
            # log_time("Transcription", start)

            st.success("Transcription complete")

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100
            )

            chunks = splitter.split_text(transcript)

            vector_db = create_vector_store(chunks)

            qa_chain = build_qa_chain(vector_db)

            st.session_state.qa_chain = qa_chain
            st.session_state.processed = True

            summary = summarize_video(transcript)
            st.session_state.video_summary = summary

            st.success("Video ready for questions!")


# ==========================
# Display summary
# ==========================

if st.session_state.get("video_summary", None):
    st.divider()
    st.subheader("Video Summary")
    st.write(st.session_state.video_summary)


# ==========================
# Chat Interface
# ==========================
if st.session_state.get("processed", False):

    st.divider()
    st.subheader("💬 Ask Questions About the Video")

    qa_chain = st.session_state.qa_chain

    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "audio_question" not in st.session_state:
        st.session_state.audio_question = None

    # --------------------------
    # User Inputs
    # --------------------------

    # Text input
    user_text = st.chat_input("Type your question here...")

    # Microphone input
    audio_input = st.audio_input("🎤 Or ask using your microphone")

    if audio_input:
        st.session_state.audio_question = audio_input

    user_question = None

    # Handle TEXT input
    if user_text:
        user_question = user_text

    # Handle VOICE input
    elif st.session_state.audio_question:
        with st.spinner("Transcribing your voice question..."):
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_audio.write(st.session_state.audio_question.read())
            temp_audio.flush()

            user_question = transcribe_audio(temp_audio.name)
            st.info(f"🗣 Transcribed question: {user_question}")

        # Clear audio after processing
        st.session_state.audio_question = None

    # --------------------------
    # Run QA Chain
    # --------------------------

    if user_question:
        with st.spinner("Getting answer..."):
            answer = qa_chain.run(user_question)

        st.session_state.chat_history.append(("user", user_question))
        st.session_state.chat_history.append(("assistant", answer))

    # --------------------------
    # Display chat history
    # --------------------------

    for role, message in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").write(message)
        else:
            st.chat_message("assistant").write(message)


    st.divider()
    st.subheader("Ask Questions About the Video")

    qa_chain = st.session_state.qa_chain

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # user_text = st.chat_input("Type your question here...")

    if user_text:

        with st.spinner("Getting answer..."):
            answer = qa_chain.run(user_text)

        st.session_state.chat_history.append(("user", user_text))
        st.session_state.chat_history.append(("assistant", answer))

    for role, message in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").write(message)
        else:
            st.chat_message("assistant").write(message)