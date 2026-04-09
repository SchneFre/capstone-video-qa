import os
import subprocess
import tempfile
from tracemalloc import start
import streamlit as st
import whisper
from dotenv import load_dotenv
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
# from faster_whisper import WhisperModel
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
st.write("Upload a video and ask questions using text or your microphone.")

# ==========================
# Load Whisper model
# ==========================

model = whisper.load_model("tiny.en")  # change to "base", "small", "medium", "large-v2" if needed


# ==========================
# Functions
# ==========================

def log_time(step_name, start_time):
    elapsed = time.time() - start_time
    st.write(f"{step_name}: {elapsed:.2f} sec")


def split_audio(audio_path, chunk_length=30):
    output_dir = "chunks"
    os.makedirs(output_dir, exist_ok=True)

    command = [
        "ffmpeg",
        "-i", audio_path,
        "-f", "segment",
        "-segment_time", str(chunk_length),
        "-c", "copy",
        f"{output_dir}/chunk_%03d.wav"
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir)])


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
    
    # model = whisper.load_model("tiny.en")

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
    
    # clear previous video vectors
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

def summarize_video(qa_chain, transcript):
    prompt = f"Summarize the following video content in a concise paragraph:\n\n{transcript}"
    summary = qa_chain.run(prompt)
    return summary

# ==========================
# Upload video
# ==========================

uploaded_file = st.file_uploader(
    "Upload a video file",
    type=["mp4", "mov", "mkv"]
)

if uploaded_file:

    # Detect if a new video was uploaded
    if "current_video" not in st.session_state:
        st.session_state.current_video = None

    if uploaded_file.name != st.session_state.current_video:

        # Reset session state
        st.session_state.current_video = uploaded_file.name
        st.session_state.processed = False
        st.session_state.chat_history = []
        st.session_state.qa_chain = None

        # Force full refresh so old chat disappears
        st.rerun()
        
    # Run processing if not yet processed
    if not st.session_state.get("processed", False):

        with st.spinner("Processing video..."):

            temp_video = tempfile.NamedTemporaryFile(delete=False)
            temp_video.write(uploaded_file.read())

            video_path = temp_video.name
            audio_path = video_path + ".wav"

            # Extract audio
            convert_video_to_audio(video_path, audio_path)

            # Transcribe 
            start = time.time()
            transcript = transcribe_audio(audio_path)
            log_time("transcribe_audio", start) 

            st.success("Transcription complete")

            # Split transcript
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150
            )

            chunks = splitter.split_text(transcript)

            # Create vector DB
            start = time.time()
            vector_db = create_vector_store(chunks)
            log_time("vector_db", start) 

            # Build QA chain
            qa_chain = build_qa_chain(vector_db)

            st.session_state.qa_chain = qa_chain
            st.session_state.processed = True

            # Generate summary immediately           
            summary = summarize_video(qa_chain, transcript)
            st.session_state.video_summary = summary

            st.success("Video ready for questions!")




# ==========================
# Display summary
# ==========================
if st.session_state.get("video_summary", None):
    st.divider()
    st.subheader("📄 Video Summary")
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