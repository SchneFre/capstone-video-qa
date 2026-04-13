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

INDEX_NAME = "video-qa-index-multinamespace" # All videos share the same index
PINECONE_ENV = "us-east-1"     # Pinecone environment/region
EMBEDDING_DIMENSION = 1536      # Must match OpenAI embedding dimension

# ==========================
# Streamlit page configuration
# ==========================
st.set_page_config(page_title="Video Q&A", layout="wide")

# ==========================
# Sidebar navigation
# ==========================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Chatbot", "Architecture"])

# ==========================
# Architecture Page
# ==========================
if page == "Architecture":
    st.title("System Architecture")
    col1, col2 = st.columns([2, 1])
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
        6. Stored in Pinecone (namespace = video filename)  
        7. User asks question  
        8. Retriever finds relevant chunks  
        9. LLM generates answer  
        """)
    st.stop()

# ==========================
# Chatbot Page
# ==========================
st.title("Video Q&A with AI")
st.write("Upload a video and ask questions using text or your microphone.")

# ==========================
# Load Whisper model
# ==========================
whisper_model = whisper.load_model("tiny.en")

# ==========================
# Helper functions
# ==========================
def sanitize_namespace(filename: str) -> str:
    """
    Convert video filename into a Pinecone-compatible namespace
    by replacing non-alphanumeric characters with hyphens and lowercasing.
    """
    return "".join(c.lower() if c.isalnum() else "-" for c in filename)

def convert_video_to_audio(video_path: str, audio_path: str,
                           ffmpeg_dir: str = r"C:\FFmpeg\ffmpeg-8.0.1-essentials_build\bin"):
    os.environ["PATH"] += os.pathsep + ffmpeg_dir
    ffmpeg_exe = os.path.join(ffmpeg_dir, "ffmpeg.exe")
    subprocess.run([ffmpeg_exe, "-y", "-i", video_path, "-ac", "1", "-ar", "16000", audio_path], check=True)

def transcribe_audio(audio_file: str) -> str:
    result = whisper_model.transcribe(audio_file)
    return result["text"]

def create_or_get_vector_store(chunks: list[str], namespace: str) -> PineconeVectorStore:
    """
    Creates a Pinecone vector store if it doesn't exist.
    If the namespace already contains vectors, it returns the existing store.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create shared index if it doesn't exist
    existing_indexes = [i["name"] for i in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
        )

    # Create embeddings object
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Initialize vector store with namespace
    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        text_key="text",
        namespace=namespace
    )

    # Check if namespace already has vectors
    try:
        existing_vectors = vector_store.similarity_search("test", k=1)
        namespace_exists = len(existing_vectors) > 0
    except Exception:
        namespace_exists = False

    if not namespace_exists:
        # Add chunks to the vector store
        vector_store.add_texts(chunks)

    return vector_store

def build_qa_chain(vector_store: PineconeVectorStore) -> RetrievalQA:
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY),
        chain_type="stuff",
        retriever=retriever
    )
    return qa_chain

def summarize_text(text: str) -> str:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
    short_text = text[:4000]
    return llm.predict(f"Summarize this video:\n\n{short_text}")

# ==========================
# Video upload and processing
# ==========================
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "mkv"])

if uploaded_file:

    video_namespace = sanitize_namespace(uploaded_file.name)

    if "current_video" not in st.session_state:
        st.session_state.current_video = None

    if uploaded_file.name != st.session_state.current_video:
        st.session_state.current_video = uploaded_file.name
        st.session_state.processed = False
        st.session_state.chat_history = []
        st.session_state.qa_chain = None
        st.session_state.video_summary = None
        st.rerun()

    if not st.session_state.get("processed", False):
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Try to get vector store for this namespace
        vector_store = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
            text_key="text",
            namespace=video_namespace
        )
        try:
            existing_vectors = vector_store.similarity_search("test", k=1)
            namespace_exists = len(existing_vectors) > 0
        except Exception:
            namespace_exists = False

        if namespace_exists:
            st.success(f"Vector store for '{uploaded_file.name}' already exists. Skipping processing.")
            qa_chain = build_qa_chain(vector_store)
            st.session_state.qa_chain = qa_chain
            st.session_state.processed = True           

        else:
            with st.spinner("Processing video..."):
                # Save uploaded video to a temporary file
                temp_video_file = tempfile.NamedTemporaryFile(delete=False)
                temp_video_file.write(uploaded_file.read())
                video_path = temp_video_file.name
                audio_path = video_path + ".wav"

                # Convert to audio and transcribe
                convert_video_to_audio(video_path, audio_path)
                transcript_text = transcribe_audio(audio_path)
                st.success("Transcription complete")

                # Split transcript into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                chunks = text_splitter.split_text(transcript_text)

                # Create vector store and add chunks
                vector_store = create_or_get_vector_store(chunks, video_namespace)

                # Build QA chain
                qa_chain = build_qa_chain(vector_store)

                st.session_state.qa_chain = qa_chain
                st.session_state.processed = True

                # Summarize video
                summary = summarize_text(transcript_text)
                st.session_state.video_summary = summary
                st.success("Video ready for questions!")

# ==========================
# Display video summary
# ==========================
if st.session_state.get("video_summary", None):
    st.divider()
    st.subheader("Video Summary")
    st.write(st.session_state.video_summary)

# ==========================
# Chat interface
# ==========================
if st.session_state.get("processed", False):
    st.divider()
    st.subheader("Ask Questions About the Video")
    qa_chain = st.session_state.qa_chain

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.chat_input("Type your question here...")

    if user_question:
        with st.spinner("Getting answer..."):
            answer_text = qa_chain.run(user_question)
        st.session_state.chat_history.append(("user", user_question))
        st.session_state.chat_history.append(("assistant", answer_text))

    for role, message in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").write(message)
        else:
            st.chat_message("assistant").write(message)