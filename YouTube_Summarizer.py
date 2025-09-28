# Import necessary libraries for the YouTube bot
import os
import re  # For extracting video id
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import gradio as gr
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (  # type: ignore[attr-defined]
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter  # For splitting text into manageable segments
from langchain_community.embeddings import HuggingFaceEmbeddings  # Open-source embeddings provider
from langchain_community.llms import Ollama  # Local inference for open-source chat models
from langchain_community.vectorstores import FAISS  # For efficient vector storage and similarity search
from langchain_core.prompts import PromptTemplate  # For defining prompt templates
from langchain_core.output_parsers import StrOutputParser
import tempfile
import whisper
from pytube import YouTube
import yt_dlp
import subprocess


@dataclass
class ModelConfig:
    """Lightweight holder for LLM and embedding configuration."""

    provider: str = os.getenv("LLM_PROVIDER", "ollama")
    llm_model: str = os.getenv("LLM_MODEL", "llama3.1")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "512"))


def load_model_config() -> ModelConfig:
    """Load runtime configuration from environment variables."""

    return ModelConfig()


def initialize_llm(config: ModelConfig):
    """Instantiate the selected open-source language model."""

    provider = config.provider.lower()

    if provider == "ollama":
        return Ollama(
            model=config.llm_model,
            base_url=config.ollama_base_url,
            temperature=config.temperature,
            num_predict=config.max_tokens,
        )

    raise ValueError(
        f"Unsupported LLM provider '{config.provider}'. Configure LLM_PROVIDER to 'ollama'."
    )


def initialize_embedding_model(config: ModelConfig):
    """Instantiate the embedding model used for retrieval."""

    return HuggingFaceEmbeddings(model_name=config.embedding_model)

def get_video_id(url: str) -> Optional[str]:
    """Extract the canonical 11-character YouTube video identifier."""

    if not url:
        return None

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    if host in {"www.youtube.com", "youtube.com", "m.youtube.com"}:
        if parsed.path.startswith("/shorts/"):
            return parsed.path.split("/")[-1][:11] or None
        query_params = parse_qs(parsed.query)
        if "v" in query_params:
            return query_params["v"][0]
    elif host == "youtu.be":
        return parsed.path.lstrip("/")[:11] or None

    match = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None


def _preferred_translation_code(transcript) -> Optional[str]:
    """Return an English translation code, if the transcript supports it."""

    languages = getattr(transcript, "translation_languages", None)
    if not languages:
        return None

    english_codes = [
        entry.get("language_code")
        for entry in languages
        if isinstance(entry, dict) and entry.get("language_code", "").startswith("en")
    ]

    if not english_codes:
        return None

    # Prefer the plain "en" code when available, otherwise fall back to the first match
    for code in english_codes:
        if code == "en":
            return code

    return english_codes[0]


def get_transcript_with_whisper(url: str) -> Optional[str]:
    """Fetch transcript using Whisper when YouTube API fails."""
    try:
        # Download audio
        youtube = YouTube(url)
        audio = youtube.streams.filter(only_audio=True).first()
        
        # Load Whisper model
        whisper_model = whisper.load_model("base")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Download and transcribe
            file = audio.download(output_path=tmpdir)
            transcription = whisper_model.transcribe(file, fp16=False)["text"].strip()
            
            # Convert to format compatible with YouTube API format
            return [{"text": transcription, "start": 0.0}]
    except Exception as e:
        print(f"Whisper transcription failed: {str(e)}")
        return None

def get_transcript(url: str) -> Optional[List[dict]]:
    """Fetch an English transcript for a YouTube video, if available."""
    video_id = get_video_id(url)
    if not video_id:
        return None

    # Check FFmpeg installation first
    if not check_ffmpeg():
        print("Error: FFmpeg is not installed. Please install FFmpeg first:")
        print("  brew install ffmpeg")
        return None
    
    try:
        print("Trying Whisper transcription...")
        
        # Configure yt-dlp
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
            'ffmpeg_location': '/opt/homebrew/bin/ffmpeg'
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ydl_opts['outtmpl'] = os.path.join(tmpdir, '%(id)s.%(ext)s')
            
            # Download audio
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # Find downloaded audio file
            audio_file = None
            for file in os.listdir(tmpdir):
                if file.endswith('.mp3'):
                    audio_file = os.path.join(tmpdir, file)
                    break
            
            if not audio_file:
                return None
            
            # Load Whisper model and transcribe
            whisper_model = whisper.load_model("base")
            result = whisper_model.transcribe(
                audio_file,
                fp16=False,
                language='en'
            )
            
            if result and "text" in result:
                return [{"text": result["text"].strip(), "start": 0.0}]
                
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        
    return None

def process(transcript: Optional[List[dict]]) -> str:
    """Flatten transcript entries into a plain-text block for downstream use."""

    if not transcript:
        return ""

    lines: List[str] = []

    for entry in transcript:
        # Handle both dict and FetchedTranscriptSnippet objects
        if hasattr(entry, 'text') and hasattr(entry, 'start'):
            # FetchedTranscriptSnippet object
            text = entry.text.strip() if entry.text else ""
            start = entry.start
        elif isinstance(entry, dict):
            # Dictionary format
            text = entry.get("text", "").strip()
            start = entry.get("start")
        else:
            continue

        if not text:
            continue

        if start is None:
            lines.append(f"Text: {text}")
        else:
            lines.append(f"Text: {text} Start: {start}")

    return "\n".join(lines)

def chunk_transcript(processed_transcript, chunk_size=200, chunk_overlap=20):
    # Initialize the RecursiveCharacterTextSplitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Split the transcript into chunks
    chunks = text_splitter.split_text(processed_transcript)
    return chunks

def create_faiss_index(chunks, embedding_model):
    """
    Create a FAISS index from text chunks using the specified embedding model.

    :param chunks: List of text chunks
    :param embedding_model: The embedding model to use
    :return: FAISS index
    """
    # Use the FAISS library to create an index from the provided text chunks
    return FAISS.from_texts(chunks, embedding_model)


def format_documents(documents) -> str:
    """Convert retrieved LangChain documents into a single context string."""

    if not documents:
        return ""

    return "\n\n".join(getattr(doc, "page_content", "") for doc in documents if doc)


def create_summary_prompt():
    """
    Create a PromptTemplate for summarizing a YouTube video transcript.
    
    :return: PromptTemplate object
    """
    # Define the template for the summary prompt
    template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an AI assistant tasked with summarizing YouTube video transcripts. Provide concise, informative summaries that capture the main points of the video content.

    Instructions:
    1. Summarize the transcript in a single concise paragraph.
    2. Ignore any timestamps in your summary.
    3. Focus on the spoken content (Text) of the video.

    Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Please summarize the following YouTube video transcript:

    {transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    
    # Create the PromptTemplate object with the defined template
    prompt = PromptTemplate(
        input_variables=["transcript"],
        template=template
    )
    
    return prompt


def create_summary_chain(llm, prompt):
    """Compose an LCEL runnable that formats the prompt and parses string output."""

    return prompt | llm | StrOutputParser()


def retrieve(query, faiss_index, k=7):
    """
    Retrieve relevant context from the FAISS index based on the user's query.

    Parameters:
        query (str): The user's query string.
        faiss_index (FAISS): The FAISS index containing the embedded documents.
        k (int, optional): The number of most relevant documents to retrieve (default is 7).

    Returns:
        list: A list of the k most relevant documents (or document chunks).
    """
    relevant_context = faiss_index.similarity_search(query, k=k)
    return relevant_context

def create_qa_prompt_template():
    """
    Create a PromptTemplate for question answering based on video content.
    Returns:
        PromptTemplate: A PromptTemplate object configured for Q&A tasks.
    """
    
    # Define the template string
    qa_template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an expert assistant providing detailed and accurate answers based on the following video content. Your responses should be:
    1. Precise and free from repetition
    2. Consistent with the information provided in the video
    3. Well-organized and easy to understand
    4. Focused on addressing the user's question directly
    If you encounter conflicting information in the video content, use your best judgment to provide the most likely correct answer based on context.
    Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.<|eot_id|>

    <|start_header_id|>user<|end_header_id|>
    Relevant Video Context: {context}
    Based on the above context, please answer the following question:
    {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    # Create the PromptTemplate object
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=qa_template
    )
    return prompt_template


def create_qa_chain(llm, prompt_template):
    """Return a runnable pipeline that feeds the prompt to the model and parses text."""

    return prompt_template | llm | StrOutputParser()


def generate_answer(question, faiss_index, qa_chain, k=7):
    """
    Retrieve relevant context and generate an answer based on user input.

    Args:
        question: str
            The user's question.
        faiss_index: FAISS
            The FAISS index containing the embedded documents.
        qa_chain:
            Runnable pipeline that applies the prompt to the model and parses text output.
        k: int, optional (default=3)
            The number of relevant documents to retrieve.

    Returns:
        str: The generated answer to the user's question.
    """

    # Retrieve relevant context
    relevant_context = retrieve(question, faiss_index, k=k)
    context = format_documents(relevant_context)

    if not context:
        return "I couldn't find enough information in the transcript to answer that question."

    # Generate answer using the QA chain
    return qa_chain.invoke({"context": context, "question": question})


# Initialize caches for the processed transcript and FAISS index
processed_transcript = ""
cached_video_id: Optional[str] = None
faiss_index_cache = None


def summarize_video(video_url):
    """
    Title: Summarize Video

    Description:
    This function generates a summary of the video using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.

    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.

    Returns:
        str: The generated summary of the video or a message indicating that no transcript is available.
    """
    global processed_transcript, cached_video_id, faiss_index_cache

    if not video_url:
        return "Please provide a valid YouTube URL."

    video_id = get_video_id(video_url)
    if not video_id:
        return "Please provide a valid YouTube URL."

    transcript_payload = get_transcript(video_url)
    if not transcript_payload:
        processed_transcript = ""
        cached_video_id = None
        faiss_index_cache = None
        return "Unable to fetch an English transcript for this video."

    processed_transcript = process(transcript_payload)
    if not processed_transcript:
        cached_video_id = None
        faiss_index_cache = None
        return "Transcript is empty after processing."

    cached_video_id = video_id
    faiss_index_cache = None  # Reset Q&A cache because transcript has changed

    config = load_model_config()

    try:
        llm = initialize_llm(config)
    except Exception as exc:
        return f"Failed to initialise language model: {exc}"

    summary_prompt = create_summary_prompt()
    summary_chain = create_summary_chain(llm, summary_prompt)

    try:
        summary = summary_chain.invoke({"transcript": processed_transcript})
    except Exception as exc:
        return f"Failed to generate summary: {exc}"

    return summary.strip()


def answer_question(video_url, user_question):
    """
    Title: Answer User's Question

    Description:
    This function retrieves relevant context from the FAISS index based on the user’s query 
    and generates an answer using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.

    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.
        user_question (str): The question posed by the user regarding the video.

    Returns:
        str: The answer to the user's question or a message indicating that the transcript 
             has not been fetched.
    """
    global processed_transcript, cached_video_id, faiss_index_cache

    if not user_question:
        return "Please provide a question about the video."

    if not video_url:
        return "Please provide a valid YouTube URL."

    video_id = get_video_id(video_url)
    if not video_id:
        return "Please provide a valid YouTube URL."

    if (not processed_transcript) or (cached_video_id and video_id != cached_video_id):
        transcript_payload = get_transcript(video_url)
        if not transcript_payload:
            processed_transcript = ""
            cached_video_id = None
            faiss_index_cache = None
            return "Unable to fetch an English transcript for this video."

        processed_transcript = process(transcript_payload)
        if not processed_transcript:
            cached_video_id = None
            faiss_index_cache = None
            return "Transcript is empty after processing."

        cached_video_id = video_id
        faiss_index_cache = None

    if not processed_transcript:
        return "No transcript available. Please fetch the transcript first."

    config = load_model_config()

    if faiss_index_cache is None:
        chunks = chunk_transcript(processed_transcript)
        if not chunks:
            return "Transcript is too short to build a knowledge base."

        try:
            embedding_model = initialize_embedding_model(config)
            faiss_index_cache = create_faiss_index(chunks, embedding_model)
        except Exception as exc:
            return f"Failed to create vector index: {exc}"

    try:
        llm = initialize_llm(config)
    except Exception as exc:
        return f"Failed to initialise language model: {exc}"

    qa_prompt = create_qa_prompt_template()
    qa_chain = create_qa_chain(llm, qa_prompt)

    try:
        answer = generate_answer(user_question, faiss_index_cache, qa_chain)
    except Exception as exc:
        return f"Failed to generate answer: {exc}"

    return answer.strip()


def build_interface() -> gr.Blocks:
    """Construct the Gradio interface used for the summarizer tool."""

    with gr.Blocks() as interface:
        gr.Markdown(
            "<h2 style='text-align: center;'>YouTube Video Summarizer and Q&A</h2>"
        )

        # Input field for YouTube URL
        video_url = gr.Textbox(
            label="YouTube Video URL", placeholder="Enter the YouTube Video URL"
        )

        # Outputs for summary and answer
        summary_output = gr.Textbox(label="Video Summary", lines=5)
        question_input = gr.Textbox(
            label="Ask a Question About the Video", placeholder="Ask your question"
        )
        answer_output = gr.Textbox(label="Answer to Your Question", lines=5)

        # Buttons for selecting functionalities after fetching transcript
        summarize_btn = gr.Button("Summarize Video")
        question_btn = gr.Button("Ask a Question")

        # Display status message for transcript fetch
        transcript_status = gr.Textbox(label="Transcript Status", interactive=False)

        # Set up button actions
        summarize_btn.click(summarize_video, inputs=video_url, outputs=summary_output)
        question_btn.click(
            answer_question, inputs=[video_url, question_input], outputs=answer_output
        )

    return interface


def main() -> None:
    """Launch the Gradio application using the configured port."""

    port = os.getenv("GRADIO_SERVER_PORT", "7860")
    try:
        port_number = int(port)
    except ValueError:
        port_number = 7860

    interface = build_interface()
    interface.launch(server_name="0.0.0.0", server_port=port_number)


def check_ffmpeg():
    """Check if FFmpeg is installed and available."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def test_transcript():
    url = "https://www.youtube.com/watch?v=2ePf9rue1Ao"
    video_id = get_video_id(url)
    print(f"Video ID: {video_id}")
    
    # Check FFmpeg installation
    if not check_ffmpeg():
        print("Error: FFmpeg is not installed. Please install FFmpeg first:")
        print("  brew install ffmpeg")
        return None
    
    try:
        print("\nTrying Whisper transcription...")
        
        # Configure yt-dlp with explicit FFmpeg path
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
            'ffmpeg_location': '/opt/homebrew/bin/ffmpeg'  # Typical Homebrew FFmpeg location
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set output template for the temporary directory
            ydl_opts['outtmpl'] = os.path.join(tmpdir, '%(id)s.%(ext)s')
            
            try:
                print("Downloading audio...")
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                
                # Find the downloaded audio file
                audio_file = None
                for file in os.listdir(tmpdir):
                    if file.endswith('.mp3'):
                        audio_file = os.path.join(tmpdir, file)
                        break
                
                if not audio_file:
                    print("Audio download failed")
                    return None
                
                print(f"Audio file size: {os.path.getsize(audio_file)} bytes")
                
                # Load Whisper model
                print("Loading Whisper model...")
                whisper_model = whisper.load_model("base")
                
                print("Transcribing with Whisper...")
                result = whisper_model.transcribe(
                    audio_file,
                    fp16=False,
                    language='en'
                )
                
                if result and "text" in result:
                    transcription = result["text"].strip()
                    print("\nTranscription successful!")
                    print(f"First 100 characters: {transcription[:100]}...")
                    return [{"text": transcription, "start": 0.0}]
                else:
                    print("No transcription generated")
                    return None
                    
            except Exception as e:
                print(f"Processing error: {str(e)}")
                return None
                
    except Exception as e:
        print(f"Setup failed: {str(e)}")
        return None

# Update the imports at the top of the file
# ...existing imports...
import yt_dlp

if __name__ == "__main__":
    # Launch the Gradio interface
    main()
