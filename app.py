"""
PDF Voice Reader - Production Ready Application
A comprehensive PDF reading application with Text-to-Speech integration.
Supports multiple TTS engines: gTTS, OpenAI, and ElevenLabs.
"""

import streamlit as st
import os
import json
import tempfile
from pathlib import Path
from datetime import datetime
import base64
from io import BytesIO

# PDF and Audio Processing
from pypdf import PdfReader
import gtts
from gtts import gTTS
import pyaudio
import wave

# External APIs (conditional imports)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import requests
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

PAGE_ICON = "📖"
APP_TITLE = "PDF Reader AI"
TEMP_AUDIO_DIR = Path(tempfile.gettempdir()) / "pdf_reader_audio"
TEMP_AUDIO_DIR.mkdir(exist_ok=True)
RECENT_FILES_JSON = "recent_files.json"

# TTS Configuration
TTS_ENGINES = {
    "gTTS (Free)": "gtts",
    "OpenAI TTS": "openai",
    "ElevenLabs": "elevenlabs"
}

VOICE_PROFILES = {
    "gtts": ["en", "es", "fr", "de", "it", "pt", "ja", "zh"],
    "openai": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
    "elevenlabs": ["Adam", "Bella", "Charlie", "Victoria"]
}

SPEED_OPTIONS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize all session state variables."""
    if "pdf_file" not in st.session_state:
        st.session_state.pdf_file = None
    if "pdf_text" not in st.session_state:
        st.session_state.pdf_text = {}  # Dictionary: {page_num: text}
    if "current_page" not in st.session_state:
        st.session_state.current_page = 0
    if "total_pages" not in st.session_state:
        st.session_state.total_pages = 0
    if "is_playing" not in st.session_state:
        st.session_state.is_playing = False
    if "current_audio_file" not in st.session_state:
        st.session_state.current_audio_file = None
    if "playback_speed" not in st.session_state:
        st.session_state.playback_speed = 1.0
    if "tts_engine" not in st.session_state:
        st.session_state.tts_engine = "gTTS (Free)"
    if "voice_profile" not in st.session_state:
        st.session_state.voice_profile = "en"
    if "recent_files" not in st.session_state:
        st.session_state.recent_files = load_recent_files()

# ============================================================================
# FILE MANAGEMENT
# ============================================================================

def load_recent_files() -> list:
    """Load recent files from local storage."""
    if Path(RECENT_FILES_JSON).exists():
        try:
            with open(RECENT_FILES_JSON, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_recent_files(files: list):
    """Save recent files to local storage."""
    with open(RECENT_FILES_JSON, 'w') as f:
        json.dump(files[-10:], f)  # Keep only last 10 files

def add_to_recent_files(filename: str):
    """Add file to recent files list."""
    recent = st.session_state.recent_files
    if filename in recent:
        recent.remove(filename)
    recent.insert(0, filename)
    st.session_state.recent_files = recent[-10:]
    save_recent_files(st.session_state.recent_files)

# ============================================================================
# PDF EXTRACTION ENGINE
# ============================================================================

def extract_pdf_text(pdf_file) -> dict:
    """
    Extract text from PDF on a page-by-page basis.
    Returns dict: {page_num: text_content}
    Handles scanned/blank pages gracefully.
    """
    text_dict = {}
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_path = tmp_file.name
        
        # Extract with pypdf
        pdf_reader = PdfReader(tmp_path)
        total_pages = len(pdf_reader.pages)
        
        for page_num in range(total_pages):
            try:
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                # Handle empty pages
                text_dict[page_num] = text if text.strip() else "[Blank/Scanned Page - No Text Extracted]"
            except Exception as e:
                text_dict[page_num] = f"[Error extracting page {page_num + 1}: {str(e)}]"
        
        # Cleanup
        os.unlink(tmp_path)
        return text_dict
    
    except Exception as e:
        st.error(f"❌ PDF Extraction Error: {str(e)}")
        return {}

# ============================================================================
# TEXT-TO-SPEECH ENGINE
# ============================================================================

class TTSEngine:
    """Abstracted TTS engine for easy plugin switching."""
    
    @staticmethod
    def generate_audio(text: str, engine: str, voice_profile: str, speed: float = 1.0) -> BytesIO:
        """
        Generate audio from text using specified engine.
        Returns: BytesIO object containing MP3/WAV audio data.
        """
        if engine == "gtts":
            return TTSEngine._generate_gtts(text, voice_profile, speed)
        elif engine == "openai" and OPENAI_AVAILABLE:
            return TTSEngine._generate_openai(text, voice_profile, speed)
        elif engine == "elevenlabs" and ELEVENLABS_AVAILABLE:
            return TTSEngine._generate_elevenlabs(text, voice_profile, speed)
        else:
            raise ValueError(f"TTS Engine '{engine}' not available or not configured.")
    
    @staticmethod
    def _generate_gtts(text: str, language: str, speed: float) -> BytesIO:
        """Google Text-to-Speech implementation."""
        try:
            # Chunk text for better processing
            max_chars = 100
            chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
            
            audio_buffer = BytesIO()
            mp3_buffer = BytesIO()
            
            for chunk in chunks:
                if chunk.strip():
                    tts = gTTS(text=chunk, lang=language, slow=(speed < 1.0))
                    chunk_buffer = BytesIO()
                    tts.write_to_fp(chunk_buffer)
                    chunk_buffer.seek(0)
                    mp3_buffer.write(chunk_buffer.read())
            
            mp3_buffer.seek(0)
            return mp3_buffer
        
        except Exception as e:
            st.error(f"❌ gTTS Error: {str(e)}")
            return None
    
    @staticmethod
    def _generate_openai(text: str, voice: str, speed: float) -> BytesIO:
        """OpenAI Text-to-Speech implementation."""
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
            if not api_key:
                st.error("❌ OpenAI API Key not configured in secrets")
                return None
            
            openai.api_key = api_key
            response = openai.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text[:4096],  # OpenAI has 4096 char limit
                speed=speed
            )
            
            audio_buffer = BytesIO()
            audio_buffer.write(response.content)
            audio_buffer.seek(0)
            return audio_buffer
        
        except Exception as e:
            st.error(f"❌ OpenAI TTS Error: {str(e)}")
            return None
    
    @staticmethod
    def _generate_elevenlabs(text: str, voice_id: str, speed: float) -> BytesIO:
        """ElevenLabs Text-to-Speech implementation."""
        try:
            api_key = st.secrets.get("ELEVENLABS_API_KEY")
            if not api_key:
                st.error("❌ ElevenLabs API Key not configured in secrets")
                return None
            
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers = {
                "xi-api-key": api_key,
                "Content-Type": "application/json"
            }
            data = {
                "text": text[:5000],  # ElevenLabs limit
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75
                }
            }
            
            response = requests.post(url, json=data, headers=headers)
            if response.status_code == 200:
                audio_buffer = BytesIO(response.content)
                return audio_buffer
            else:
                st.error(f"❌ ElevenLabs Error: {response.status_code} - {response.text}")
                return None
        
        except Exception as e:
            st.error(f"❌ ElevenLabs Error: {str(e)}")
            return None

# ============================================================================
# AUDIO PLAYBACK & CACHE
# ============================================================================

def generate_and_cache_audio(page_num: int) -> str:
    """
    Generate audio for current page and cache it.
    Returns: Path to cached audio file.
    """
    if page_num not in st.session_state.pdf_text:
        return None
    
    text = st.session_state.pdf_text[page_num]
    cache_filename = f"page_{page_num}_{st.session_state.tts_engine}_{st.session_state.voice_profile}.mp3"
    cache_path = TEMP_AUDIO_DIR / cache_filename
    
    # Return if already cached
    if cache_path.exists():
        return str(cache_path)
    
    # Generate new audio
    with st.spinner(f"🎙️ Generating audio for page {page_num + 1}..."):
        engine = TTS_ENGINES[st.session_state.tts_engine]
        audio_buffer = TTSEngine.generate_audio(
            text,
            engine,
            st.session_state.voice_profile,
            st.session_state.playback_speed
        )
        
        if audio_buffer:
            with open(cache_path, 'wb') as f:
                f.write(audio_buffer.getvalue())
            return str(cache_path)
    
    return None

def get_audio_download_link(file_path: str, file_name: str) -> str:
    """Generate download link for audio file."""
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:audio/mp3;base64,{b64}" download="{file_name}">⬇️ Download Audio</a>'

# ============================================================================
# STATE & NAVIGATION LOGIC
# ============================================================================

def calculate_progress() -> float:
    """Calculate reading progress percentage."""
    if st.session_state.total_pages == 0:
        return 0.0
    return (st.session_state.current_page / st.session_state.total_pages) * 100

def next_page():
    """Navigate to next page."""
    if st.session_state.current_page < st.session_state.total_pages - 1:
        st.session_state.current_page += 1
        st.session_state.is_playing = False

def prev_page():
    """Navigate to previous page."""
    if st.session_state.current_page > 0:
        st.session_state.current_page -= 1
        st.session_state.is_playing = False

def jump_to_page(page_num: int):
    """Jump to specific page."""
    if 0 <= page_num < st.session_state.total_pages:
        st.session_state.current_page = page_num
        st.session_state.is_playing = False

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_header():
    """Render application header."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(f"{PAGE_ICON} {APP_TITLE}")
    with col2:
        st.caption(f"_v1.0 | Built with Streamlit_")

def render_upload_section():
    """Render PDF upload and recent files."""
    st.subheader("📥 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file:
        # Extract text from PDF
        text_dict = extract_pdf_text(uploaded_file)
        
        if text_dict:
            st.session_state.pdf_file = uploaded_file.name
            st.session_state.pdf_text = text_dict
            st.session_state.total_pages = len(text_dict)
            st.session_state.current_page = 0
            st.session_state.is_playing = False
            
            add_to_recent_files(uploaded_file.name)
            st.success(f"✅ PDF loaded: {uploaded_file.name} ({st.session_state.total_pages} pages)")
    
    # Recent files section
    if st.session_state.recent_files:
        st.divider()
        st.subheader("📋 Recent Files")
        for file in st.session_state.recent_files:
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(f"📄 {file}", key=f"recent_{file}", use_container_width=True):
                    st.info(f"Selected: {file}")
            with col2:
                if st.button("🗑️", key=f"delete_{file}"):
                    st.session_state.recent_files.remove(file)
                    save_recent_files(st.session_state.recent_files)
                    st.rerun()

def render_reader_section():
    """Render main PDF reader and controls."""
    if not st.session_state.pdf_file:
        st.info("👆 Please upload a PDF to get started")
        return
    
    # Active document info
    st.subheader("📖 Active Document")
    doc_info_col1, doc_info_col2 = st.columns(2)
    with doc_info_col1:
        st.metric("Current Document", st.session_state.pdf_file)
    with doc_info_col2:
        st.metric("Pages", f"{st.session_state.current_page + 1} / {st.session_state.total_pages}")
    
    st.divider()
    
    # Progress bar
    progress = calculate_progress()
    st.progress(progress / 100, text=f"Progress: {progress:.1f}%")
    
    st.divider()
    
    # Page navigation controls
    st.subheader("🎮 Playback Controls")
    nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns(5)
    
    with nav_col1:
        if st.button("⏮️ First", use_container_width=True):
            jump_to_page(0)
            st.rerun()
    
    with nav_col2:
        if st.button("⏪ Previous", use_container_width=True):
            prev_page()
            st.rerun()
    
    with nav_col3:
        st.session_state.is_playing = st.toggle("▶️ Play", value=st.session_state.is_playing)
    
    with nav_col4:
        if st.button("⏩ Next", use_container_width=True):
            next_page()
            st.rerun()
    
    with nav_col5:
        if st.button("⏭️ Last", use_container_width=True):
            jump_to_page(st.session_state.total_pages - 1)
            st.rerun()
    
    st.divider()
    
    # Page text display
    current_text = st.session_state.pdf_text.get(st.session_state.current_page, "")
    st.subheader(f"📄 Page {st.session_state.current_page + 1} Content")
    
    text_container = st.container(border=True)
    with text_container:
        st.write(current_text[:1000] + "..." if len(current_text) > 1000 else current_text)
    
    st.divider()
    
    # Audio player
    st.subheader("🔊 Audio Playback")
    
    if st.session_state.is_playing:
        audio_file_path = generate_and_cache_audio(st.session_state.current_page)
        
        if audio_file_path:
            with open(audio_file_path, 'rb') as f:
                st.audio(f.read(), format="audio/mp3")
            
            # Download option
            st.markdown(
                get_audio_download_link(audio_file_path, f"page_{st.session_state.current_page + 1}.mp3"),
                unsafe_allow_html=True
            )

def render_sidebar():
    """Render settings sidebar."""
    st.sidebar.title("⚙️ Settings")
    
    st.sidebar.subheader("🎙️ Voice Settings")
    
    # TTS Engine selection
    tts_engine = st.sidebar.selectbox(
        "Text-to-Speech Engine",
        list(TTS_ENGINES.keys()),
        index=list(TTS_ENGINES.keys()).index(st.session_state.tts_engine)
    )
    st.session_state.tts_engine = tts_engine
    
    # Voice profile selection
    current_engine = TTS_ENGINES[tts_engine]
    voice_options = VOICE_PROFILES.get(current_engine, [])
    
    if voice_options:
        voice_profile = st.sidebar.selectbox(
            "Voice Profile",
            voice_options,
            index=voice_options.index(st.session_state.voice_profile) if st.session_state.voice_profile in voice_options else 0
        )
        st.session_state.voice_profile = voice_profile
    
    st.sidebar.divider()
    
    # Playback speed
    st.sidebar.subheader("⚡ Playback Speed")
    speed = st.sidebar.select_slider(
        "Speed Multiplier",
        options=SPEED_OPTIONS,
        value=st.session_state.playback_speed
    )
    st.session_state.playback_speed = speed
    
    st.sidebar.divider()
    
    # Application info
    st.sidebar.subheader("ℹ️ Information")
    st.sidebar.info(
        "**PDF Reader AI** v1.0\n\n"
        "A production-ready PDF voice reader with support for multiple TTS engines.\n\n"
        "📚 **Features:**\n"
        "- Multi-page PDF support\n"
        "- Google TTS (free)\n"
        "- OpenAI TTS API\n"
        "- ElevenLabs API\n"
        "- Audio caching\n"
        "- Page navigation\n"
        "- Recent files tracking"
    )

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    # Configure Streamlit
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Render UI
    render_header()
    
    # Main content area
    render_upload_section()
    st.divider()
    render_reader_section()
    
    # Sidebar
    render_sidebar()

if __name__ == "__main__":
    main()
