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
from gtts import gTTS

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
    "gtts": {
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
        "Portuguese": "pt",
        "Japanese": "ja",
        "Chinese": "zh-CN"
    },
    "openai": {
        "Alloy": "alloy",
        "Echo": "echo",
        "Fable": "fable",
        "Onyx": "onyx",
        "Nova": "nova",
        "Shimmer": "shimmer"
    },
    "elevenlabs": {
        "Adam": "adam",
        "Bella": "bella",
        "Charlie": "charlie",
        "Victoria": "victoria"
    }
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
        st.session_state.pdf_text = {}
    if "current_page" not in st.session_state:
        st.session_state.current_page = 0
    if "total_pages" not in st.session_state:
        st.session_state.total_pages = 0
    if "current_audio_file" not in st.session_state:
        st.session_state.current_audio_file = None
    if "playback_speed" not in st.session_state:
        st.session_state.playback_speed = 1.0
    if "tts_engine" not in st.session_state:
        st.session_state.tts_engine = "gTTS (Free)"
    if "voice_profile" not in st.session_state:
        st.session_state.voice_profile = "English"
    if "recent_files" not in st.session_state:
        st.session_state.recent_files = load_recent_files()
    if "audio_buffer_cache" not in st.session_state:
        st.session_state.audio_buffer_cache = {}

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
        json.dump(files[-10:], f)

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
    """Extract text from PDF on a page-by-page basis."""
    text_dict = {}
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_path = tmp_file.name
        
        pdf_reader = PdfReader(tmp_path)
        total_pages = len(pdf_reader.pages)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for page_num in range(total_pages):
            try:
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                text_dict[page_num] = text if text.strip() else f"[Blank/Scanned Page {page_num + 1}]"
                
                progress = (page_num + 1) / total_pages
                progress_bar.progress(progress)
                status_text.text(f"Extracting... {page_num + 1}/{total_pages}")
                
            except Exception as e:
                text_dict[page_num] = f"[Error on page {page_num + 1}]"
        
        progress_bar.empty()
        status_text.empty()
        
        os.unlink(tmp_path)
        return text_dict
    
    except Exception as e:
        st.error(f"❌ PDF Error: {str(e)}")
        return {}

# ============================================================================
# TEXT-TO-SPEECH ENGINE
# ============================================================================

class TTSEngine:
    """Abstracted TTS engine for easy plugin switching."""
    
    @staticmethod
    def generate_audio(text: str, engine: str, voice_profile: str, speed: float = 1.0) -> BytesIO:
        """Generate audio from text using specified engine."""
        if engine == "gtts":
            return TTSEngine._generate_gtts(text, voice_profile, speed)
        elif engine == "openai" and OPENAI_AVAILABLE:
            return TTSEngine._generate_openai(text, voice_profile, speed)
        elif engine == "elevenlabs" and ELEVENLABS_AVAILABLE:
            return TTSEngine._generate_elevenlabs(text, voice_profile, speed)
        else:
            raise ValueError(f"TTS Engine '{engine}' not available.")
    
    @staticmethod
    def _generate_gtts(text: str, language_key: str, speed: float) -> BytesIO:
        """Google Text-to-Speech implementation."""
        try:
            lang_code = VOICE_PROFILES["gtts"].get(language_key, "en")
            
            max_chars = 100
            chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
            
            mp3_buffer = BytesIO()
            
            for chunk in chunks:
                if chunk.strip():
                    tts = gTTS(text=chunk, lang=lang_code, slow=(speed < 1.0))
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
    def _generate_openai(text: str, voice_key: str, speed: float) -> BytesIO:
        """OpenAI Text-to-Speech implementation."""
        try:
            api_key = st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("❌ OpenAI API Key not configured.")
                return None
            
            openai.api_key = api_key
            
            voice_code = VOICE_PROFILES["openai"].get(voice_key, "alloy")
            text_chunk = text[:4096]
            
            response = openai.audio.speech.create(
                model="tts-1",
                voice=voice_code,
                input=text_chunk,
                speed=speed
            )
            
            audio_buffer = BytesIO()
            audio_buffer.write(response.content)
            audio_buffer.seek(0)
            return audio_buffer
        
        except Exception as e:
            st.error(f"❌ OpenAI Error: {str(e)}")
            return None
    
    @staticmethod
    def _generate_elevenlabs(text: str, voice_key: str, speed: float) -> BytesIO:
        """ElevenLabs Text-to-Speech implementation."""
        try:
            api_key = st.secrets.get("ELEVENLABS_API_KEY") if hasattr(st, "secrets") else os.getenv("ELEVENLABS_API_KEY")
            if not api_key:
                st.error("❌ ElevenLabs API Key not configured.")
                return None
            
            voice_id = VOICE_PROFILES["elevenlabs"].get(voice_key, "adam")
            
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            headers = {
                "xi-api-key": api_key,
                "Content-Type": "application/json"
            }
            data = {
                "text": text[:5000],
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
                st.error(f"❌ ElevenLabs Error: {response.status_code}")
                return None
        
        except Exception as e:
            st.error(f"❌ ElevenLabs Error: {str(e)}")
            return None

# ============================================================================
# AUDIO PLAYBACK & CACHE
# ============================================================================

def get_cache_key(page_num: int) -> str:
    """Generate cache key for audio."""
    engine = TTS_ENGINES[st.session_state.tts_engine]
    return f"page_{page_num}_{engine}_{st.session_state.voice_profile}"

def generate_and_get_audio(page_num: int) -> BytesIO:
    """Generate audio for current page or retrieve from cache."""
    if page_num not in st.session_state.pdf_text:
        return None
    
    cache_key = get_cache_key(page_num)
    
    if cache_key in st.session_state.audio_buffer_cache:
        buffer = st.session_state.audio_buffer_cache[cache_key]
        buffer.seek(0)
        return buffer
    
    text = st.session_state.pdf_text[page_num]
    with st.spinner(f"🎙️ Generating audio..."):
        engine = TTS_ENGINES[st.session_state.tts_engine]
        audio_buffer = TTSEngine.generate_audio(
            text,
            engine,
            st.session_state.voice_profile,
            st.session_state.playback_speed
        )
        
        if audio_buffer:
            st.session_state.audio_buffer_cache[cache_key] = audio_buffer
            audio_buffer.seek(0)
            return audio_buffer
    
    return None

def get_audio_download_link(audio_buffer: BytesIO, file_name: str) -> str:
    """Generate download link for audio file."""
    if audio_buffer is None:
        return ""
    
    audio_buffer.seek(0)
    data = audio_buffer.read()
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

def prev_page():
    """Navigate to previous page."""
    if st.session_state.current_page > 0:
        st.session_state.current_page -= 1

def first_page():
    """Go to first page."""
    st.session_state.current_page = 0

def last_page():
    """Go to last page."""
    st.session_state.current_page = st.session_state.total_pages - 1

def jump_to_page(page_num: int):
    """Jump to specific page."""
    if 0 <= page_num < st.session_state.total_pages:
        st.session_state.current_page = page_num

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_header():
    """Render application header."""
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.title(f"{PAGE_ICON} {APP_TITLE}")
    with col2:
        st.caption("🚀 Ready")
    with col3:
        st.caption("_v1.0_")
    st.markdown("---")

def render_upload_section():
    """Render PDF upload and recent files."""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📥 Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")
        
        if uploaded_file:
            with st.spinner("📖 Processing PDF..."):
                text_dict = extract_pdf_text(uploaded_file)
            
            if text_dict:
                st.session_state.pdf_file = uploaded_file.name
                st.session_state.pdf_text = text_dict
                st.session_state.total_pages = len(text_dict)
                st.session_state.current_page = 0
                st.session_state.audio_buffer_cache = {}
                
                add_to_recent_files(uploaded_file.name)
                st.success(f"✅ {uploaded_file.name} ({st.session_state.total_pages} pages)")
    
    with col2:
        if st.session_state.recent_files:
            st.subheader("📋 Recent Files")
            for file in st.session_state.recent_files[:5]:
                if st.button(f"📄 {file}", key=f"recent_{file}", use_container_width=True):
                    st.info(f"✅ {file}")

def render_reader_section():
    """Render main PDF reader and controls."""
    if not st.session_state.pdf_file:
        st.info("👆 Please upload a PDF to get started")
        return
    
    st.subheader("📖 Active Document")
    doc_col1, doc_col2, doc_col3 = st.columns([2, 1, 1])
    with doc_col1:
        st.metric("📄 Document", st.session_state.pdf_file)
    with doc_col2:
        st.metric("📍 Page", st.session_state.current_page + 1)
    with doc_col3:
        st.metric("📊 Total", st.session_state.total_pages)
    
    st.divider()
    
    progress = calculate_progress()
    st.progress(progress / 100, text=f"Progress: {progress:.1f}%")
    
    st.divider()
    
    st.subheader("🎮 Playback Controls")
    nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns(5)
    
    with nav_col1:
        if st.button("⏮️ First", use_container_width=True):
            first_page()
            st.rerun()
    
    with nav_col2:
        if st.button("⏪ Previous", use_container_width=True):
            prev_page()
            st.rerun()
    
    with nav_col3:
        if st.button("▶️ Play", use_container_width=True):
            st.session_state.current_audio_file = generate_and_get_audio(st.session_state.current_page)
            st.rerun()
    
    with nav_col4:
        if st.button("⏩ Next", use_container_width=True):
            next_page()
            st.rerun()
    
    with nav_col5:
        if st.button("⏭️ Last", use_container_width=True):
            last_page()
            st.rerun()
    
    st.divider()
    
    st.subheader("🔍 Jump to Page")
    jump_page = st.selectbox(
        "Select page",
        range(1, st.session_state.total_pages + 1),
        index=st.session_state.current_page,
        key="page_selector"
    )
    if jump_page - 1 != st.session_state.current_page:
        jump_to_page(jump_page - 1)
        st.rerun()
    
    st.divider()
    
    current_text = st.session_state.pdf_text.get(st.session_state.current_page, "")
    st.subheader(f"📄 Page {st.session_state.current_page + 1} Content")
    
    with st.container(border=True):
        st.write(current_text[:2000] + "..." if len(current_text) > 2000 else current_text)
    
    st.divider()
    
    st.subheader("🔊 Audio Generation & Playback")
    
    audio_col1, audio_col2 = st.columns([3, 1])
    
    with audio_col1:
        if st.button("🎙️ Generate Audio", use_container_width=True):
            audio_buffer = generate_and_get_audio(st.session_state.current_page)
            
            if audio_buffer:
                st.session_state.current_audio_file = audio_buffer
                st.success("✅ Audio ready!")
    
    with audio_col2:
        if st.button("🧹 Clear Cache", use_container_width=True):
            st.session_state.audio_buffer_cache = {}
            st.info("Cache cleared!")
    
    if st.session_state.current_audio_file:
        st.audio(st.session_state.current_audio_file, format="audio/mp3")
        
        download_link = get_audio_download_link(
            st.session_state.current_audio_file,
            f"page_{st.session_state.current_page + 1}.mp3"
        )
        st.markdown(download_link, unsafe_allow_html=True)

def render_sidebar():
    """Render settings sidebar."""
    st.sidebar.title("⚙️ Settings")
    
    st.sidebar.subheader("🎙️ Voice Settings")
    
    tts_engine = st.sidebar.selectbox(
        "TTS Engine",
        list(TTS_ENGINES.keys()),
        index=list(TTS_ENGINES.keys()).index(st.session_state.tts_engine)
    )
    st.session_state.tts_engine = tts_engine
    
    current_engine = TTS_ENGINES[tts_engine]
    voice_options = list(VOICE_PROFILES.get(current_engine, {}).keys())
    
    if voice_options:
        voice_profile = st.sidebar.selectbox(
            "Voice Profile",
            voice_options,
            index=voice_options.index(st.session_state.voice_profile) if st.session_state.voice_profile in voice_options else 0
        )
        st.session_state.voice_profile = voice_profile
    
    st.sidebar.divider()
    
    st.sidebar.subheader("⚡ Playback Speed")
    speed = st.sidebar.select_slider(
        "Speed",
        options=SPEED_OPTIONS,
        value=st.session_state.playback_speed
    )
    st.session_state.playback_speed = speed
    st.sidebar.caption(f"Current: {speed}x")
    
    st.sidebar.divider()
    
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.info(
        "**PDF Reader AI** v1.0\n\n"
        "Production-ready PDF voice reader.\n\n"
        "**Features:**\n"
        "✅ Multi-page PDF\n"
        "✅ Google TTS (free)\n"
        "✅ OpenAI TTS\n"
        "✅ ElevenLabs API\n"
        "✅ Audio caching\n"
        "✅ Page navigation\n"
        "✅ Recent files\n\n"
        "Built with Streamlit, pypdf, gTTS"
    )
    
    st.sidebar.divider()
    
    st.sidebar.subheader("📊 Stats")
    if st.session_state.pdf_file:
        st.sidebar.metric("Cached Audio", len(st.session_state.audio_buffer_cache))
        st.sidebar.metric("Recent Files", len(st.session_state.recent_files))

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    init_session_state()
    
    render_header()
    render_upload_section()
    st.divider()
    render_reader_section()
    
    render_sidebar()

if __name__ == "__main__":
    main()
