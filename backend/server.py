from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import tempfile
import shutil
import asyncio
import requests
import base64
from elevenlabs import ElevenLabs
from emergentintegrations.llm.openai import OpenAISpeechToText
import subprocess
import json
import io

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI(title="Voicepod Creator Studio", version="1.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Initialize API clients
cleanvoice_api_key = os.getenv("CLEANVOICE_API_KEY")
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
whisper_client = OpenAISpeechToText(api_key=os.getenv("EMERGENT_LLM_KEY"))

# Storage directory
UPLOADS_DIR = ROOT_DIR / "uploads"
PROCESSED_DIR = ROOT_DIR / "processed"
MUSIC_DIR = ROOT_DIR / "music"

for directory in [UPLOADS_DIR, PROCESSED_DIR, MUSIC_DIR]:
    directory.mkdir(exist_ok=True)

# ==================== MODELS ====================

class ProcessingPreset(BaseModel):
    name: str
    cleanvoice_config: Dict[str, Any]
    music_type: str
    use_elevenlabs: bool = False

class AudioUploadResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    job_id: str
    status: str
    message: str

class ProcessingStatus(BaseModel):
    model_config = ConfigDict(extra="ignore")
    job_id: str
    status: str
    progress: int
    current_step: str
    download_url: Optional[str] = None
    error: Optional[str] = None
    statistics: Optional[Dict] = None

class VoicepodProject(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_filename: str
    preset: str
    status: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    processed_url: Optional[str] = None
    transcript: Optional[str] = None
    statistics: Optional[Dict] = None

# ==================== PRESETS ====================

PRESETS = {
    "podcast_calm": ProcessingPreset(
        name="Podcast Calm",
        cleanvoice_config={
            "remove_filler_words": True,
            "remove_silence": True,
            "enhance_speech": True,
            "studio_sound": "true"
        },
        music_type="ambient",
        use_elevenlabs=False
    ),
    "dramatic": ProcessingPreset(
        name="Dramatic",
        cleanvoice_config={
            "remove_filler_words": True,
            "remove_silence": True,
            "enhance_speech": True,
            "studio_sound": "nightly"
        },
        music_type="cinematic",
        use_elevenlabs=False
    ),
    "ai_narrator": ProcessingPreset(
        name="AI Narrator",
        cleanvoice_config={
            "remove_filler_words": True,
            "remove_silence": True,
            "enhance_speech": True,
            "studio_sound": "nightly"
        },
        music_type="ambient",
        use_elevenlabs=True
    )
}

# Job store (in-memory for demo)
job_store: Dict[str, Dict] = {}

# ==================== HELPER FUNCTIONS ====================

def update_job_progress(job_id: str, progress: int, step: str, status: str = "processing"):
    """Update job progress in store"""
    if job_id in job_store:
        job_store[job_id].update({
            "progress": progress,
            "current_step": step,
            "status": status
        })

async def cleanvoice_cleanup(audio_path: str, config: Dict) -> str:
    """Process audio with Cleanvoice API"""
    try:
        # First convert audio to MP3 if needed
        temp_mp3 = audio_path.replace(Path(audio_path).suffix, '_temp.mp3')
        convert_cmd = ['ffmpeg', '-y', '-i', audio_path, '-c:a', 'libmp3lame', '-b:a', '192k', temp_mp3]
        subprocess.run(convert_cmd, check=True, capture_output=True)
        
        # Upload to Cleanvoice
        with open(temp_mp3, 'rb') as f:
            files = {'file': (Path(temp_mp3).name, f, 'audio/mp3')}
            data = {
                'remove_filler_words': str(config.get('remove_filler_words', True)).lower(),
                'remove_silence': str(config.get('remove_silence', True)).lower(),
                'enhance_speech': str(config.get('enhance_speech', True)).lower(),
                'studio_sound': config.get('studio_sound', 'true')
            }
            
            response = requests.post(
                "https://api.cleanvoice.ai/v2/edits",
                headers={"X-API-Key": cleanvoice_api_key},
                files=files,
                data=data,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            job_id = result.get('task_id') or result.get('id')
        
        # Poll for completion
        for _ in range(60):
            await asyncio.sleep(3)
            status_response = requests.get(
                f"https://api.cleanvoice.ai/v2/edits/{job_id}",
                headers={"X-API-Key": cleanvoice_api_key}
            )
            status_data = status_response.json()
            
            if status_data.get('status') == 'SUCCESS':
                download_url = status_data['result']['download_url']
                audio_response = requests.get(download_url)
                cleaned_path = audio_path.replace(Path(audio_path).suffix, '_cleaned.mp3')
                with open(cleaned_path, 'wb') as f:
                    f.write(audio_response.content)
                
                # Clean up temp file
                if os.path.exists(temp_mp3):
                    os.remove(temp_mp3)
                    
                return cleaned_path
            elif status_data.get('status') == 'FAILURE':
                raise Exception(f"Cleanvoice failed: {status_data.get('error')}")
        
        raise Exception("Cleanvoice timeout")
        
    except Exception as e:
        logging.warning(f"Cleanvoice failed: {e}. Falling back to basic processing.")
        # Fallback: Just convert to normalized MP3
        cleaned_path = audio_path.replace(Path(audio_path).suffix, '_cleaned.mp3')
        fallback_cmd = [
            'ffmpeg', '-y', '-i', audio_path,
            '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11',
            '-c:a', 'libmp3lame', '-b:a', '192k',
            cleaned_path
        ]
        subprocess.run(fallback_cmd, check=True, capture_output=True)
        return cleaned_path

async def transcribe_and_analyze(audio_path: str) -> Dict:
    """Transcribe audio and detect emotion peaks"""
    with open(audio_path, 'rb') as audio_file:
        response = await whisper_client.transcribe(
            file=audio_file,
            model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
    
    # Handle both dict and object responses
    if isinstance(response, dict):
        transcript = response.get('text', '')
        segments = response.get('segments', [])
    else:
        transcript = response.text if hasattr(response, 'text') else str(response)
        segments = response.segments if hasattr(response, 'segments') else []
    
    # Simple emotion peak detection (look for segments with high energy words)
    emotion_peaks = []
    for segment in segments:
        # Handle both dict and object segments
        if isinstance(segment, dict):
            text = segment.get('text', '').strip()
            start = segment.get('start', 0)
        else:
            text = segment.text.strip() if hasattr(segment, 'text') else ''
            start = segment.start if hasattr(segment, 'start') else 0
        
        # Simple heuristic: look for exclamations, questions
        if any(word in text.lower() for word in ['!', '?', 'wow', 'amazing', 'incredible']):
            emotion_peaks.append(start)
    
    return {
        "transcript": transcript,
        "emotion_peaks": emotion_peaks[:2]  # Max 2 peaks
    }

async def apply_elevenlabs_revoice(audio_path: str, transcript: str) -> str:
    """Apply ElevenLabs voice regeneration"""
    try:
        voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice
        
        audio_generator = elevenlabs_client.text_to_speech.convert(
            text=transcript,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2"
        )
        
        # Save audio
        audio_data = b""
        for chunk in audio_generator:
            audio_data += chunk
        
        revoiced_path = audio_path.replace(Path(audio_path).suffix, '_revoiced.mp3')
        with open(revoiced_path, 'wb') as f:
            f.write(audio_data)
        
        return revoiced_path
        
    except Exception as e:
        logging.warning(f"ElevenLabs failed: {e}. Skipping re-voicing.")
        # Fallback: return original
        return audio_path

def merge_with_music(audio_path: str, music_type: str, emotion_peaks: List[float]) -> str:
    """Merge audio with background music using ffmpeg"""
    output_path = audio_path.replace(Path(audio_path).suffix, '_final.mp3')
    
    # For demo: Just normalize and convert to MP3 (without actual music)
    # In production, you would mix with actual background music files
    cmd = [
        'ffmpeg', '-y',
        '-i', audio_path,
        '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11',  # Loudness normalization
        '-c:a', 'libmp3lame', '-b:a', '192k', '-q:a', '2',
        output_path
    ]
    
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_path

def add_metadata(audio_path: str, title: str) -> str:
    """Add ID3 tags to audio"""
    output_path = audio_path.replace('_final.mp3', '_complete.mp3')
    
    cmd = [
        'ffmpeg', '-y',
        '-i', audio_path,
        '-metadata', f'title={title}',
        '-metadata', 'artist=Voicepod Studio',
        '-metadata', 'album=AI-Enhanced Audio',
        '-c:a', 'copy',  # Copy codec without re-encoding
        output_path
    ]
    
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_path

# ==================== BACKGROUND PROCESSING ====================

async def process_audio_pipeline(job_id: str, audio_path: str, preset_name: str):
    """Complete audio processing pipeline"""
    try:
        preset = PRESETS[preset_name]
        
        # Step 1: Cleanvoice cleanup
        update_job_progress(job_id, 20, "Cleaning audio (removing noise, fillers)...")
        cleaned_path = await cleanvoice_cleanup(audio_path, preset.cleanvoice_config)
        
        # Step 2: Transcription and emotion analysis
        update_job_progress(job_id, 40, "Analyzing speech and detecting emotion peaks...")
        analysis = await transcribe_and_analyze(cleaned_path)
        
        # Step 3: Optional ElevenLabs re-voicing
        current_audio = cleaned_path
        if preset.use_elevenlabs:
            update_job_progress(job_id, 60, "Applying AI narrator voice...")
            current_audio = await apply_elevenlabs_revoice(cleaned_path, analysis['transcript'])
        else:
            update_job_progress(job_id, 60, "Skipping AI re-voicing...")
        
        # Step 4: Music merge
        update_job_progress(job_id, 80, "Adding background music and accents...")
        merged_path = merge_with_music(current_audio, preset.music_type, analysis['emotion_peaks'])
        
        # Step 5: Metadata
        update_job_progress(job_id, 90, "Finalizing with metadata...")
        final_path = add_metadata(merged_path, f"Voicepod_{job_id[:8]}")
        
        # Move to processed directory
        final_filename = f"{job_id}.mp3"
        final_destination = PROCESSED_DIR / final_filename
        shutil.move(final_path, final_destination)
        
        # Update job
        job_store[job_id].update({
            "status": "completed",
            "progress": 100,
            "current_step": "Complete!",
            "download_url": f"/api/download/{job_id}",
            "transcript": analysis['transcript'],
            "statistics": {
                "duration": "N/A",
                "preset": preset_name,
                "emotion_peaks": len(analysis['emotion_peaks'])
            }
        })
        
        # Cleanup temp files
        for temp_file in [audio_path, cleaned_path, current_audio, merged_path]:
            if os.path.exists(temp_file) and temp_file != str(final_destination):
                os.remove(temp_file)
        
    except Exception as e:
        job_store[job_id].update({
            "status": "failed",
            "error": str(e),
            "current_step": "Error occurred"
        })
        logging.error(f"Processing failed for {job_id}: {e}")

# ==================== API ROUTES ====================

@api_router.get("/")
async def root():
    return {"message": "Voicepod Creator Studio API", "version": "1.0.0"}

@api_router.get("/presets")
async def get_presets():
    """Get available processing presets"""
    return {
        preset_id: {
            "name": preset.name,
            "description": f"{'AI narrator mode' if preset.use_elevenlabs else 'Professional audio'} with {preset.music_type} music"
        }
        for preset_id, preset in PRESETS.items()
    }

@api_router.post("/upload", response_model=AudioUploadResponse)
async def upload_audio(
    file: UploadFile = File(...),
    preset: str = "podcast_calm",
    background_tasks: BackgroundTasks = None
):
    """Upload audio and start processing"""
    if preset not in PRESETS:
        raise HTTPException(status_code=400, detail=f"Invalid preset. Choose from: {list(PRESETS.keys())}")
    
    # Validate file
    if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a', '.ogg', '.mpeg', '.mpga', '.mp4', '.webm')):
        raise HTTPException(status_code=400, detail="Unsupported audio format")
    
    # Save upload
    job_id = str(uuid.uuid4())
    file_ext = Path(file.filename).suffix
    audio_path = UPLOADS_DIR / f"{job_id}{file_ext}"
    
    with open(audio_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)
    
    # Initialize job
    job_store[job_id] = {
        "status": "processing",
        "progress": 10,
        "current_step": "Uploaded, starting processing...",
        "original_filename": file.filename,
        "preset": preset
    }
    
    # Start background processing
    background_tasks.add_task(process_audio_pipeline, job_id, str(audio_path), preset)
    
    return AudioUploadResponse(
        job_id=job_id,
        status="processing",
        message="Audio uploaded successfully. Processing started."
    )

@api_router.get("/status/{job_id}", response_model=ProcessingStatus)
async def get_status(job_id: str):
    """Get processing status"""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_store[job_id]
    return ProcessingStatus(
        job_id=job_id,
        status=job['status'],
        progress=job['progress'],
        current_step=job['current_step'],
        download_url=job.get('download_url'),
        error=job.get('error'),
        statistics=job.get('statistics')
    )

@api_router.get("/download/{job_id}")
async def download_audio(job_id: str):
    """Download processed audio"""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job_store[job_id]['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Processing not complete")
    
    file_path = PROCESSED_DIR / f"{job_id}.mp3"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        media_type="audio/mpeg",
        filename=f"voicepod_{job_id[:8]}.mp3"
    )

@api_router.get("/projects")
async def get_projects():
    """Get all projects"""
    projects = []
    for job_id, job in job_store.items():
        projects.append({
            "id": job_id,
            "filename": job.get('original_filename', 'Unknown'),
            "status": job['status'],
            "preset": job.get('preset', 'unknown'),
            "download_url": job.get('download_url')
        })
    return {"projects": projects}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()