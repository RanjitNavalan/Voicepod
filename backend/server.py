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
    """Process audio with Demucs for high-quality vocal extraction and enhancement"""
    cleaned_path = audio_path.replace(Path(audio_path).suffix, '_cleaned.mp3')
    
    try:
        # Use Demucs for vocal separation and enhancement
        logging.info("Using Demucs for high-quality vocal extraction...")
        
        import torch
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        import torchaudio
        
        # Load Demucs model (htdemucs is the best for vocals)
        model = get_model('htdemucs')
        model.eval()
        
        # Convert to WAV for Demucs
        temp_wav = audio_path.replace(Path(audio_path).suffix, '_temp.wav')
        convert_cmd = ['ffmpeg', '-y', '-i', audio_path, '-ar', '44100', '-ac', '2', temp_wav]
        subprocess.run(convert_cmd, check=True, capture_output=True)
        
        # Load audio
        wav, sr = torchaudio.load(temp_wav)
        
        # Ensure correct sample rate
        if sr != model.samplerate:
            wav = torchaudio.functional.resample(wav, sr, model.samplerate)
        
        # Apply Demucs to separate vocals
        logging.info("Separating vocals with Demucs (this may take a moment)...")
        with torch.no_grad():
            sources = apply_model(model, wav[None], device='cpu')[0]
        
        # Extract vocals (index 3 in htdemucs: drums, bass, other, vocals)
        vocals = sources[3]  # Just the vocals, no background noise
        
        # Save vocals
        vocals_wav = temp_wav.replace('_temp.wav', '_vocals.wav')
        torchaudio.save(vocals_wav, vocals.cpu(), model.samplerate)
        
        # Apply noise reduction on vocals for extra clarity
        logging.info("Applying light noise reduction to vocals...")
        try:
            import noisereduce as nr
            import librosa
            import soundfile as sf
            
            audio_data, sample_rate = librosa.load(vocals_wav, sr=None, mono=False)
            
            # VERY LIGHT noise reduction (Demucs already did the work)
            reduced_noise = nr.reduce_noise(
                y=audio_data,
                sr=sample_rate,
                stationary=True,
                prop_decrease=0.3,  # Only 30% reduction - very light
                freq_mask_smooth_hz=300,
                time_mask_smooth_ms=30
            )
            
            enhanced_wav = vocals_wav.replace('_vocals.wav', '_enhanced.wav')
            sf.write(enhanced_wav, reduced_noise.T if len(reduced_noise.shape) > 1 else reduced_noise, sample_rate)
            final_input = enhanced_wav
        except Exception as nr_error:
            logging.warning(f"Light noise reduction failed: {nr_error}, using Demucs vocals directly")
            final_input = vocals_wav
        
        # Convert to MP3 with MINIMAL processing to preserve Demucs quality
        # Demucs already did the heavy lifting - just normalize and export
        logging.info("Applying minimal post-processing to preserve Demucs quality...")
        
        final_cmd = [
            'ffmpeg', '-y', '-i', final_input,
            '-af', 
            # MINIMAL filters to preserve Demucs quality:
            'silenceremove=start_periods=1:start_duration=0.3:start_threshold=-50dB:'
            'stop_periods=-1:stop_duration=0.5:stop_threshold=-50dB,'  # Remove silence only
            'loudnorm=I=-16:TP=-1.5:LRA=11',  # Normalize to -16 LUFS
            '-c:a', 'libmp3lame', '-b:a', '192k', '-q:a', '2',
            cleaned_path
        ]
        subprocess.run(final_cmd, check=True, capture_output=True)
        
        # Cleanup
        for temp_file in [temp_wav, vocals_wav, enhanced_wav]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        logging.info("Demucs vocal extraction completed successfully")
        return cleaned_path
        
    except Exception as e:
        logging.warning(f"Demucs processing failed: {e}. Trying Lightning AI fallback...")
        
        # Fallback 1: Lightning AI API
        try:
            LIGHTNING_AI_ID = "9b2560a5-efb2-4748-a0f0-98f525fd63bd"
            api_url = f"https://lightning.ai/lightning-ai/api/{LIGHTNING_AI_ID}"
            
            temp_wav = audio_path.replace(Path(audio_path).suffix, '_temp.wav')
            convert_cmd = ['ffmpeg', '-y', '-i', audio_path, '-ar', '16000', '-ac', '1', temp_wav]
            subprocess.run(convert_cmd, check=True, capture_output=True)
            
            with open(temp_wav, 'rb') as f:
                response = requests.post(api_url, files={'audio': f}, timeout=60)
                
                if response.status_code == 200:
                    enhanced_wav = temp_wav.replace('_temp.wav', '_enhanced.wav')
                    with open(enhanced_wav, 'wb') as ef:
                        ef.write(response.content)
                    
                    final_cmd = [
                        'ffmpeg', '-y', '-i', enhanced_wav,
                        '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11',  # -16 LUFS target
                        '-c:a', 'libmp3lame', '-b:a', '192k',
                        cleaned_path
                    ]
                    subprocess.run(final_cmd, check=True, capture_output=True)
                    
                    for temp_file in [temp_wav, enhanced_wav]:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    
                    logging.info("Lightning AI processing completed")
                    return cleaned_path
        except Exception as lightning_error:
            logging.warning(f"Lightning AI also failed: {lightning_error}. Using noisereduce fallback...")
        
        # Fallback 2: noisereduce + SOX
        try:
            import noisereduce as nr
            import librosa
            import soundfile as sf
            
            audio_data, sample_rate = librosa.load(audio_path, sr=None, mono=False)
            
            reduced_noise = nr.reduce_noise(
                y=audio_data,
                sr=sample_rate,
                stationary=True,
                prop_decrease=0.9,
                freq_mask_smooth_hz=1000,
                time_mask_smooth_ms=100
            )
            
            temp_wav = cleaned_path.replace('.mp3', '_temp.wav')
            sf.write(temp_wav, reduced_noise.T if len(reduced_noise.shape) > 1 else reduced_noise, sample_rate)
            
            sox_processed = temp_wav.replace('_temp.wav', '_sox.wav')
            sox_cmd = ['sox', temp_wav, sox_processed, 'highpass', '100', 'lowpass', '8000', 
                      'compand', '0.3,1', '6:-70,-60,-20', '-5', '-90', '0.2', 'norm', '-3']
            sox_result = subprocess.run(sox_cmd, capture_output=True, text=True)
            
            final_input = sox_processed if sox_result.returncode == 0 else temp_wav
            
            final_cmd = [
                'ffmpeg', '-y', '-i', final_input,
                '-af', 
                'highpass=f=80,'
                'lowpass=f=10000,'
                'adeclick=w=5:t=2,'  # Click removal
                'adeclip,'  # Declip
                'silenceremove=start_periods=1:start_duration=0.2:start_threshold=-50dB:'
                'stop_periods=-1:stop_duration=0.5:stop_threshold=-50dB,'
                'loudnorm=I=-16:TP=-1.5:LRA=11,'  # -16 LUFS target
                'acompressor=threshold=-18dB:ratio=3:attack=5:release=50',
                '-c:a', 'libmp3lame', '-b:a', '192k',
                cleaned_path
            ]
            subprocess.run(final_cmd, check=True, capture_output=True)
            
            for temp_file in [temp_wav, sox_processed]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            logging.info("Fallback noise reduction completed")
            return cleaned_path
            
        except Exception as final_error:
            logging.error(f"All methods failed: {final_error}")
            simple_cmd = [
                'ffmpeg', '-y', '-i', audio_path,
                '-af', 'highpass=f=100,lowpass=f=8000,adeclick,adeclip,afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11',
                '-c:a', 'libmp3lame', '-b:a', '192k',
                cleaned_path
            ]
            subprocess.run(simple_cmd, check=True, capture_output=True)
            return cleaned_path

async def transcribe_and_analyze(audio_path: str) -> Dict:
    """Transcribe audio and detect emotion peaks"""
    with open(audio_path, 'rb') as audio_file:
        response = await whisper_client.transcribe(
            file=audio_file,
            model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["segment"]  # Only segment, word-level not always available
        )
    
    # Handle both dict and object responses
    if isinstance(response, dict):
        transcript = response.get('text', '')
        segments = response.get('segments', [])
    else:
        transcript = response.text if hasattr(response, 'text') else str(response)
        segments = response.segments if hasattr(response, 'segments') else []
    
    # Detect filler words from transcript (simpler approach)
    filler_words = ['um', 'uh', 'like', 'you know', 'so', 'actually', 'basically', 'literally']
    filler_timestamps = []
    
    # Try to detect fillers from segments
    if segments:
        for segment in segments:
            if isinstance(segment, dict):
                text = segment.get('text', '').lower()
                start = segment.get('start', 0)
                end = segment.get('end', 0)
            else:
                text = (segment.text if hasattr(segment, 'text') else '').lower()
                start = segment.start if hasattr(segment, 'start') else 0
                end = segment.end if hasattr(segment, 'end') else 0
            
            # Check if segment contains filler words
            for filler in filler_words:
                if filler in text:
                    filler_timestamps.append((start, end))
                    break
    
    # Simple emotion peak detection
    emotion_peaks = []
    if segments:
        for segment in segments:
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
        "emotion_peaks": emotion_peaks[:2] if emotion_peaks else [],
        "filler_timestamps": filler_timestamps[:10] if filler_timestamps else []
    }

def remove_filler_words(audio_path: str, filler_timestamps: List[tuple]) -> str:
    """Remove filler words from audio using timestamps"""
    if not filler_timestamps or len(filler_timestamps) == 0:
        logging.info("No filler words detected, skipping removal")
        return audio_path
    
    output_path = audio_path.replace(Path(audio_path).suffix, '_no_fillers.mp3')
    
    try:
        logging.info(f"Removing {len(filler_timestamps)} filler word segments...")
        
        # Get total duration
        duration_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                       '-of', 'default=noprint_wrappers=1:nokey=1', audio_path]
        duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
        total_duration = float(duration_result.stdout.strip())
        
        # Build keep segments (everything EXCEPT filler timestamps)
        keep_segments = []
        current_time = 0.0
        
        for start, end in sorted(filler_timestamps):
            # Add segment before filler
            if current_time < start:
                keep_segments.append((current_time, start))
            current_time = end
        
        # Add final segment after last filler
        if current_time < total_duration:
            keep_segments.append((current_time, total_duration))
        
        # If no segments to keep, return original
        if not keep_segments:
            logging.warning("No segments to keep after filler removal, returning original")
            return audio_path
        
        # Extract and concatenate segments
        temp_segments = []
        for i, (start, end) in enumerate(keep_segments):
            segment_path = audio_path.replace(Path(audio_path).suffix, f'_seg{i}.mp3')
            extract_cmd = [
                'ffmpeg', '-y',
                '-i', audio_path,
                '-ss', str(start),
                '-to', str(end),
                '-c', 'copy',
                segment_path
            ]
            subprocess.run(extract_cmd, check=True, capture_output=True)
            temp_segments.append(segment_path)
        
        # Concatenate segments
        if len(temp_segments) == 1:
            shutil.move(temp_segments[0], output_path)
        else:
            # Create concat file
            concat_file = audio_path.replace(Path(audio_path).suffix, '_concat.txt')
            with open(concat_file, 'w') as f:
                for seg in temp_segments:
                    f.write(f"file '{seg}'\n")
            
            concat_cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c', 'copy',
                output_path
            ]
            subprocess.run(concat_cmd, check=True, capture_output=True)
            
            # Cleanup
            os.remove(concat_file)
        
        # Cleanup temp segments
        for seg in temp_segments:
            if os.path.exists(seg):
                os.remove(seg)
        
        logging.info(f"Successfully removed {len(filler_timestamps)} filler words")
        return output_path
        
    except Exception as e:
        logging.error(f"Filler word removal failed: {e}")
        # Return original if removal fails
        if os.path.exists(output_path):
            return output_path
        return audio_path

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
    """Add professional background music with smart ducking and intro/outro"""
    output_path = audio_path.replace(Path(audio_path).suffix, '_final.mp3')
    
    try:
        # Select professional music tracks
        music_map = {
            'ambient': MUSIC_DIR / 'ambient' / 'arietta.mp3',  # Calm, professional
            'cinematic': MUSIC_DIR / 'cinematic' / 'epic_journey.mp3'  # Dramatic
        }
        music_file = music_map.get(music_type, music_map['ambient'])
        intro_file = MUSIC_DIR / 'stingers' / 'intro_smooth.mp3'
        outro_file = MUSIC_DIR / 'stingers' / 'outro_smooth.mp3'
        
        if not music_file.exists():
            raise Exception(f"Music file not found: {music_file}")
        
        logging.info(f"Adding professional background music: {music_file.name}")
        
        # Get voice duration
        duration_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                       '-of', 'default=noprint_wrappers=1:nokey=1', audio_path]
        duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
        voice_duration = float(duration_result.stdout.strip())
        
        # IMPROVED FILTER: Smart ducking based on voice presence
        filter_complex = (
            # === VOICE PROCESSING ===
            '[0:a]aformat=sample_rates=48000:channel_layouts=stereo[voice_formatted];'
            
            # === MUSIC PROCESSING ===
            # Trim music to voice length + add fades
            f'[1:a]atrim=0:{voice_duration},'
            'aformat=sample_rates=48000:channel_layouts=stereo,'
            'volume=0.12,'  # Base music volume (12%)
            'afade=t=in:st=0:d=3,'  # Smooth 3s fade in
            f'afade=t=out:st={voice_duration-4}:d=4[music_base];'  # Smooth 4s fade out
            
            # === SMART DUCKING ===
            # Detect voice presence and duck music intelligently
            '[voice_formatted]asplit=2[voice_for_duck][voice_clean];'
            '[voice_for_duck]acompressor=threshold=-25dB:ratio=6:attack=20:release=200[voice_trigger];'
            '[music_base][voice_trigger]sidechaincompress=threshold=-40dB:ratio=3:attack=100:release=400:makeup=1[music_ducked];'
            
            # === MIX VOICE + DUCKED MUSIC ===
            '[voice_clean][music_ducked]amix=inputs=2:duration=first:dropout_transition=3[voice_music];'
        )
        
        # Add intro/outro if they exist
        if intro_file.exists() and outro_file.exists():
            filter_complex += (
                # === INTRO/OUTRO ===
                '[2:a]volume=0.5,afade=t=out:st=0.8:d=0.4[intro];'
                '[3:a]volume=0.5,afade=t=in:st=0:d=0.5[outro];'
                
                # === CONCATENATE ===
                '[intro][voice_music][outro]concat=n=3:v=0:a=1[mixed];'
                
                # === FINAL NORMALIZATION ===
                '[mixed]loudnorm=I=-16:TP=-1.5:LRA=11'
            )
            
            cmd = [
                'ffmpeg', '-y',
                '-i', audio_path,        # 0: voice
                '-i', str(music_file),   # 1: music
                '-i', str(intro_file),   # 2: intro
                '-i', str(outro_file),   # 3: outro
                '-filter_complex', filter_complex,
                '-c:a', 'libmp3lame', '-b:a', '192k', '-q:a', '2',
                output_path
            ]
        else:
            # No stingers, just voice + music
            filter_complex += '[voice_music]loudnorm=I=-16:TP=-1.5:LRA=11'
            
            cmd = [
                'ffmpeg', '-y',
                '-i', audio_path,        # 0: voice
                '-i', str(music_file),   # 1: music
                '-filter_complex', filter_complex,
                '-c:a', 'libmp3lame', '-b:a', '192k', '-q:a', '2',
                output_path
            ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logging.error(f"Music merge failed: {result.stderr[:500]}")
            raise Exception("Music merge failed")
        
        logging.info("Professional music with smart ducking added successfully")
        return output_path
        
    except Exception as e:
        logging.warning(f"Music addition failed: {e}. Keeping clean voice only.")
        # Fallback: just normalize without music (preserve clean audio)
        fallback_cmd = [
            'ffmpeg', '-y', '-i', audio_path,
            '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11',
            '-c:a', 'libmp3lame', '-b:a', '192k',
            output_path
        ]
        subprocess.run(fallback_cmd, check=True, capture_output=True)
        logging.info("Fallback: clean voice without music")
        return output_path

def add_metadata(audio_path: str, title: str, format: str = 'mp3') -> str:
    """Add ID3 tags and cover image to audio"""
    ext = '.m4a' if format == 'm4a' else '.mp3'
    output_path = audio_path.replace('_final.mp3', f'_complete{ext}')
    cover_image = MUSIC_DIR / 'cover.jpg'
    
    try:
        if format == 'm4a':
            # M4A export with cover art
            cmd = [
                'ffmpeg', '-y',
                '-i', audio_path,
                '-i', str(cover_image),
                '-map', '0:a',
                '-map', '1:v',
                '-c:a', 'aac', '-b:a', '192k',
                '-c:v', 'copy',
                '-disposition:v:0', 'attached_pic',
                '-metadata', f'title={title}',
                '-metadata', 'artist=Voicepod Studio',
                '-metadata', 'album=AI-Enhanced Audio',
                output_path
            ]
        else:
            # MP3 export with cover art
            cmd = [
                'ffmpeg', '-y',
                '-i', audio_path,
                '-i', str(cover_image),
                '-map', '0:a',
                '-map', '1:v',
                '-c:a', 'copy',
                '-c:v', 'copy',
                '-id3v2_version', '3',
                '-metadata:s:v', 'title=Album cover',
                '-metadata:s:v', 'comment=Cover (front)',
                '-metadata', f'title={title}',
                '-metadata', 'artist=Voicepod Studio',
                '-metadata', 'album=AI-Enhanced Audio',
                output_path
            ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Metadata embedding failed: {result.stderr}")
            
        logging.info(f"Metadata and cover art added successfully ({format})")
        return output_path
        
    except Exception as e:
        logging.error(f"Metadata addition failed: {e}")
        # Fallback: simple copy
        shutil.copy(audio_path, output_path)
        return output_path

# ==================== BACKGROUND PROCESSING ====================

async def process_audio_pipeline(job_id: str, audio_path: str, preset_name: str, export_format: str = 'mp3'):
    """Complete audio processing pipeline"""
    try:
        preset = PRESETS[preset_name]
        
        # Step 1: AI Noise Reduction
        update_job_progress(job_id, 15, "Cleaning audio with AI noise reduction...")
        cleaned_path = await cleanvoice_cleanup(audio_path, preset.cleanvoice_config)
        
        # Step 2: Transcription and analysis
        update_job_progress(job_id, 35, "Analyzing speech, detecting fillers and emotion peaks...")
        analysis = await transcribe_and_analyze(cleaned_path)
        
        # Step 3: Filler word removal (if detected)
        update_job_progress(job_id, 50, "Removing filler words...")
        current_audio = remove_filler_words(cleaned_path, analysis.get('filler_timestamps', []))
        
        # Step 4: Optional ElevenLabs re-voicing
        if preset.use_elevenlabs:
            update_job_progress(job_id, 60, "Applying AI narrator voice...")
            current_audio = await apply_elevenlabs_revoice(current_audio, analysis['transcript'])
        else:
            update_job_progress(job_id, 60, "Skipping AI re-voicing...")
        
        # Step 5: Final normalization
        update_job_progress(job_id, 75, "Normalizing audio levels...")
        merged_path = merge_with_music(current_audio, preset.music_type, analysis['emotion_peaks'])
        
        # Step 6: Metadata and cover art
        update_job_progress(job_id, 90, "Adding metadata and cover art...")
        final_path = add_metadata(merged_path, f"Voicepod_{job_id[:8]}", export_format)
        
        # Move to processed directory
        file_ext = '.m4a' if export_format == 'm4a' else '.mp3'
        final_filename = f"{job_id}{file_ext}"
        final_destination = PROCESSED_DIR / final_filename
        shutil.move(final_path, final_destination)
        
        # Update job
        job_store[job_id].update({
            "status": "completed",
            "progress": 100,
            "current_step": "Complete!",
            "download_url": f"/api/download/{job_id}",
            "transcript": analysis['transcript'],
            "format": export_format,
            "statistics": {
                "duration": "N/A",
                "preset": preset_name,
                "emotion_peaks": len(analysis['emotion_peaks']),
                "fillers_detected": len(analysis.get('filler_timestamps', []))
            }
        })
        
        # Cleanup temp files
        for temp_file in [audio_path, cleaned_path, current_audio, merged_path]:
            if os.path.exists(temp_file) and temp_file != str(final_destination):
                try:
                    os.remove(temp_file)
                except:
                    pass
        
    except Exception as e:
        job_store[job_id].update({
            "status": "failed",
            "error": str(e),
            "current_step": "Error occurred"
        })
        logging.error(f"Processing failed for {job_id}: {e}")

# ==================== API ROUTES ====================
        
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
    format: str = "mp3",
    background_tasks: BackgroundTasks = None
):
    """Upload audio and start processing"""
    if preset not in PRESETS:
        raise HTTPException(status_code=400, detail=f"Invalid preset. Choose from: {list(PRESETS.keys())}")
    
    if format not in ['mp3', 'm4a']:
        raise HTTPException(status_code=400, detail="Format must be 'mp3' or 'm4a'")
    
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
        "preset": preset,
        "format": format
    }
    
    # Start background processing
    background_tasks.add_task(process_audio_pipeline, job_id, str(audio_path), preset, format)
    
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
    
    # Check format
    format = job_store[job_id].get('format', 'mp3')
    file_ext = '.m4a' if format == 'm4a' else '.mp3'
    file_path = PROCESSED_DIR / f"{job_id}{file_ext}"
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    media_type = "audio/mp4" if format == 'm4a' else "audio/mpeg"
    
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=f"voicepod_{job_id[:8]}{file_ext}"
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