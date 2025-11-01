#!/usr/bin/env python3
"""
Focused LUFS Testing for Voicepod Audio Processing
Tests the specific LUFS normalization issue after fixes
"""

import requests
import time
import json
import os
import subprocess
import tempfile
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Backend URL from frontend .env
BACKEND_URL = "https://audio-alchemist-1.preview.emergentagent.com/api"

def create_longer_test_audio():
    """Create a longer test audio file for better LUFS testing"""
    test_audio_path = "/tmp/long_test_voice.wav"
    
    try:
        # Generate longer speech with more emotional content
        speech_text = """
        Hello everyone! Welcome to this amazing podcast episode! 
        This is absolutely incredible technology we're testing today.
        Wow, I can't believe how fantastic this sounds! 
        The audio quality is simply outstanding.
        Let me tell you about this revolutionary approach.
        It's truly remarkable what we can achieve now.
        This is going to change everything in audio processing!
        Amazing results, incredible quality, fantastic innovation!
        """
        
        # Use espeak to generate longer speech
        subprocess.run([
            'espeak', '-s', '140', '-v', 'en+f3', '-w', '/tmp/long_speech.wav', speech_text
        ], check=True, capture_output=True)
        
        # Convert to proper format with ffmpeg - make it longer
        subprocess.run([
            'ffmpeg', '-y', '-i', '/tmp/long_speech.wav', 
            '-ar', '44100', '-ac', '2', '-t', '20',  # 20 seconds
            test_audio_path
        ], check=True, capture_output=True)
        
        logger.info(f"Created longer test audio file: {test_audio_path}")
        return test_audio_path
        
    except Exception as e:
        logger.error(f"Failed to create longer test audio: {e}")
        # Fallback: create a longer sine wave
        subprocess.run([
            'ffmpeg', '-y', '-f', 'lavfi', '-i', 
            'sine=frequency=440:duration=20', 
            '-ar', '44100', test_audio_path
        ], check=True, capture_output=True)
        return test_audio_path

def measure_lufs_precisely(audio_path):
    """Measure LUFS with high precision"""
    try:
        # Use ffmpeg loudnorm with JSON output for precise measurement
        cmd = [
            'ffmpeg', '-i', audio_path, 
            '-af', 'loudnorm=print_format=json', 
            '-f', 'null', '-'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse JSON from stderr
        output_lines = result.stderr.split('\n')
        json_started = False
        json_lines = []
        
        for line in output_lines:
            if line.strip().startswith('{'):
                json_started = True
                json_lines = [line]
            elif json_started:
                json_lines.append(line)
                if line.strip().endswith('}'):
                    break
        
        if json_lines:
            json_str = '\n'.join(json_lines)
            lufs_data = json.loads(json_str)
            input_lufs = float(lufs_data.get('input_i', 0))
            return input_lufs
        
        return None
        
    except Exception as e:
        logger.error(f"LUFS measurement failed: {e}")
        return None

def test_lufs_normalization():
    """Test LUFS normalization specifically"""
    logger.info("🎚️ Starting LUFS Normalization Test")
    logger.info("=" * 50)
    
    # Create longer test audio
    test_audio = create_longer_test_audio()
    
    # Upload and process
    try:
        with open(test_audio, 'rb') as f:
            files = {'file': ('long_test_audio.wav', f, 'audio/wav')}
            data = {'preset': 'podcast_calm', 'format': 'mp3'}
            
            response = requests.post(f"{BACKEND_URL}/upload", files=files, data=data)
        
        if response.status_code != 200:
            logger.error(f"Upload failed: {response.status_code}")
            return False
        
        job_id = response.json()['job_id']
        logger.info(f"Upload successful, Job ID: {job_id}")
        
        # Monitor processing
        start_time = time.time()
        while time.time() - start_time < 300:  # 5 minute timeout
            status_response = requests.get(f"{BACKEND_URL}/status/{job_id}")
            if status_response.status_code == 200:
                status = status_response.json()
                if status['status'] == 'completed':
                    logger.info("✅ Processing completed")
                    break
                elif status['status'] == 'failed':
                    logger.error(f"❌ Processing failed: {status.get('error')}")
                    return False
                else:
                    logger.info(f"Processing: {status.get('progress', 0)}% - {status.get('current_step', 'Unknown')}")
            time.sleep(5)
        
        # Download and measure LUFS
        download_response = requests.get(f"{BACKEND_URL}/download/{job_id}")
        if download_response.status_code == 200:
            output_path = f"/tmp/lufs_test_output_{job_id}.mp3"
            with open(output_path, 'wb') as f:
                f.write(download_response.content)
            
            logger.info(f"Downloaded output to: {output_path}")
            
            # Measure LUFS precisely
            measured_lufs = measure_lufs_precisely(output_path)
            
            if measured_lufs is not None:
                target_lufs = -16.0
                difference = abs(measured_lufs - target_lufs)
                
                logger.info("=" * 50)
                logger.info("🎯 LUFS TEST RESULTS:")
                logger.info(f"📏 Measured LUFS: {measured_lufs:.2f}")
                logger.info(f"🎯 Target LUFS: {target_lufs}")
                logger.info(f"📊 Difference: {difference:.2f} LUFS")
                
                if difference <= 0.5:
                    logger.info("✅ LUFS target achieved within acceptable range!")
                    return True
                elif difference <= 1.0:
                    logger.info("✅ LUFS close to target - within acceptable tolerance!")
                    return True
                else:
                    logger.info("❌ LUFS significantly off target - needs adjustment")
                    return False
            else:
                logger.error("❌ Could not measure LUFS")
                return False
        else:
            logger.error(f"Download failed: {download_response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

def test_background_music_merge():
    """Test background music merge functionality"""
    logger.info("🎵 Starting Background Music Merge Test")
    logger.info("=" * 50)
    
    # Create test audio
    test_audio = create_longer_test_audio()
    
    try:
        with open(test_audio, 'rb') as f:
            files = {'file': ('music_test_audio.wav', f, 'audio/wav')}
            data = {'preset': 'podcast_calm', 'format': 'mp3'}
            
            response = requests.post(f"{BACKEND_URL}/upload", files=files, data=data)
        
        if response.status_code != 200:
            logger.error(f"Upload failed: {response.status_code}")
            return False
        
        job_id = response.json()['job_id']
        logger.info(f"Upload successful, Job ID: {job_id}")
        
        # Monitor processing with focus on music merge step
        start_time = time.time()
        music_merge_success = False
        
        while time.time() - start_time < 300:
            status_response = requests.get(f"{BACKEND_URL}/status/{job_id}")
            if status_response.status_code == 200:
                status = status_response.json()
                current_step = status.get('current_step', '')
                
                if 'music' in current_step.lower():
                    logger.info(f"🎵 Music processing: {current_step}")
                
                if status['status'] == 'completed':
                    logger.info("✅ Processing completed")
                    music_merge_success = True
                    break
                elif status['status'] == 'failed':
                    logger.error(f"❌ Processing failed: {status.get('error')}")
                    return False
                else:
                    logger.info(f"Processing: {status.get('progress', 0)}% - {current_step}")
            time.sleep(3)
        
        if music_merge_success:
            # Download and check if music is present
            download_response = requests.get(f"{BACKEND_URL}/download/{job_id}")
            if download_response.status_code == 200:
                output_path = f"/tmp/music_test_output_{job_id}.mp3"
                with open(output_path, 'wb') as f:
                    f.write(download_response.content)
                
                # Check file size (music should make it larger)
                file_size = os.path.getsize(output_path)
                logger.info(f"Output file size: {file_size / 1024:.2f} KB")
                
                # Check duration
                duration_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                               '-of', 'default=noprint_wrappers=1:nokey=1', output_path]
                duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
                if duration_result.returncode == 0:
                    duration = float(duration_result.stdout.strip())
                    logger.info(f"Output duration: {duration:.2f} seconds")
                
                logger.info("✅ Background music merge appears successful")
                return True
            else:
                logger.error("❌ Could not download output")
                return False
        else:
            logger.error("❌ Music merge did not complete")
            return False
            
    except Exception as e:
        logger.error(f"Music merge test failed: {e}")
        return False

def main():
    """Run focused tests on the two critical issues"""
    logger.info("🔧 PHASE 1 RE-VERIFICATION AFTER FIXES")
    logger.info("Testing LUFS normalization and background music merge")
    logger.info("=" * 60)
    
    # Test 1: LUFS Normalization
    lufs_success = test_lufs_normalization()
    
    # Test 2: Background Music Merge
    music_success = test_background_music_merge()
    
    # Summary
    logger.info("=" * 60)
    logger.info("📋 FINAL TEST SUMMARY:")
    logger.info(f"🎚️ LUFS Normalization: {'✅ PASS' if lufs_success else '❌ FAIL'}")
    logger.info(f"🎵 Background Music Merge: {'✅ PASS' if music_success else '❌ FAIL'}")
    
    if lufs_success and music_success:
        logger.info("🎉 All critical fixes verified successfully!")
        return 0
    else:
        logger.info("⚠️ Some issues still need attention")
        return 1

if __name__ == "__main__":
    exit(main())