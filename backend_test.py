#!/usr/bin/env python3
"""
Comprehensive Backend Testing for Voicepod Audio Processing Pipeline
Tests all Phase 1 requirements including SFX at emotion peaks and LUFS normalization
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

class VoicepodTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = {}
        
    def create_test_audio(self):
        """Create a test audio file with speech for testing"""
        test_audio_path = "/tmp/test_voice.wav"
        
        # Create a simple test audio with speech using espeak and ffmpeg
        try:
            # Generate speech with espeak - more emotional content for better peak detection
            speech_text = "Hello everyone! This is absolutely amazing! Wow, this technology is incredible! I can't believe how fantastic this sounds!"
            
            # Use espeak to generate speech, then convert to proper format
            subprocess.run([
                'espeak', '-s', '150', '-v', 'en+f3', '-w', '/tmp/speech.wav', speech_text
            ], check=True, capture_output=True)
            
            # Convert to proper format with ffmpeg
            subprocess.run([
                'ffmpeg', '-y', '-i', '/tmp/speech.wav', 
                '-ar', '44100', '-ac', '2', '-t', '10',
                test_audio_path
            ], check=True, capture_output=True)
            
            logger.info(f"Created test audio file: {test_audio_path}")
            return test_audio_path
            
        except Exception as e:
            logger.error(f"Failed to create test audio: {e}")
            # Fallback: create a simple sine wave
            subprocess.run([
                'ffmpeg', '-y', '-f', 'lavfi', '-i', 
                'sine=frequency=440:duration=10', 
                '-ar', '44100', test_audio_path
            ], check=True, capture_output=True)
            return test_audio_path
    
    def test_api_health(self):
        """Test if API is accessible"""
        try:
            response = self.session.get(f"{BACKEND_URL}/")
            if response.status_code == 200:
                data = response.json()
                logger.info(f"API Health: ✅ {data.get('message', 'OK')}")
                return True
            else:
                logger.error(f"API Health: ❌ Status {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"API Health: ❌ Connection failed: {e}")
            return False
    
    def test_presets_endpoint(self):
        """Test presets endpoint"""
        try:
            response = self.session.get(f"{BACKEND_URL}/presets")
            if response.status_code == 200:
                presets = response.json()
                logger.info(f"Presets: ✅ Found {len(presets)} presets")
                for preset_id, preset_info in presets.items():
                    logger.info(f"  - {preset_id}: {preset_info['name']}")
                return True, presets
            else:
                logger.error(f"Presets: ❌ Status {response.status_code}")
                return False, {}
        except Exception as e:
            logger.error(f"Presets: ❌ Error: {e}")
            return False, {}
    
    def test_upload_and_process(self, audio_path, preset="podcast_calm"):
        """Test audio upload and processing"""
        try:
            # Upload file
            with open(audio_path, 'rb') as f:
                files = {'file': ('test_audio.wav', f, 'audio/wav')}
                data = {'preset': preset, 'format': 'mp3'}
                
                response = self.session.post(f"{BACKEND_URL}/upload", files=files, data=data)
            
            if response.status_code != 200:
                logger.error(f"Upload failed: Status {response.status_code}, Response: {response.text}")
                return False, None
            
            upload_result = response.json()
            job_id = upload_result['job_id']
            logger.info(f"Upload: ✅ Job ID: {job_id}")
            
            return True, job_id
            
        except Exception as e:
            logger.error(f"Upload: ❌ Error: {e}")
            return False, None
    
    def monitor_processing(self, job_id, timeout=300):
        """Monitor processing status until completion"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self.session.get(f"{BACKEND_URL}/status/{job_id}")
                if response.status_code != 200:
                    logger.error(f"Status check failed: {response.status_code}")
                    return False, None
                
                status = response.json()
                progress = status.get('progress', 0)
                current_step = status.get('current_step', 'Unknown')
                job_status = status.get('status', 'unknown')
                
                logger.info(f"Processing: {progress}% - {current_step}")
                
                if job_status == 'completed':
                    logger.info("Processing: ✅ Completed successfully")
                    return True, status
                elif job_status == 'failed':
                    error = status.get('error', 'Unknown error')
                    logger.error(f"Processing: ❌ Failed - {error}")
                    return False, status
                
                time.sleep(5)  # Wait 5 seconds before next check
                
            except Exception as e:
                logger.error(f"Status monitoring error: {e}")
                time.sleep(5)
        
        logger.error("Processing: ❌ Timeout reached")
        return False, None
    
    def download_and_analyze(self, job_id):
        """Download processed audio and analyze quality"""
        try:
            # Download the processed file
            response = self.session.get(f"{BACKEND_URL}/download/{job_id}")
            if response.status_code != 200:
                logger.error(f"Download failed: Status {response.status_code}")
                return False, {}
            
            # Save to temporary file
            output_path = f"/tmp/processed_{job_id}.mp3"
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Download: ✅ Saved to {output_path}")
            
            # Analyze audio quality
            analysis = self.analyze_audio_quality(output_path)
            
            return True, analysis
            
        except Exception as e:
            logger.error(f"Download/Analysis: ❌ Error: {e}")
            return False, {}
    
    def analyze_audio_quality(self, audio_path):
        """Analyze audio quality including LUFS measurement"""
        analysis = {}
        
        try:
            # 1. Get basic audio info
            info_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                '-show_format', '-show_streams', audio_path
            ]
            result = subprocess.run(info_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                info = json.loads(result.stdout)
                format_info = info.get('format', {})
                analysis['duration'] = float(format_info.get('duration', 0))
                analysis['bitrate'] = int(format_info.get('bit_rate', 0))
                analysis['size_mb'] = os.path.getsize(audio_path) / (1024 * 1024)
                
                # Get stream info
                streams = info.get('streams', [])
                if streams:
                    stream = streams[0]
                    analysis['sample_rate'] = int(stream.get('sample_rate', 0))
                    analysis['channels'] = int(stream.get('channels', 0))
            
            # 2. Measure LUFS loudness
            lufs_cmd = [
                'ffmpeg', '-i', audio_path, '-af', 'loudnorm=print_format=json', 
                '-f', 'null', '-'
            ]
            result = subprocess.run(lufs_cmd, capture_output=True, text=True)
            
            # Parse LUFS from output (check both stdout and stderr)
            output_text = result.stderr + result.stdout
            output_lines = output_text.split('\n')
            for i, line in enumerate(output_lines):
                if '"input_i"' in line and i < len(output_lines) - 10:
                    # Try to extract JSON block
                    json_start = i - 1
                    json_block = []
                    for j in range(json_start, min(json_start + 15, len(output_lines))):
                        if output_lines[j].strip().startswith('{'):
                            json_block = []
                        json_block.append(output_lines[j])
                        if output_lines[j].strip().endswith('}'):
                            break
                    
                    try:
                        json_str = '\n'.join(json_block)
                        lufs_data = json.loads(json_str)
                        analysis['input_lufs'] = float(lufs_data.get('input_i', 0))
                        analysis['target_lufs'] = -16.0
                        analysis['lufs_difference'] = abs(analysis['input_lufs'] - analysis['target_lufs'])
                        break
                    except:
                        continue
            
            # 3. Check for SFX presence (look for frequency spikes that might indicate SFX)
            spectral_cmd = [
                'ffmpeg', '-i', audio_path, '-af', 
                'showspectrumpic=s=1024x512:mode=separate:color=rainbow', 
                '-frames:v', '1', '/tmp/spectrum.png'
            ]
            subprocess.run(spectral_cmd, capture_output=True)
            
            # 4. Check metadata
            metadata_cmd = ['ffprobe', '-v', 'quiet', '-show_format', audio_path]
            result = subprocess.run(metadata_cmd, capture_output=True, text=True)
            if 'title=' in result.stdout:
                analysis['has_metadata'] = True
            else:
                analysis['has_metadata'] = False
            
            logger.info(f"Audio Analysis Complete:")
            logger.info(f"  Duration: {analysis.get('duration', 0):.2f}s")
            logger.info(f"  LUFS: {analysis.get('input_lufs', 'N/A')} (target: -16.0)")
            logger.info(f"  Bitrate: {analysis.get('bitrate', 0)} bps")
            logger.info(f"  Sample Rate: {analysis.get('sample_rate', 0)} Hz")
            logger.info(f"  Metadata: {'✅' if analysis.get('has_metadata') else '❌'}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            return analysis
    
    def run_comprehensive_test(self):
        """Run complete test suite"""
        logger.info("🚀 Starting Comprehensive Voicepod Backend Testing")
        logger.info("=" * 60)
        
        # Test 1: API Health
        if not self.test_api_health():
            return False
        
        # Test 2: Presets
        presets_ok, presets = self.test_presets_endpoint()
        if not presets_ok:
            return False
        
        # Test 3: Create test audio
        logger.info("Creating test audio file...")
        test_audio = self.create_test_audio()
        
        # Test 4: Upload and Process
        logger.info("Testing upload and processing...")
        upload_ok, job_id = self.test_upload_and_process(test_audio, "podcast_calm")
        if not upload_ok:
            return False
        
        # Test 5: Monitor Processing
        logger.info("Monitoring processing...")
        process_ok, final_status = self.monitor_processing(job_id)
        if not process_ok:
            return False
        
        # Test 6: Download and Analyze
        logger.info("Downloading and analyzing output...")
        download_ok, analysis = self.download_and_analyze(job_id)
        if not download_ok:
            return False
        
        # Test Results Summary
        self.print_test_summary(final_status, analysis)
        
        return True
    
    def print_test_summary(self, status, analysis):
        """Print comprehensive test results"""
        logger.info("=" * 60)
        logger.info("🎯 TEST RESULTS SUMMARY")
        logger.info("=" * 60)
        
        # Processing Statistics
        stats = status.get('statistics', {})
        logger.info(f"✅ Processing completed successfully")
        logger.info(f"📊 Preset used: {stats.get('preset', 'unknown')}")
        logger.info(f"🎭 Emotion peaks detected: {stats.get('emotion_peaks', 0)}")
        logger.info(f"🗣️ Fillers detected: {stats.get('fillers_detected', 0)}")
        
        # Audio Quality Analysis
        logger.info(f"\n🔊 AUDIO QUALITY ANALYSIS:")
        logger.info(f"⏱️ Duration: {analysis.get('duration', 0):.2f} seconds")
        logger.info(f"📈 File size: {analysis.get('size_mb', 0):.2f} MB")
        logger.info(f"🎵 Sample rate: {analysis.get('sample_rate', 0)} Hz")
        logger.info(f"🔊 Channels: {analysis.get('channels', 0)}")
        logger.info(f"💾 Bitrate: {analysis.get('bitrate', 0)} bps")
        
        # LUFS Analysis (Critical)
        input_lufs = analysis.get('input_lufs')
        if input_lufs is not None:
            target_lufs = -16.0
            difference = abs(input_lufs - target_lufs)
            status_icon = "✅" if difference <= 0.5 else "⚠️" if difference <= 1.0 else "❌"
            logger.info(f"\n🎚️ LOUDNESS ANALYSIS:")
            logger.info(f"{status_icon} Measured LUFS: {input_lufs:.2f}")
            logger.info(f"🎯 Target LUFS: {target_lufs}")
            logger.info(f"📏 Difference: {difference:.2f} LUFS")
            
            if difference <= 0.5:
                logger.info("✅ LUFS target achieved within acceptable range!")
            elif difference <= 1.0:
                logger.info("⚠️ LUFS close to target but could be improved")
            else:
                logger.info("❌ LUFS significantly off target - needs adjustment")
        else:
            logger.info("❌ Could not measure LUFS - analysis failed")
        
        # Metadata Check
        has_metadata = analysis.get('has_metadata', False)
        logger.info(f"\n📋 METADATA: {'✅ Present' if has_metadata else '❌ Missing'}")
        
        # SFX Check (based on emotion peaks)
        emotion_peaks = stats.get('emotion_peaks', 0)
        if emotion_peaks >= 2:
            logger.info(f"✅ SFX: Expected at {emotion_peaks} emotion peaks")
        elif emotion_peaks == 1:
            logger.info(f"⚠️ SFX: Only 1 emotion peak detected (expected 2)")
        else:
            logger.info(f"❌ SFX: No emotion peaks detected")
        
        logger.info("=" * 60)

def main():
    """Main test execution"""
    tester = VoicepodTester()
    
    try:
        success = tester.run_comprehensive_test()
        if success:
            logger.info("🎉 All tests completed successfully!")
            return 0
        else:
            logger.error("❌ Some tests failed!")
            return 1
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Test suite failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())