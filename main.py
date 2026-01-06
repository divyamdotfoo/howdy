import time
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from openai import OpenAI
import whisper
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

WATCH_FOLDER = Path.home() / "Pictures" / "Photo Booth Library" / "Pictures"
OUTPUT_FOLDER = Path.home() / "Desktop" / "inglis"

model = whisper.load_model("tiny")

# Initialize OpenAI client with API key from .env
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file. Please add OPENAI_API_KEY=your_key_here to your .env file.")
openai_client = OpenAI(api_key=api_key)


class Toast:
    @staticmethod
    def _show(title, message, subtitle="", sound="default"):
        cmd = ['terminal-notifier', '-title', str(title), '-message', str(message)]
        
        if subtitle:
            cmd.extend(['-subtitle', str(subtitle)])
        
        if sound:
            cmd.extend(['-sound', str(sound)])
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    
    @staticmethod
    def loading():
        Toast._show("Howdy mate", "All the best!", sound="default")
    
    @staticmethod
    def transcribing():
        Toast._show("Transcribing", "Creating transcription...", sound="default")
    
    @staticmethod
    def success():
        Toast._show("Complete", "Processing completed successfully", sound="Glass")
    
    @staticmethod
    def error(message=""):
        title = "Error" if message else "Processing Error"
        Toast._show(title, message if message else "An error occurred", sound="Basso")


def wait_for_file_complete(file_path, stability_seconds=3, check_interval=0.5):
    previous_size = -1
    previous_mtime = -1
    stable_count = 0
    required_stable_checks = int(stability_seconds / check_interval)
    
    print(f"Waiting for {file_path.name} to finish recording...")
    
    while True:
        if not file_path.exists():
            time.sleep(check_interval)
            continue
        
        try:
            current_size = file_path.stat().st_size
            current_mtime = file_path.stat().st_mtime
            
            if current_size == 0:
                time.sleep(check_interval)
                continue
            
            if current_size == previous_size and current_mtime == previous_mtime:
                stable_count += 1
                if stable_count >= required_stable_checks:
                    print(f"File is complete: {file_path.name} ({current_size} bytes)")
                    return True
            else:
                stable_count = 0
            
            previous_size = current_size
            previous_mtime = current_mtime
            time.sleep(check_interval)
            
        except (OSError, PermissionError):
            time.sleep(check_interval)


class FileAddedHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        if file_path.suffix.lower() != '.mov':
            return
        
        print(f"New .mov file detected: {file_path.name}")
        Toast.loading()
        
        wait_for_file_complete(file_path)
        Toast.transcribing()
        process_video(file_path)


def convert_to_mp3(video_path):
    mp3_path = OUTPUT_FOLDER / f"{video_path.stem}.mp3"
    print(f"Converting {video_path.name} to MP3...")
    
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-vn',
        '-acodec', 'libmp3lame',
        '-q:a', '0',
        '-map', 'a',
        str(mp3_path)
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        print(f"Conversion complete: {mp3_path.name}")
        return mp3_path
    except subprocess.CalledProcessError as e:
        raise Exception(f"FFmpeg conversion failed: {e.stderr.decode()}")
    except FileNotFoundError:
        raise Exception("ffmpeg not found. Install with: brew install ffmpeg")


def transcribe_audio(audio_path):
    result = model.transcribe(str(audio_path))
    transcript_text = result['text']
    
    txt_path = OUTPUT_FOLDER / f"{audio_path.stem}.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(transcript_text)
    print(f"Transcription saved: {txt_path.name}")
    
    return transcript_text


def get_openai_recommendations(transcript):
    """Send transcript to OpenAI for recommendations and analysis."""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that analyzes transcripts and provides recommendations, insights, and suggestions. Format your response in a clear, organized manner with sections."
                },
                {
                    "role": "user",
                    "content": f"Please analyze this transcript and provide recommendations, key insights, and any suggestions:\n\n{transcript}"
                }
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"OpenAI API error: {str(e)}")


def save_recommendations(audio_path, transcript, recommendations):
    """Save recommendations in a clean, organized text file."""
    recommendations_path = OUTPUT_FOLDER / f"{audio_path.stem}_recommendations.txt"
    
    with open(recommendations_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"RECOMMENDATIONS & ANALYSIS\n")
        f.write(f"File: {audio_path.stem}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("TRANSCRIPT:\n")
        f.write("-" * 80 + "\n")
        f.write(transcript)
        f.write("\n\n")
        
        f.write("=" * 80 + "\n\n")
        
        f.write("RECOMMENDATIONS:\n")
        f.write("-" * 80 + "\n")
        f.write(recommendations)
        f.write("\n")
        
    print(f"Recommendations saved: {recommendations_path.name}")


def process_video(video_path):
    try:
        mp3_path = convert_to_mp3(video_path)
        transcript = transcribe_audio(mp3_path)
        
        print("Getting recommendations from OpenAI...")
        recommendations = get_openai_recommendations(transcript)
        save_recommendations(mp3_path, transcript, recommendations)
        
        Toast.success()
    except Exception as e:
        Toast.error(str(e))
        print(f"Error: {e}")


def start_watching():
    if not WATCH_FOLDER.exists():
        print(f"Error: Folder {WATCH_FOLDER} does not exist")
        return
    
    OUTPUT_FOLDER.mkdir(exist_ok=True)
    
    print(f"Watching {WATCH_FOLDER} for .mov files...")
    print("Press Ctrl+C to stop...")
    
    event_handler = FileAddedHandler()
    observer = Observer()
    observer.schedule(event_handler, str(WATCH_FOLDER), recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nStopped watching folder")
    
    observer.join()


if __name__ == "__main__":
    start_watching()
