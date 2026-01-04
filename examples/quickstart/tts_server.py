from flask import Flask, request, send_file
import subprocess
import io

app = Flask(__name__)

# --- CONFIGURATION ---
# The path you used in your command
MODEL_PATH = "/home/devbuntu23/pipecat/models/piper-voices/en_US-lessac-medium.onnx"
# The binary installed by 'uv add piper-tts'
PIPER_BINARY = "piper" 

@app.route('/', methods=['POST'])
def tts():
    """
    Mimics the Piper HTTP API.
    Receives JSON: {"text": "hello"}
    Returns: WAV audio data
    """
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return "No text provided", 400

    # Run the Piper CLI locally to generate audio
    # Equivalent to: echo "text" | piper --model ... --output_file -
    try:
        proc = subprocess.Popen(
            [PIPER_BINARY, '--model', MODEL_PATH, '--output_file', '-'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        wav_data, stderr = proc.communicate(input=text.encode('utf-8'))
        
        if proc.returncode != 0:
            print(f"Piper Error: {stderr.decode()}")
            return f"Piper failed: {stderr.decode()}", 500
            
        return send_file(io.BytesIO(wav_data), mimetype='audio/wav')
        
    except FileNotFoundError:
        return "Piper binary not found. Did you run 'uv add piper-tts'?", 500

if __name__ == '__main__':
    print(f"🚀 Starting Local TTS Server on port 5000...")
    print(f"🗣️  Model: {MODEL_PATH}")
    app.run(port=5000)