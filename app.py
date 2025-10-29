from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import whisper
import os
import tempfile
import librosa
import numpy as np
from sklearn.cluster import AgglomerativeClustering

app = FastAPI()
model = whisper.load_model("base")

@app.get("/", response_class=HTMLResponse)
async def home():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio Transcription App</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px 0;
        }
        .main-card {
            background: white;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            padding: 40px;
            margin-top: 30px;
        }
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            transition: all 0.3s;
            background: #f8f9ff;
        }
        .upload-area:hover {
            border-color: #764ba2;
            background: #f0f2ff;
        }
        .btn-custom {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            padding: 12px 40px;
            font-size: 18px;
            border-radius: 50px;
            transition: transform 0.2s;
        }
        .btn-custom:hover {
            transform: scale(1.05);
        }
        .speaker-badge {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
            margin-right: 10px;
        }
        .segment-item {
            border-left: 4px solid #667eea;
            margin-bottom: 15px;
            padding: 15px;
            background: #f8f9ff;
            border-radius: 10px;
            transition: all 0.3s;
        }
        .segment-item:hover {
            background: #f0f2ff;
            transform: translateX(5px);
        }
        .time-badge {
            background: #764ba2;
            color: white;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 12px;
        }
        h1 {
            color: white;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        .loading-spinner {
            color: #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center mb-4">🎤 AI Audio Transcription</h1>
        
        <div class="main-card">
            <div class="upload-area mb-4">
                <h3>📁 Upload Audio/Video File</h3>
                <p class="text-muted">Supports MP3, MP4, WAV, and more</p>
                <input type="file" id="audioFile" class="form-control mt-3" accept="audio/*,video/*" style="max-width: 500px; margin: 0 auto;">
            </div>
            
            <div class="text-center">
                <button onclick="transcribe()" class="btn btn-primary btn-custom">
                    ✨ Transcribe Now
                </button>
            </div>
        </div>

        <div id="loading" class="text-center mt-5" style="display:none;">
            <div class="spinner-border loading-spinner" role="status" style="width: 3rem; height: 3rem;"></div>
            <h4 class="text-white mt-3">Processing your audio...</h4>
            <p class="text-white">This may take a minute</p>
        </div>

        <div id="result" class="main-card mt-5" style="display:none;">
            <h2 class="mb-4">📝 Full Transcript</h2>
            <div class="alert alert-info">
                <p id="fullTranscript" class="mb-0" style="font-size: 16px; line-height: 1.8;"></p>
            </div>

            <h3 class="mt-5 mb-4">👥 Speaker-wise Segments</h3>
            <div id="segments"></div>
        </div>
    </div>

    <script>
        async function transcribe() {
            const fileInput = document.getElementById('audioFile');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('⚠️ Please select an audio/video file first!');
                return;
            }

            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/transcribe', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                
                if (data.error) {
                    alert('❌ Error: ' + data.error);
                    return;
                }

                document.getElementById('fullTranscript').textContent = data.transcript;

                const segmentsDiv = document.getElementById('segments');
                segmentsDiv.innerHTML = '';
                
                data.segments.forEach((segment, index) => {
                    const start = formatTime(segment.start);
                    const end = formatTime(segment.end);
                    const speaker = segment.speaker || "Speaker Unknown";
                    
                    const item = document.createElement('div');
                    item.className = 'segment-item';
                    item.innerHTML = `
                        <span class="speaker-badge">${speaker}</span>
                        <span class="time-badge">${start} - ${end}</span>
                        <p class="mt-2 mb-0" style="font-size: 15px;">${segment.text}</p>
                    `;
                    segmentsDiv.appendChild(item);
                });

                document.getElementById('result').style.display = 'block';
                document.getElementById('result').scrollIntoView({ behavior: 'smooth' });
                
            } catch (error) {
                alert('❌ Error: ' + error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }

        function formatTime(seconds) {
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${mins}:${secs.toString().padStart(2, '0')}`;
        }
    </script>
</body>
</html>"""

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        temp.write(await file.read())
        file_path = temp.name
    
    try:
        result = model.transcribe(file_path)
        print(f"Detected Language: {result['language']}")
        
        audio, sr = librosa.load(file_path, sr=16000)
        features = []
        valid_segs = []
        
        for seg in result["segments"]:
            s = int(seg["start"] * sr)
            e = int(seg["end"] * sr)
            if e > len(audio): e = len(audio)
            
            chunk = audio[s:e]
            if len(chunk) > 8000:
                pitch = librosa.feature.zero_crossing_rate(chunk).mean()
                energy = np.mean(np.abs(chunk))
                features.append([pitch, energy])
                valid_segs.append(seg)
        
        if len(features) > 1:
            X = np.array(features)
            n = min(6, max(2, len(X) // 3))
            labels = AgglomerativeClustering(n_clusters=n).fit_predict(X)
            
            mapping = {}
            counter = 1
            segments = []
            
            for i, seg in enumerate(valid_segs):
                if labels[i] not in mapping:
                    mapping[labels[i]] = counter
                    counter += 1
                
                segments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                    "speaker": f"Speaker {mapping[labels[i]]}"
                })
        else:
            segments = [{
                "start": s["start"],
                "end": s["end"],
                "text": s["text"],
                "speaker": "Speaker 1"
            } for s in result["segments"]]
        
        os.remove(file_path)
        return JSONResponse({
            "transcript": result["text"],
            "segments": segments,
            "language": result["language"]
        })
        
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
