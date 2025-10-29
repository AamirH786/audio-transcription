# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.staticfiles import StaticFiles
# import whisper
# import os
# import librosa
# import numpy as np
# from sklearn.cluster import AgglomerativeClustering

# app = FastAPI()
# app.mount("/static", StaticFiles(directory="static"), name="static")

# model = whisper.load_model("base")

# @app.get("/", response_class=HTMLResponse)
# async def home():
#     with open("templates/index.html", encoding="utf-8") as f:
#         return f.read()

# @app.post("/transcribe")
# async def transcribe_audio(file: UploadFile = File(...)):
#     file_path = f"uploads/{file.filename}"
#     with open(file_path, "wb") as f:
#         f.write(await file.read())
    
#     try:
#         # Remove language="hi" - Let Whisper auto-detect
#         result = model.transcribe(file_path)  # Auto-detect language
        
#         print(f"Detected Language: {result['language']}")  # Show detected language
        
#         audio, sr = librosa.load(file_path, sr=16000)
        
#         features = []
#         valid_segs = []
        
#         for seg in result["segments"]:
#             s = int(seg["start"] * sr)
#             e = int(seg["end"] * sr)
#             if e > len(audio): e = len(audio)
            
#             chunk = audio[s:e]
#             if len(chunk) > 8000:
#                 pitch = librosa.feature.zero_crossing_rate(chunk).mean()
#                 energy = np.mean(np.abs(chunk))
#                 features.append([pitch, energy])
#                 valid_segs.append(seg)
        
#         if len(features) > 1:
#             X = np.array(features)
#             n = min(6, max(2, len(X) // 3))
#             labels = AgglomerativeClustering(n_clusters=n).fit_predict(X)
            
#             mapping = {}
#             counter = 1
#             segments = []
            
#             for i, seg in enumerate(valid_segs):
#                 if labels[i] not in mapping:
#                     mapping[labels[i]] = counter
#                     counter += 1
                
#                 segments.append({
#                     "start": seg["start"],
#                     "end": seg["end"],
#                     "text": seg["text"],
#                     "speaker": f"Speaker {mapping[labels[i]]}"
#                 })
#         else:
#             segments = [{
#                 "start": s["start"],
#                 "end": s["end"],
#                 "text": s["text"],
#                 "speaker": "Speaker 1"
#             } for s in result["segments"]]
        
#         os.remove(file_path)
#         return JSONResponse({
#             "transcript": result["text"],
#             "segments": segments,
#             "language": result["language"]  # Language info bhi send karo
#         })
        
#     except Exception as e:
#         if os.path.exists(file_path):
#             os.remove(file_path)
#         return JSONResponse({"error": str(e)}, status_code=500)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
#
#-----------------------------------------------------------------------------
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import whisper
import os
import tempfile  # YE ADD KIYA
import librosa
import numpy as np
from sklearn.cluster import AgglomerativeClustering


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


model = whisper.load_model("base")


@app.get("/", response_class=HTMLResponse)
async def home():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # CHANGE: Temporary file use kiya (render pe /uploads folder nahi banega)
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
    # CHANGE: PORT environment variable 
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
