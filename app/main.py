from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.image_processing import (
    get_image_vector_from_url,
    get_image_vector_from_file,
    get_video_vector_from_file,
    get_video_vector_from_url
)

from fastapi import UploadFile, File

app = FastAPI()
origins = [
    "http://localhost:8080",       
    "http://localhost:3000",       
    "http://127.0.0.1:8080",
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          
    allow_credentials=True,
    allow_methods=["*"],             
    allow_headers=["*"],            
)

class ImageURL(BaseModel):
    url: str
class MediaURL(BaseModel):
    url: str

@app.post("/extract-vector")
def extract_vector(data: MediaURL):
    try:
        if any(data.url.lower().endswith(ext) for ext in [".mp4", ".webm", ".mov", ".avi"]):
            vector = get_video_vector_from_url(data.url)
        else:
            vector = get_image_vector_from_url(data.url)
        return {"vector": vector}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/extract-vector-from-upload")
async def extract_vector_from_upload(image: UploadFile = File(...)):
    try:
        image_vector = get_image_vector_from_file(image.file)
        return {"vector": image_vector}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/extract-video-vector")
async def extract_video_vector(video: UploadFile = File(...)):
    try:
        vector = get_video_vector_from_file(video.file)
        return {"vector": vector}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    


