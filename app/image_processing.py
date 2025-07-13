import torch
import clip
from PIL import Image
import requests
from io import BytesIO
import tempfile
import cv2
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def get_image_vector_from_url(url: str) -> list:
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = preprocess(Image.open(BytesIO(response.content))).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.squeeze().tolist()
    except Exception as e:
        raise RuntimeError(f"Failed to process image from URL: {e}")

def get_image_vector_from_file(file) -> list:
    try:
        image = preprocess(Image.open(file)).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.squeeze().tolist()
    except Exception as e:
        raise RuntimeError(f"Failed to process uploaded image: {e}")

def get_video_vector_from_file(file) -> list:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // 3)

        vectors = []
        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_tensor = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                vector = model.encode_image(image_tensor)
                vector = vector / vector.norm(dim=-1, keepdim=True)
                vectors.append(vector)

        cap.release()
        os.remove(tmp_path)

        if not vectors:
            raise RuntimeError("No valid frames extracted from video")

        avg_vector = torch.mean(torch.stack(vectors), dim=0)
        return avg_vector.squeeze().tolist()

    except Exception as e:
        raise RuntimeError(f"Failed to process video: {e}")

def get_video_vector_from_url(url: str) -> list:
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            return get_video_vector_from_file(f)

    except Exception as e:
        raise RuntimeError(f"Failed to process video from URL: {e}")
