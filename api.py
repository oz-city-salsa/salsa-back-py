from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import random
import os
from typing import Dict, Any

app = FastAPI(
    title="Video Scoring API",
    description="API that receives video files and returns random scores",
    version="1.0.0"
)

# Supported video formats
SUPPORTED_VIDEO_FORMATS = {
    'video/mp4', 'video/avi', 'video/mov', 'video/mkv', 
    'video/wmv', 'video/flv', 'video/webm', 'video/quicktime'
}

@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint with API information"""
    return {
        "message": "Video Scoring API",
        "description": "Upload a video to get a random score between 10 and 100",
        "endpoints": {
            "POST /score-video": "Upload video and get score",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/score-video")
async def score_video(
    video: UploadFile = File(..., description="Video file to score")
) -> Dict[str, Any]:
    """
    Upload a video file and receive a random score between 10 and 100
    
    Args:
        video: The video file to process
        
    Returns:
        JSON response with video info and random score
    """
    
    # Validate file is provided
    if not video:
        raise HTTPException(status_code=400, detail="No video file provided")
    
    # Validate file type
    if video.content_type not in SUPPORTED_VIDEO_FORMATS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported video format. Supported formats: {', '.join(SUPPORTED_VIDEO_FORMATS)}"
        )
    
    # Get file size
    file_size = 0
    if hasattr(video, 'size') and video.size:
        file_size = video.size
    else:
        # If size is not available, read the content to get size
        content = await video.read()
        file_size = len(content)
        # Reset file pointer
        await video.seek(0)
    
    # Validate file size (max 100MB)
    max_size = 100 * 1024 * 1024  # 100MB in bytes
    if file_size > max_size:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Maximum size allowed: {max_size // (1024*1024)}MB"
        )
    
    # Generate random score between 10 and 100
    score = random.randint(10, 100)
    
    # Return response with video info and score
    return {
        "status": "success",
        "video_info": {
            "filename": video.filename,
            "content_type": video.content_type,
            "size_bytes": file_size,
            "size_mb": round(file_size / (1024 * 1024), 2)
        },
        "score": score,
        "score_range": "10-100",
        "message": f"Video '{video.filename}' scored {score} points!"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
