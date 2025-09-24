from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
import os
import base64
import requests
import time
import uvicorn
from io import BytesIO
from fastapi.responses import StreamingResponse
import logging
import uuid
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CONFIG ===
API_KEY = os.getenv("OPENAI_API_KEY", "")
ENDPOINT = "https://api.openai.com/v1/images/edits"
MODEL = "gpt-image-1"
OUTPUT_FILE = "edited_image.png"

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # Service role key for storage operations
STORAGE_BUCKET = "images"

# Initialize Supabase client
supabase: Client = None
if SUPABASE_URL:
    # Try service key first (bypasses RLS), then anon key
    key_to_use = SUPABASE_SERVICE_KEY if SUPABASE_SERVICE_KEY else SUPABASE_ANON_KEY
    key_type = "service" if SUPABASE_SERVICE_KEY else "anon"
    
    if key_to_use:
        try:
            supabase = create_client(SUPABASE_URL, key_to_use)
            logger.info(f"✅ Supabase client initialized successfully using {key_type} key")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase client: {e}")
    else:
        logger.warning("⚠️ No Supabase key found (SUPABASE_ANON_KEY or SUPABASE_SERVICE_KEY)")
else:
    logger.warning("⚠️ Supabase URL not found. Storage upload will be disabled.")

# FastAPI app
app = FastAPI(
    title="AI Image Editor API",
    description="API for editing images using OpenAI's image editing capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add trusted host middleware (helps prevent invalid requests)
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]  # In production, specify actual domains
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler for better error handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handler caught: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )

# Handle validation errors
@app.exception_handler(422)
async def validation_exception_handler(request: Request, exc):
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request format or data"}
    )

# Request model to receive input data
class ImageRequest(BaseModel):
    image_url: HttpUrl  # This validates the URL format
    prompt: str
    
    class Config:
        # Example values for API documentation
        schema_extra = {
            "example": {
                "image_url": "https://example.com/image.jpg",
                "prompt": "Make this image more colorful and vibrant"
            }
        }

# Response model for image editing
class ImageResponse(BaseModel):
    success: bool
    message: str
    storage_info: dict = None
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Image edited and uploaded successfully",
                "storage_info": {
                    "uploaded": True,
                    "url": "https://your-project.supabase.co/storage/v1/object/public/images/edited_image_123.jpg",
                    "filename": "edited_image_123.jpg",
                    "bucket": "images"
                }
            }
        }

def get_content_type_from_url(url):
    """Determine content type based on URL extension"""
    url_lower = url.lower()
    if url_lower.endswith(('.png', '.PNG')):
        return "image/png"
    elif url_lower.endswith(('.jpg', '.jpeg', '.JPG', '.JPEG')):
        return "image/jpeg"
    elif url_lower.endswith(('.gif', '.GIF')):
        return "image/gif"
    elif url_lower.endswith(('.webp', '.WEBP')):
        return "image/webp"
    else:
        return "image/jpeg"  # default fallback

def download_image_from_url(url):
    """Download image from URL and return image data"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image from URL {url}: {e}")

def optimize_image_to_jpg(image_data: bytes, quality: int = 85) -> bytes:
    """Convert and optimize image to JPG format with compression while preserving original resolution"""
    try:
        # Open image from bytes
        image = Image.open(BytesIO(image_data))
        original_size_info = f"{image.width}x{image.height}"
        
        # Convert to RGB if necessary (PNG with transparency, etc.)
        if image.mode in ('RGBA', 'LA', 'P'):
            # Create white background for transparent images
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            if image.mode in ('RGBA', 'LA'):
                background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save as JPG with compression (keeping original resolution)
        output_buffer = BytesIO()
        image.save(output_buffer, format='JPEG', quality=quality, optimize=True)
        optimized_data = output_buffer.getvalue()
        
        # Log compression results
        original_size = len(image_data)
        optimized_size = len(optimized_data)
        compression_ratio = (1 - optimized_size / original_size) * 100
        logger.info(f"Image optimized ({original_size_info}): {original_size:,} bytes → {optimized_size:,} bytes ({compression_ratio:.1f}% reduction)")
        
        return optimized_data
        
    except Exception as e:
        logger.error(f"Error optimizing image: {e}")
        # Return original data if optimization fails
        return image_data

def upload_to_supabase(image_data: bytes, filename: str) -> dict:
    """Upload image to Supabase storage and return the public URL"""
    if not supabase:
        logger.warning("Supabase client not available, skipping upload")
        return {"uploaded": False, "url": None, "message": "Supabase not configured"}

    try:
        logger.info(f"Uploading {filename} to Supabase storage bucket '{STORAGE_BUCKET}'")

        # Pass image_data directly as bytes to Supabase storage

        response = supabase.storage.from_(STORAGE_BUCKET).upload(filename, image_data, {
            'content-type' : 'image/jpeg',
            'upsert' : 'true'
        })

        # Check response type - response is an UploadResponse object
        if hasattr(response, 'full_path') and response.full_path:
            public_url = supabase.storage.from_(STORAGE_BUCKET).get_public_url(filename)
            logger.info(f"✅ Successfully uploaded to Supabase: {public_url}")

            return {
                "uploaded": True,
                "url": public_url,
                "filename": filename,
                "bucket": STORAGE_BUCKET,
                "message": "Successfully uploaded to Supabase storage"
            }

        logger.error(f"❌ Unexpected Supabase response: {response}")
        return {"uploaded": False, "url": None, "message": f"Unexpected response: {response}"}

    except Exception as e:
        logger.error(f"❌ Error uploading to Supabase: {e}")
        return {"uploaded": False, "url": None, "message": f"Upload error: {e}"}

def edit_image(image_data, prompt, image_url=None):
    """Send image to OpenAI API for editing"""
    headers = {"Authorization": f"Bearer {API_KEY}"}

    # Prepare files for upload
    # Determine content type from URL if provided, otherwise default to jpeg
    if image_url:
        image_content_type = get_content_type_from_url(image_url)
    else:
        image_content_type = "image/png"  # default fallback

    files = {"image": ("image", image_data, image_content_type)}

    data = {
        "model": MODEL,
        "prompt": prompt,
        "n": "1",
        "size": "auto"
    }

    print("Sending request to OpenAI image edits endpoint...")

    start_time = time.time()
    resp = requests.post(ENDPOINT, headers=headers, files=files, data=data, timeout=120)
    elapsed = time.time() - start_time

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Error from API: {resp.status_code}, {resp.text}")

    j = resp.json()

    # try base64 field first
    image_data = j["data"][0].get("b64_json")
    if image_data:
        image_bytes = base64.b64decode(image_data)
        return image_bytes

    # fallback: URL
    url = j["data"][0].get("url")
    if url:
        download = requests.get(url)
        download.raise_for_status()
        return download.content

    raise HTTPException(status_code=500, detail="Unexpected response JSON from OpenAI API")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Image Editor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "api_key_configured": bool(API_KEY and API_KEY.startswith("sk-")),
        "supabase_configured": bool(supabase is not None),
        "storage_bucket": STORAGE_BUCKET if supabase else None
    }

@app.post("/edit-image/", response_model=ImageResponse)
async def edit_image_endpoint(request: ImageRequest):
    try:
        # Convert HttpUrl to string for processing
        image_url_str = str(request.image_url)
        
        # Download the image from the URL provided
        logger.info(f"Downloading image from: {image_url_str}")
        image_data = download_image_from_url(image_url_str)

        # Send the image to OpenAI API for editing
        logger.info(f"Received prompt: {request.prompt}")
        edited_image = edit_image(image_data, request.prompt, image_url_str)
        
        # Optimize image to JPG format for smaller file size
        logger.info("Optimizing image to JPG format...")
        optimized_image = optimize_image_to_jpg(edited_image)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"edited_image_{timestamp}_{unique_id}.jpg"
        
        # Upload optimized image to Supabase storage
        storage_result = upload_to_supabase(optimized_image, filename)
        
        if storage_result["uploaded"]:
            return ImageResponse(
                success=True,
                message="Image edited and uploaded successfully to Supabase storage",
                storage_info=storage_result
            )
        else:
            # Even if upload fails, we can still return the image data
            logger.warning("Supabase upload failed, but image was processed successfully")
            return ImageResponse(
                success=True,
                message="Image edited successfully, but storage upload failed",
                storage_info=storage_result
            )
            
    except HTTPException as e:
        logger.error(f"HTTP Exception: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in edit_image_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

@app.post("/edit-image-stream/")
async def edit_image_stream_endpoint(request: ImageRequest):
    """Alternative endpoint that returns the image as a stream (for direct download)"""
    try:
        # Convert HttpUrl to string for processing
        image_url_str = str(request.image_url)
        
        # Download the image from the URL provided
        logger.info(f"Downloading image from: {image_url_str}")
        image_data = download_image_from_url(image_url_str)

        # Send the image to OpenAI API for editing
        logger.info(f"Received prompt: {request.prompt}")
        edited_image = edit_image(image_data, request.prompt, image_url_str)
        
        # Optimize image to JPG format for smaller file size
        logger.info("Optimizing image to JPG format...")
        optimized_image = optimize_image_to_jpg(edited_image)
        
        return StreamingResponse(
            BytesIO(optimized_image), 
            media_type="image/jpeg",
            headers={"Content-Disposition": "attachment; filename=edited_image.jpg"}
        )
    except HTTPException as e:
        logger.error(f"HTTP Exception: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in edit_image_stream_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

if __name__ == "__main__":
    print("🚀 Starting AI Image Editor Server...")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("⚡ Server running on: http://localhost:8000")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True,
        server_header=False,
        date_header=False,
        timeout_keep_alive=30,
        timeout_graceful_shutdown=10
    )
