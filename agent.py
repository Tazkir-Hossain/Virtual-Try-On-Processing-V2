# main.py - Improved FastAPI Backend with Better Inpainting
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import requests
import io
from PIL import Image, ImageFilter, ImageEnhance
import os
import json
import tempfile
import shutil
from pathlib import Path
from inference_sdk import InferenceHTTPClient
from ultralytics import SAM
import supervision as sv
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Fashion Inpainting App")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuration from environment variables
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "main-fashion-wmyfk/1")
SEGMIND_API_KEY = os.getenv("SEGMIND_API_KEY")
SEGMIND_URL = os.getenv("SEGMIND_URL", "https://api.segmind.com/v1/sdxl-inpaint")

# Validate required API keys
if not ROBOFLOW_API_KEY:
    raise ValueError("ROBOFLOW_API_KEY environment variable is required")

if not SEGMIND_API_KEY:
    raise ValueError("SEGMIND_API_KEY environment variable is required")

# Configuration
ROBOFLOW_CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

SEGMIND_API_KEY = SEGMIND_API_KEY 
SEGMIND_URL = "https://api.segmind.com/v1/sdxl-inpaint"

# Global SAM model
sam_model = None

def load_sam_model():
    """Load SAM model once at startup"""
    global sam_model
    if sam_model is None:
        sam_model = SAM("sam2.1_b.pt")
    return sam_model

def image_to_base64(image_array):
    """Convert numpy array to base64 string with better quality"""
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        # Convert RGB to BGR for OpenCV
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    # Use higher quality encoding
    encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 1]
    _, buffer = cv2.imencode('.png', image_array, encode_param)
    return base64.b64encode(buffer).decode('utf-8')

def base64_to_image(base64_string):
    """Convert base64 string to numpy array"""
    image_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Return as RGB

def base64_to_pil(base64_string):
    """Convert base64 string to PIL Image"""
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))

def pil_to_base64(pil_image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def classify_clothing_region(class_name):
    """Classify clothing items into upper, lower, or other"""
    upper_items = ['shirt', 'blouse', 'jacket', 'coat', 'sweater', 'hoodie', 'top', 't-shirt', 'tshirt']
    lower_items = ['pants', 'jeans', 'shorts', 'skirt', 'trouser', 'pant', 'leggings']
    
    class_lower = class_name.lower()
    
    if any(item in class_lower for item in upper_items):
        return 'upper'
    elif any(item in class_lower for item in lower_items):
        return 'lower'
    else:
        return 'other'

def resize_image_smart(image, target_size=1024):
    """Smart resize maintaining aspect ratio and quality"""
    if isinstance(image, np.ndarray):
        height, width = image.shape[:2]
        pil_image = Image.fromarray(image)
    else:
        pil_image = image
        width, height = pil_image.size
    
    # Calculate new dimensions
    if width > height:
        new_width = target_size
        new_height = int(height * target_size / width)
    else:
        new_height = target_size
        new_width = int(width * target_size / height)
    
    # Use high-quality resampling
    resized = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    if isinstance(image, np.ndarray):
        return np.array(resized)
    return resized

def improve_mask_quality(mask_array, dilate_iterations=2, blur_radius=2):
    """Improve mask quality with morphological operations and smoothing"""
    # Convert to grayscale if needed
    if len(mask_array.shape) == 3:
        mask_gray = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)
    else:
        mask_gray = mask_array.copy()
    
    # Ensure binary mask
    _, mask_binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to improve mask
    kernel = np.ones((5, 5), np.uint8)
    
    # Fill holes and smooth edges
    mask_closed = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Dilate slightly to ensure good coverage
    if dilate_iterations > 0:
        mask_dilated = cv2.dilate(mask_closed, kernel, iterations=dilate_iterations)
    else:
        mask_dilated = mask_closed
    
    # Convert to PIL for Gaussian blur
    mask_pil = Image.fromarray(mask_dilated)
    if blur_radius > 0:
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    return np.array(mask_pil)

def enhance_prompt_for_clothing(base_prompt, clothing_class, region):
    """Generate better prompts for clothing inpainting"""
    style_modifiers = [
        "photorealistic", "high resolution", "detailed fabric texture",
        "natural lighting", "professional photography", "sharp focus"
    ]
    
    clothing_specific = {
        'shirt': "cotton shirt, well-fitted, clean lines",
        't-shirt': "cotton t-shirt, comfortable fit, smooth fabric",
        'blouse': "elegant blouse, flowing fabric, professional style",
        'jacket': "tailored jacket, structured fit, quality material",
        'pants': "well-fitted trousers, smooth fabric, professional",
        'jeans': "denim jeans, casual fit, quality denim texture",
        'shorts': "casual shorts, comfortable fit, summer style"
    }
    
    # Get clothing-specific enhancement
    clothing_enhancement = clothing_specific.get(clothing_class.lower(), "quality clothing")
    
    # Combine all elements
    enhanced_prompt = f"{base_prompt}, {clothing_enhancement}, {', '.join(style_modifiers)}"
    
    return enhanced_prompt

@app.on_event("startup")
async def startup_event():
    """Load models on startup with error handling"""
    try:
        print("🚀 Starting up Fashion Inpainting App...")
        
        # Create directories
        Path("temp").mkdir(exist_ok=True)
        Path("static").mkdir(exist_ok=True)
        print("📁 Directories created/verified")
        
        # Test Roboflow connection
        print("🔗 Testing Roboflow connection...")
        try:
            # Test with a simple API call (you might need to adjust this)
            print(f"🔑 Roboflow API Key: {ROBOFLOW_CLIENT.api_key[:10]}...")
            print("✅ Roboflow client initialized")
        except Exception as rf_error:
            print(f"⚠️ Roboflow connection warning: {rf_error}")
        
        # Load SAM model
        print("🤖 Loading SAM model...")
        try:
            load_sam_model()
            print("✅ SAM model loaded successfully")
        except Exception as sam_error:
            print(f"❌ SAM model loading failed: {sam_error}")
            print("⚠️ App will continue but segmentation may not work")
        
        # Test Segmind API
        print("🎨 Testing Segmind API connection...")
        try:
            test_headers = {'x-api-key': SEGMIND_API_KEY}
            # Just test if we have the API key set
            if SEGMIND_API_KEY and len(SEGMIND_API_KEY) > 10:
                print("✅ Segmind API key configured")
            else:
                print("⚠️ Segmind API key may not be properly configured")
        except Exception as seg_error:
            print(f"⚠️ Segmind API warning: {seg_error}")
        
        print("🎉 Startup completed!")
        
    except Exception as e:
        print(f"💥 Startup error: {str(e)}")
        # Don't crash the app, but log the error
        import traceback
        print(f"📍 Startup traceback: {traceback.format_exc()}")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    return FileResponse("static/index.html")

@app.post("/detect-clothing/")
async def detect_clothing(file: UploadFile = File(...)):
    """Detect clothing items in uploaded image with better error handling"""
    try:
        print(f"🔍 Starting detection for file: {file.filename}")
        print(f"📁 File content type: {file.content_type}")
        print(f"📏 File size: {file.size if hasattr(file, 'size') else 'Unknown'}")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            return {"success": False, "error": "Invalid file type. Please upload an image."}
        
        # Read file content
        file_content = await file.read()
        if len(file_content) == 0:
            return {"success": False, "error": "Empty file uploaded"}
        
        print(f"📖 Read {len(file_content)} bytes from file")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        print(f"💾 Saved to temporary file: {tmp_path}")
        
        try:
            # Test if we can open the image first
            test_image = Image.open(tmp_path)
            print(f"🖼️ Image opened successfully: {test_image.size} pixels, mode: {test_image.mode}")
        except Exception as img_error:
            os.unlink(tmp_path)
            return {"success": False, "error": f"Invalid image file: {str(img_error)}"}
        
        # Run Roboflow detection
        print("🤖 Running Roboflow detection...")
        try:
            result = ROBOFLOW_CLIENT.infer(tmp_path, model_id="main-fashion-wmyfk/1")
            print(f"✅ Roboflow API response received")
            print(f"📊 Raw result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            
            if "predictions" not in result:
                print(f"❌ No 'predictions' key in result: {result}")
                os.unlink(tmp_path)
                return {"success": False, "error": "Invalid response from detection API"}
                
            predictions = result["predictions"]
            print(f"🎯 Found {len(predictions)} predictions")
            
        except Exception as api_error:
            print(f"❌ Roboflow API error: {str(api_error)}")
            os.unlink(tmp_path)
            return {"success": False, "error": f"Detection API failed: {str(api_error)}"}
        
        # Process image
        print("🖼️ Processing image...")
        try:
            image_pil = Image.open(tmp_path)
            original_size = image_pil.size
            print(f"📏 Original image size: {original_size}")
            
            # Resize smartly
            image_pil = resize_image_smart(image_pil, 1024)
            new_size = image_pil.size
            print(f"📏 Resized image size: {new_size}")
            
            image_rgb = np.array(image_pil)
            print(f"🎨 Converted to RGB array: {image_rgb.shape}")
            
        except Exception as process_error:
            print(f"❌ Image processing error: {str(process_error)}")
            os.unlink(tmp_path)
            return {"success": False, "error": f"Image processing failed: {str(process_error)}"}
        
        # Process detections
        print("🏷️ Processing detections...")
        detected_items = []
        
        # Calculate scaling factors
        scale_x = new_size[0] / original_size[0] if original_size[0] > 0 else 1
        scale_y = new_size[1] / original_size[1] if original_size[1] > 0 else 1
        print(f"📐 Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")
        
        for i, pred in enumerate(predictions):
            print(f"🔍 Processing prediction {i}: {pred}")
            
            if "class" not in pred:
                print(f"⚠️ Skipping prediction {i}: no 'class' field")
                continue
                
            region = classify_clothing_region(pred["class"])
            print(f"📍 Classified '{pred['class']}' as '{region}'")
            
            if region in ['upper', 'lower']:
                try:
                    # Calculate scaled bbox
                    bbox = [
                        int((pred["x"] - pred["width"] / 2) * scale_x),
                        int((pred["y"] - pred["height"] / 2) * scale_y),
                        int((pred["x"] + pred["width"] / 2) * scale_x),
                        int((pred["y"] + pred["height"] / 2) * scale_y)
                    ]
                    
                    detected_items.append({
                        "id": i,
                        "class": pred["class"],
                        "confidence": pred.get("confidence", 0.0),
                        "region": region,
                        "bbox": bbox
                    })
                    print(f"✅ Added item: {pred['class']} with bbox {bbox}")
                    
                except Exception as bbox_error:
                    print(f"❌ Error processing bbox for prediction {i}: {bbox_error}")
                    continue
            else:
                print(f"⏭️ Skipping '{pred['class']}' (region: {region})")
        
        print(f"📋 Final detected items: {len(detected_items)}")
        
        # Create annotated image
        try:
            if detected_items:
                print("🎨 Creating annotated image...")
                xyxy = np.array([item["bbox"] for item in detected_items])
                detections = sv.Detections(
                    xyxy=xyxy,
                    confidence=np.array([item["confidence"] for item in detected_items]),
                    class_id=np.array([item["id"] for item in detected_items])
                )
                
                box_annotator = sv.BoxAnnotator(thickness=3)
                annotated_image = box_annotator.annotate(scene=image_rgb.copy(), detections=detections)
                
                # Add labels
                for i, item in enumerate(detected_items):
                    x_min, y_min, _, _ = item["bbox"]
                    label = f"{item['class']} ({item['region']}) {item['confidence']*100:.1f}%"
                    
                    # Add background for text
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(annotated_image, (x_min, y_min-30), (x_min + text_size[0] + 10, y_min), (0, 0, 0), -1)
                    cv2.putText(annotated_image, label, (x_min + 5, y_min-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                print("✅ Annotation complete")
            else:
                print("⚠️ No detected items, using original image")
                annotated_image = image_rgb
            
            # Convert to base64
            print("🔄 Converting to base64...")
            annotated_b64 = image_to_base64(annotated_image)
            original_b64 = image_to_base64(image_rgb)
            print("✅ Base64 conversion complete")
            
        except Exception as annotation_error:
            print(f"❌ Annotation error: {str(annotation_error)}")
            # Fallback to original image
            annotated_b64 = image_to_base64(image_rgb)
            original_b64 = image_to_base64(image_rgb)
        
        # Clean up
        os.unlink(tmp_path)
        print("🗑️ Temporary file cleaned up")
        
        result_data = {
            "success": True,
            "detected_items": detected_items,
            "annotated_image": f"data:image/png;base64,{annotated_b64}",
            "original_image": f"data:image/png;base64,{original_b64}",
            "debug_info": {
                "original_size": original_size,
                "processed_size": new_size,
                "total_predictions": len(predictions),
                "clothing_items_found": len(detected_items)
            }
        }
        
        print(f"🎉 Detection complete! Found {len(detected_items)} clothing items")
        print(f"📊 Response size - annotated: {len(annotated_b64)} chars, original: {len(original_b64)} chars")
        return result_data
    
    except Exception as e:
        print(f"💥 Unexpected error in detect_clothing: {str(e)}")
        import traceback
        print(f"📍 Traceback: {traceback.format_exc()}")
        return {"success": False, "error": f"Detection failed: {str(e)}"}

@app.post("/generate-mask/")
async def generate_mask(
    image_data: str = Form(...),
    selected_item: str = Form(...)
):
    """Generate improved segmentation mask for selected clothing item"""
    try:
        item_data = json.loads(selected_item)
        bbox = item_data["bbox"]
        
        # Decode image
        image_b64 = image_data.split(',')[1]
        image_rgb = base64_to_image(image_b64)
        
        # Save temporary image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(tmp_file.name, image_bgr)
            tmp_path = tmp_file.name
        
        # Generate mask using SAM with improved settings
        model = load_sam_model()
        
        # Add some padding to bbox for better segmentation
        x1, y1, x2, y2 = bbox
        padding = 10
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image_rgb.shape[1], x2 + padding)
        y2 = min(image_rgb.shape[0], y2 + padding)
        padded_bbox = [x1, y1, x2, y2]
        
        segment_result = model(tmp_path, bboxes=[padded_bbox])
        
        if segment_result and len(segment_result) > 0 and segment_result[0].masks is not None:
            # Get the mask with highest confidence
            masks = segment_result[0].masks.data.cpu().numpy()
            if len(masks) > 0:
                # Use the first (usually best) mask
                raw_mask = masks[0]
                binary_mask = (raw_mask > 0.5).astype(np.uint8) * 255
                
                # Improve mask quality
                improved_mask = improve_mask_quality(binary_mask, dilate_iterations=1, blur_radius=1)
                
                # Convert to 3-channel for consistency
                if len(improved_mask.shape) == 2:
                    improved_mask_rgb = cv2.cvtColor(improved_mask, cv2.COLOR_GRAY2RGB)
                else:
                    improved_mask_rgb = improved_mask
                
                mask_b64 = image_to_base64(improved_mask_rgb)
                
                os.unlink(tmp_path)
                
                return {
                    "success": True,
                    "mask_image": f"data:image/png;base64,{mask_b64}"
                }
            else:
                return {"success": False, "error": "No masks generated"}
        else:
            return {"success": False, "error": "Failed to generate mask"}
    
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/inpaint/")
async def inpaint_image(
    original_image: str = Form(...),
    mask_image: str = Form(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    strength: float = Form(0.85),
    clothing_class: str = Form(""),
    region: str = Form("")
):
    """Perform high-quality inpainting with enhanced parameters"""
    try:
        # Decode images
        original_b64 = original_image.split(',')[1]
        mask_b64 = mask_image.split(',')[1]
        
        # Convert to PIL for better processing
        original_pil = base64_to_pil(original_b64)
        mask_pil = base64_to_pil(mask_b64)
        
        # Ensure both images are the same size
        if original_pil.size != mask_pil.size:
            mask_pil = mask_pil.resize(original_pil.size, Image.Resampling.LANCZOS)
        
        # Enhance the original image slightly
        enhancer = ImageEnhance.Sharpness(original_pil)
        original_pil = enhancer.enhance(1.1)
        
        # Process mask - ensure it's grayscale with smooth edges
        mask_gray = mask_pil.convert('L')
        mask_smooth = mask_gray.filter(ImageFilter.GaussianBlur(radius=1))
        mask_rgb = mask_smooth.convert('RGB')
        
        # Convert back to base64
        enhanced_original_b64 = pil_to_base64(original_pil)
        enhanced_mask_b64 = pil_to_base64(mask_rgb)
        
        # Enhance prompt based on clothing type
        if clothing_class:
            enhanced_prompt = enhance_prompt_for_clothing(prompt, clothing_class, region)
        else:
            enhanced_prompt = f"{prompt}, photorealistic, high quality, detailed, natural lighting"
        
        # Enhanced negative prompt
        enhanced_negative = f"{negative_prompt}, low quality, blurry, distorted, ugly, bad anatomy, unrealistic, artificial, plastic, synthetic, cartoonish, anime, drawing, sketch"
        
        # Optimized API parameters for better quality
        data = {
            "image": enhanced_original_b64,
            "mask": enhanced_mask_b64,
            "prompt": enhanced_prompt,
            "negative_prompt": enhanced_negative,
            "samples": 1,
            "scheduler": "DPM++_2M_Karras",  # Better scheduler for quality
            "num_inference_steps": 30,  # More steps for better quality
            "guidance_scale": 8.0,  # Slightly higher for better prompt adherence
            "seed": np.random.randint(1000, 99999),
            "strength": min(max(strength, 0.7), 0.95),  # Clamp strength for better results
            "base64": True
        }
        
        headers = {
            'x-api-key': SEGMIND_API_KEY,
            'Content-Type': 'application/json'
        }
        
        # Make API call with extended timeout
        for attempt in range(3):
            try:
                response = requests.post(
                    SEGMIND_URL, 
                    json=data, 
                    headers=headers, 
                    timeout=120
                )
                
                if response.status_code == 200:
                    # The API returns base64 directly when base64=True
                    try:
                        # Try to parse JSON response first
                        result_data = response.json()
                        if 'image' in result_data:
                            result_b64 = result_data['image']
                        else:
                            result_b64 = result_data
                    except:
                        # If not JSON, treat as direct base64
                        result_b64 = base64.b64encode(response.content).decode('utf-8')
                    
                    # Post-process the result for better quality
                    try:
                        result_pil = base64_to_pil(result_b64)
                        
                        # Slight enhancement
                        color_enhancer = ImageEnhance.Color(result_pil)
                        result_pil = color_enhancer.enhance(1.05)
                        
                        contrast_enhancer = ImageEnhance.Contrast(result_pil)
                        result_pil = contrast_enhancer.enhance(1.02)
                        
                        final_b64 = pil_to_base64(result_pil)
                    except:
                        final_b64 = result_b64
                    
                    credits = response.headers.get('x-remaining-credits', 'N/A')
                    
                    return {
                        "success": True,
                        "result_image": f"data:image/png;base64,{final_b64}",
                        "credits_remaining": credits,
                        "enhanced_prompt": enhanced_prompt
                    }
                
                elif response.status_code == 402:
                    return {"success": False, "error": "Out of credits"}
                else:
                    try:
                        error_detail = response.json()
                    except:
                        error_detail = response.text
                    
                    if attempt < 2:
                        continue
                    
                    return {
                        "success": False,
                        "error": f"API error {response.status_code}: {error_detail}"
                    }
            
            except requests.exceptions.Timeout:
                if attempt < 2:
                    continue
                return {"success": False, "error": "Request timed out. Please try again."}
            except Exception as e:
                if attempt < 2:
                    continue
                return {"success": False, "error": f"Request failed: {str(e)}"}
    
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/health")
async def health_check():
    """Health check endpoint with detailed status"""
    try:
        import datetime
        status = {
            "status": "healthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "services": {}
        }
        
        # Check SAM model
        try:
            global sam_model
            if sam_model is not None:
                status["services"]["sam_model"] = "loaded"
            else:
                status["services"]["sam_model"] = "not_loaded"
        except:
            status["services"]["sam_model"] = "error"
        
        # Check directories
        status["services"]["temp_dir"] = "exists" if Path("temp").exists() else "missing"
        status["services"]["static_dir"] = "exists" if Path("static").exists() else "missing"
        
        # Check API keys
        status["services"]["roboflow_api"] = "configured"
        status["services"]["segmind_api"] = "configured" if SEGMIND_API_KEY else "missing"
        
        return status
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/test-roboflow")
async def test_roboflow():
    """Test Roboflow API connection"""
    try:
        # Create a simple test image
        test_image = Image.new('RGB', (300, 300), color='white')
        test_path = "temp/test_image.jpg"
        test_image.save(test_path)
        
        print("🧪 Testing Roboflow API with test image...")
        result = ROBOFLOW_CLIENT.infer(test_path, model_id="main-fashion-wmyfk/1")
        
        # Clean up
        os.unlink(test_path)
        
        return {
            "success": True,
            "message": "Roboflow API is working",
            "result_keys": list(result.keys()) if isinstance(result, dict) else "Not a dict",
            "predictions_count": len(result.get("predictions", [])) if isinstance(result, dict) else 0
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Roboflow API test failed"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)