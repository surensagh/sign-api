# FastAPI Text-to-Sign API - Converted from Flask
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import os
import uuid
import time
import base64
import json
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, WebDriverException
import logging
import asyncio
from typing import Optional, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app instance
app = FastAPI(
    title="Text-to-Sign Translation API",
    description="Convert text to sign language videos using sign.mt",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class TranslationRequest(BaseModel):
    text: str
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        v = v.strip()
        if not v:
            raise ValueError('Text cannot be empty')
        if len(v) > 500:
            raise ValueError('Text too long (max 500 characters)')
        return v

class TranslationResponse(BaseModel):
    translation_id: str
    status: str
    message: str

class TranslationStatus(BaseModel):
    translation_id: str
    status: str
    text: str
    download_url: Optional[str] = None
    error: Optional[str] = None

class APIStatus(BaseModel):
    status: str
    active_translations: int
    total_translations: int
    upload_folder: str

class HealthCheck(BaseModel):
    status: str
    timestamp: float

# Global storage (in production, use Redis or database)
translations: Dict[str, Dict[str, Any]] = {}
UPLOAD_FOLDER = './sign_translations'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def setup_driver(headless=True):
    """Set up Chrome driver with working options"""
    options = Options()
    if headless:
        options.add_argument("--headless")
    
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--enable-logging")
    options.add_argument("--log-level=3")
    options.add_argument("--disable-background-timer-throttling")
    options.add_argument("--disable-backgrounding-occluded-windows")
    options.add_argument("--disable-renderer-backgrounding")
    
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    options.add_experimental_option('useAutomationExtension', False)
    options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})
    
    try:
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(30)
        driver.set_script_timeout(20)
        return driver
    except Exception as e:
        logger.error(f"Failed to create driver: {e}")
        raise

def find_text_input(driver):
    """Find text input using the exact same logic from original code"""
    logger.info("Looking for text input field...")
    text_input = None
    input_selectors = [
        "textarea[placeholder*='text' i]",
        "textarea[placeholder*='Text' i]", 
        "textarea",
        "input[type='text']",
        ".text-input",
        "#text-input",
        "[contenteditable='true']",
        "input[placeholder*='text' i]",
        "input[placeholder*='word' i]"
    ]
    
    for i, selector in enumerate(input_selectors):
        try:
            logger.info(f"  Trying selector {i+1}/{len(input_selectors)}: {selector}")
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            logger.info(f"    Found {len(elements)} elements")
            
            for element in elements:
                if element.is_displayed() and element.is_enabled():
                    text_input = element
                    logger.info(f"    Found usable input field with selector: {selector}")
                    return text_input
                    
        except Exception as e:
            logger.warning(f"    Error with selector {selector}: {e}")
            continue
    
    return text_input

def trigger_translation(driver, text_input, text):
    """Trigger translation using the same methods from original code"""
    text_input.clear()
    text_input.send_keys(text)
    logger.info(f"Entered text: {text}")
    
    time.sleep(2)
    translate_triggered = False
    
    # Method 1: Look for translate button
    try:
        translate_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Translate') or contains(text(), 'translate')]")
        if translate_button.is_displayed():
            translate_button.click()
            translate_triggered = True
            logger.info("Clicked translate button")
    except:
        pass
    
    # Method 2: Press Enter
    if not translate_triggered:
        try:
            text_input.send_keys(Keys.RETURN)
            translate_triggered = True
            logger.info("Pressed Enter to translate")
        except:
            pass
    
    return translate_triggered

def find_all_blobs(driver):
    """Use the exact blob detection from original code"""
    blob_urls = []
    
    # Method 1: JavaScript execution to find all blob URLs
    try:
        blob_script = """
        var elements = document.querySelectorAll('*');
        var blobs = [];
        for (var i = 0; i < elements.length; i++) {
            var el = elements[i];
            if (el.src && el.src.startsWith('blob:')) {
                blobs.push(el.src);
            }
            if (el.href && el.href.startsWith('blob:')) {
                blobs.push(el.href);
            }
            var style = window.getComputedStyle(el);
            if (style.backgroundImage && style.backgroundImage.includes('blob:')) {
                var match = style.backgroundImage.match(/blob:[^)]+/);
                if (match) blobs.push(match[0]);
            }
        }
        return [...new Set(blobs)]; // Remove duplicates
        """
        
        page_blobs = driver.execute_script(blob_script)
        blob_urls.extend(page_blobs)
        logger.info(f"Found {len(page_blobs)} blobs via JavaScript")
        
    except Exception as e:
        logger.error(f"Error finding blobs via JavaScript: {e}")
    
    # Method 2: Element scanning
    try:
        media_elements = driver.find_elements(By.CSS_SELECTOR, "video, img, source, canvas")
        logger.info(f"Scanning {len(media_elements)} media elements...")
        
        for element in media_elements:
            try:
                attributes_to_check = ["src", "data-src", "srcset", "data-url"]
                for attr in attributes_to_check:
                    attr_value = element.get_attribute(attr)
                    if attr_value and attr_value.startswith('blob:'):
                        blob_urls.append(attr_value)
            except:
                continue
                
    except Exception as e:
        logger.error(f"Error scanning media elements: {e}")
    
    # Method 3: Performance logs
    try:
        logs = driver.get_log('performance')
        for log in logs:
            message = json.loads(log['message'])
            if message['message']['method'] == 'Network.responseReceived':
                url = message['message']['params']['response']['url']
                if url.startswith('blob:'):
                    blob_urls.append(url)
    except Exception as e:
        logger.error(f"Error checking performance logs: {e}")
    
    unique_blobs = list(set(blob_urls))
    if unique_blobs:
        logger.info(f"Total unique blobs found: {len(unique_blobs)}")
        for i, blob in enumerate(unique_blobs):
            logger.info(f"  Blob {i+1}: {blob}")
    
    return unique_blobs

def trigger_page_interactions(driver):
    """Use the exact interaction triggers from original code"""
    try:
        body = driver.find_element(By.TAG_NAME, "body")
        body.click()
        time.sleep(1)
        
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(1)
        
        clickable_elements = driver.find_elements(By.CSS_SELECTOR, "video, canvas, .video-container, .translation-result")
        for elem in clickable_elements[:3]:
            try:
                if elem.is_displayed():
                    driver.execute_script("arguments[0].click();", elem)
                    time.sleep(0.5)
            except:
                continue
        
        time.sleep(3)
        
    except Exception as e:
        logger.error(f"Error triggering page interactions: {e}")

def download_content_from_session(driver, url, word, save_path):
    """Use the exact download logic from original code"""
    try:
        logger.info(f"Downloading content from: {url}")
        
        if url.startswith('blob:'):
            blob_script = f"""
            return new Promise((resolve) => {{
                const timeoutId = setTimeout(() => resolve(null), 15000);
                
                fetch('{url}')
                    .then(response => {{
                        if (!response.ok) throw new Error('Network response was not ok');
                        return response.blob();
                    }})
                    .then(blob => {{
                        const reader = new FileReader();
                        reader.onloadend = function() {{
                            clearTimeout(timeoutId);
                            resolve(reader.result);
                        }};
                        reader.onerror = function(err) {{
                            clearTimeout(timeoutId);
                            resolve(null);
                        }};
                        reader.readAsDataURL(blob);
                    }})
                    .catch(err => {{
                        clearTimeout(timeoutId);
                        resolve(null);
                    }});
            }});
            """
            
            try:
                base64_data = driver.execute_async_script(blob_script)
            except Exception as e:
                logger.error(f"Error executing blob download script: {e}")
                simple_blob_script = f"""
                var callback = arguments[0];
                fetch('{url}')
                    .then(r => r.blob())
                    .then(blob => {{
                        var reader = new FileReader();
                        reader.onload = () => callback(reader.result);
                        reader.onerror = () => callback(null);
                        reader.readAsDataURL(blob);
                    }})
                    .catch(() => callback(null));
                """
                try:
                    base64_data = driver.execute_async_script(simple_blob_script)
                except:
                    base64_data = None
            
            if base64_data and base64_data.startswith('data:'):
                header, data = base64_data.split(',', 1)
                
                if 'image/gif' in header:
                    ext = '.gif'
                elif 'video/mp4' in header:
                    ext = '.mp4'
                elif 'video/webm' in header:
                    ext = '.webm'
                else:
                    ext = '.webm'
                
                safe_word = "".join(c for c in word if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_word = safe_word.replace(' ', '_')
                filename = f"{safe_word}_sign_translation_{uuid.uuid4().hex[:8]}{ext}"
                filepath = os.path.join(save_path, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(base64.b64decode(data))
                
                file_size = os.path.getsize(filepath)
                logger.info(f"Successfully saved blob content to: {filepath} ({file_size} bytes)")
                return filepath
            else:
                logger.warning("Could not extract valid base64 data from blob")
                return None
        
        else:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                ext = '.gif' if '.gif' in url else '.mp4'
                safe_word = "".join(c for c in word if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_word = safe_word.replace(' ', '_')
                filename = f"{safe_word}_sign_translation_{uuid.uuid4().hex[:8]}{ext}"
                filepath = os.path.join(save_path, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"Successfully downloaded to: {filepath}")
                return filepath
        
    except Exception as e:
        logger.error(f"Error downloading content: {e}")
    
    return None

def get_additional_urls(driver):
    """Get video and GIF URLs from performance logs"""
    video_urls = []
    gif_urls = []
    
    try:
        logs = driver.get_log('performance')
        for log in logs:
            message = json.loads(log['message'])
            if message['message']['method'] == 'Network.responseReceived':
                url = message['message']['params']['response']['url']
                mime_type = message['message']['params']['response'].get('mimeType', '')
                
                if any(skip in url.lower() for skip in ['manifest', 'favicon', '.css', '.js', '.html', '.txt']):
                    continue
                
                if any(ext in url.lower() for ext in ['.gif', '.mp4', '.webm', '.mov']):
                    if '.gif' in url.lower():
                        gif_urls.append(url)
                    else:
                        video_urls.append(url)
                elif any(mime in mime_type.lower() for mime in ['video/', 'image/gif']):
                    if 'gif' in mime_type.lower():
                        gif_urls.append(url)
                    else:
                        video_urls.append(url)
    except Exception as e:
        logger.error(f"Error reading performance logs: {e}")
    
    return list(set(video_urls)), list(set(gif_urls))

def translate_text_to_sign(text, save_path, max_retries=3):
    """Main translation function using original code logic"""
    driver = None
    try:
        logger.info(f"Starting translation for: '{text}'")
        driver = setup_driver(headless=True)
        
        logger.info("Navigating to sign.mt...")
        driver.get("https://sign.mt/")
        
        wait = WebDriverWait(driver, 15)
        logger.info("Page loaded, waiting 3 seconds...")
        time.sleep(3)
        
        logger.info(f"Current page title: {driver.title}")
        logger.info(f"Current URL: {driver.current_url}")
        
        text_input = find_text_input(driver)
        if not text_input:
            raise Exception("Could not find text input field")
        
        if not trigger_translation(driver, text_input, text):
            logger.warning("Could not explicitly trigger translation, relying on auto-translation")
        
        word_count = len(text.split())
        base_wait_time = max(8, word_count * 3)
        logger.info(f"Waiting for translation ({word_count} words, {base_wait_time}s base wait)...")
        
        blob_urls = []
        for attempt in range(max_retries):
            logger.info(f"Blob detection attempt {attempt + 1}/{max_retries}")
            
            wait_time = base_wait_time + (attempt * 5)
            
            for i in range(wait_time):
                time.sleep(1)
                if i % 3 == 0:
                    try:
                        video_elements = driver.find_elements(By.CSS_SELECTOR, "video, img[src*='blob:']")
                        if any(elem.get_attribute("src") and elem.get_attribute("src").startswith("blob:") for elem in video_elements):
                            logger.info(f"  Translation appears ready after {i+1}s (attempt {attempt + 1})")
                            break
                    except:
                        pass
            
            blob_urls = find_all_blobs(driver)
            
            if blob_urls:
                logger.info(f"Found {len(blob_urls)} blob URLs on attempt {attempt + 1}")
                break
            else:
                logger.info(f"No blobs found on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    logger.info("Triggering page interactions and retrying...")
                    trigger_page_interactions(driver)
        
        video_urls, gif_urls = get_additional_urls(driver)
        
        primary_url = None
        if blob_urls:
            primary_url = blob_urls[0]
        elif gif_urls:
            primary_url = gif_urls[0]
        elif video_urls:
            primary_url = video_urls[0]
        
        if not primary_url:
            raise Exception("No video content found after translation")
        
        logger.info(f"Primary URL found: {primary_url}")
        
        filepath = download_content_from_session(driver, primary_url, text, save_path)
        
        if not filepath:
            raise Exception("Failed to download video content")
        
        return filepath
        
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass

async def process_translation_async(translation_id: str, text: str):
    """Process translation in background - runs in thread pool"""
    try:
        logger.info(f"Processing translation {translation_id}: {text}")
        
        # Run the blocking Selenium operation in a thread pool
        loop = asyncio.get_event_loop()
        file_path = await loop.run_in_executor(
            None, 
            translate_text_to_sign, 
            text, 
            UPLOAD_FOLDER
        )
        
        if file_path and os.path.exists(file_path):
            translations[translation_id].update({
                'status': 'completed',
                'file_path': file_path
            })
            logger.info(f"Translation {translation_id} completed successfully")
        else:
            translations[translation_id].update({
                'status': 'failed',
                'error': 'Failed to generate or save video file'
            })
            logger.error(f"Translation {translation_id} failed - no file generated")
            
    except Exception as e:
        error_msg = str(e)
        translations[translation_id].update({
            'status': 'failed',
            'error': error_msg
        })
        logger.error(f"Translation {translation_id} failed with error: {error_msg}")

# FastAPI Routes
@app.post("/api/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest, background_tasks: BackgroundTasks):
    """Start a new text-to-sign translation"""
    text = request.text
    translation_id = str(uuid.uuid4())
    
    translations[translation_id] = {
        'status': 'processing',
        'text': text,
        'file_path': None,
        'error': None,
        'created_at': time.time()
    }

    # Add the async task to background tasks
    background_tasks.add_task(process_translation_async, translation_id, text)

    return TranslationResponse(
        translation_id=translation_id,
        status='processing',
        message='Translation started'
    )

@app.get("/api/translation/{translation_id}", response_model=TranslationStatus)
async def get_translation_status(translation_id: str):
    """Get the status of a translation"""
    if translation_id not in translations:
        raise HTTPException(status_code=404, detail='Translation not found')

    translation = translations[translation_id]
    response = TranslationStatus(
        translation_id=translation_id,
        status=translation['status'],
        text=translation['text']
    )

    if translation['status'] == 'completed':
        response.download_url = f"/api/download/{translation_id}"
    elif translation['status'] == 'failed':
        response.error = translation['error']

    return response

@app.get("/api/download/{translation_id}")
async def download_translation(translation_id: str):
    """Download the translated sign video"""
    if translation_id not in translations:
        raise HTTPException(status_code=404, detail='Translation not found')

    translation = translations[translation_id]
    if translation['status'] != 'completed' or not translation['file_path']:
        raise HTTPException(status_code=400, detail='Translation not ready')

    if not os.path.exists(translation['file_path']):
        raise HTTPException(status_code=404, detail='File not found')

    return FileResponse(
        path=translation['file_path'],
        media_type='video/webm',
        filename=os.path.basename(translation['file_path'])
    )

@app.get("/api/status", response_model=APIStatus)
async def api_status():
    """Get API status information"""
    return APIStatus(
        status='online',
        active_translations=len([t for t in translations.values() if t['status'] == 'processing']),
        total_translations=len(translations),
        upload_folder=UPLOAD_FOLDER
    )

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(status='healthy', timestamp=time.time())

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Text-to-Sign Translation API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == '__main__':
    import uvicorn
    logger.info("Starting FastAPI Text-to-Sign API...")
    uvicorn.run(
        "api:app",  # Changed from "main:app" to "api:app"
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        workers=1  # Keep at 1 due to Selenium driver management
    )