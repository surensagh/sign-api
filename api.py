# FastAPI Text-to-Sign API - Fixed Version with Better Error Handling
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import os
import uuid
import time
import base64
import json
import requests
import asyncio
import aiohttp
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import tempfile
import shutil
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment variables
class Config:
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    USE_S3 = bool(AWS_S3_BUCKET and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)
    MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "500"))
    API_TIMEOUT = int(os.getenv("API_TIMEOUT", "120"))
    SELENIUM_GRID_URL = os.getenv("SELENIUM_GRID_URL")
    BROWSERLESS_URL = os.getenv("BROWSERLESS_URL")
    BROWSERLESS_API_KEY = os.getenv("BROWSERLESS_API_KEY")
    USE_PLAYWRIGHT = os.getenv("USE_PLAYWRIGHT", "false").lower() == "true"

config = Config()

# Optional imports based on configuration
storage_backend = None
redis_client = None

try:
    if config.USE_S3:
        import boto3
        s3_client = boto3.client(
            's3',
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            region_name=config.AWS_REGION
        )
        storage_backend = "s3"
        logger.info("Using S3 for file storage")
    else:
        storage_backend = "local"
        logger.info("Using local file storage")
except ImportError:
    storage_backend = "local"
    logger.warning("boto3 not available, falling back to local storage")

try:
    import redis
    redis_client = redis.from_url(config.REDIS_URL, decode_responses=True)
    # Test connection
    redis_client.ping()
    logger.info("Connected to Redis for state management")
except (ImportError, redis.ConnectionError):
    logger.warning("Redis not available, using in-memory storage")
    redis_client = None

# In-memory fallback storage
translations: Dict[str, Dict[str, Any]] = {}

# Pydantic models
class TranslationRequest(BaseModel):
    text: str
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        v = v.strip()
        if not v:
            raise ValueError('Text cannot be empty')
        if len(v) > config.MAX_TEXT_LENGTH:
            raise ValueError(f'Text too long (max {config.MAX_TEXT_LENGTH} characters)')
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
    storage_backend: str
    state_backend: str

class HealthCheck(BaseModel):
    status: str
    timestamp: float
    version: str

# Storage utilities
class StorageManager:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="sign_translations_")
        
    async def save_file(self, file_content: bytes, filename: str) -> str:
        """Save file and return download URL or file path"""
        if storage_backend == "s3":
            try:
                s3_client.put_object(
                    Bucket=config.AWS_S3_BUCKET,
                    Key=filename,
                    Body=file_content,
                    ContentType='video/webm'
                )
                return f"s3://{config.AWS_S3_BUCKET}/{filename}"
            except Exception as e:
                logger.error(f"S3 upload failed: {e}")
                # Fallback to local
                pass
        
        # Local storage
        file_path = Path(self.temp_dir) / filename
        with open(file_path, 'wb') as f:
            f.write(file_content)
        return str(file_path)
    
    async def get_file(self, file_path: str) -> Optional[bytes]:
        """Retrieve file content"""
        if file_path.startswith("s3://"):
            try:
                bucket, key = file_path[5:].split("/", 1)
                response = s3_client.get_object(Bucket=bucket, Key=key)
                return response['Body'].read()
            except Exception as e:
                logger.error(f"S3 download failed: {e}")
                return None
        else:
            try:
                with open(file_path, 'rb') as f:
                    return f.read()
            except FileNotFoundError:
                return None

# State management utilities
class StateManager:
    @staticmethod
    async def get_translation(translation_id: str) -> Optional[Dict[str, Any]]:
        if redis_client:
            try:
                data = redis_client.hgetall(f"translation:{translation_id}")
                if data:
                    # Convert string timestamps back to float
                    if 'created_at' in data:
                        data['created_at'] = float(data['created_at'])
                    return data
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        
        return translations.get(translation_id)
    
    @staticmethod
    async def set_translation(translation_id: str, data: Dict[str, Any]):
        if redis_client:
            try:
                # Convert non-string values for Redis
                redis_data = {k: str(v) if v is not None else '' for k, v in data.items()}
                redis_client.hset(f"translation:{translation_id}", mapping=redis_data)
                redis_client.expire(f"translation:{translation_id}", 86400)  # 24 hours
                return
            except Exception as e:
                logger.error(f"Redis set error: {e}")
        
        translations[translation_id] = data
    
    @staticmethod
    async def get_all_translations() -> Dict[str, Dict[str, Any]]:
        if redis_client:
            try:
                keys = redis_client.keys("translation:*")
                result = {}
                for key in keys:
                    translation_id = key.split(":", 1)[1]
                    data = redis_client.hgetall(key)
                    if data and 'created_at' in data:
                        data['created_at'] = float(data['created_at'])
                    result[translation_id] = data
                return result
            except Exception as e:
                logger.error(f"Redis get_all error: {e}")
        
        return translations

# Playwright automation (local)
class PlaywrightAutomation:
    def __init__(self):
        self.playwright = None
        
    async def initialize(self):
        """Initialize Playwright"""
        try:
            from playwright.async_api import async_playwright
            self.playwright = await async_playwright().start()
            return True
        except ImportError:
            logger.error("Playwright not installed. Run: pip install playwright && playwright install chromium")
            return False
    
    async def cleanup(self):
        """Cleanup Playwright"""
        if self.playwright:
            await self.playwright.stop()
    
    async def translate_text_to_sign(self, text: str) -> Optional[bytes]:
        """Use Playwright for browser automation"""
        if not self.playwright:
            if not await self.initialize():
                return None
            
        try:
            browser = await self.playwright.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )
            page = await browser.new_page()
            
            try:
                logger.info("Navigating to sign.mt...")
                await page.goto('https://sign.mt/', wait_until='networkidle', timeout=30000)
                await page.wait_for_timeout(3000)
                
                # Find and fill text input
                logger.info("Finding text input...")
                text_input = await page.wait_for_selector('textarea, input[type="text"], [contenteditable="true"]', timeout=10000)
                await text_input.click()
                await page.keyboard.select_all()
                await page.keyboard.type(text)
                
                # Trigger translation
                logger.info("Triggering translation...")
                try:
                    translate_btn = await page.query_selector('button[type="submit"], button:has-text("Translate"), .translate-btn, #translate')
                    if translate_btn:
                        await translate_btn.click()
                    else:
                        await page.keyboard.press('Enter')
                except:
                    await page.keyboard.press('Enter')
                
                # Wait for video generation
                logger.info("Waiting for video generation...")
                await page.wait_for_timeout(15000)
                
                # Find blob URLs
                blob_urls = await page.evaluate('''
                    () => {
                        const elements = document.querySelectorAll('*');
                        const blobs = [];
                        for (let i = 0; i < elements.length; i++) {
                            const el = elements[i];
                            if (el.src && el.src.startsWith('blob:')) {
                                blobs.push(el.src);
                            }
                            if (el.href && el.href.startsWith('blob:')) {
                                blobs.push(el.href);
                            }
                        }
                        return blobs;
                    }
                ''')
                
                logger.info(f"Found {len(blob_urls)} blob URLs")
                
                if not blob_urls:
                    logger.error("No video content found")
                    return None
                
                # Download blob content
                logger.info("Downloading video content...")
                video_content = await page.evaluate(f'''
                    async () => {{
                        try {{
                            const response = await fetch('{blob_urls[0]}');
                            if (!response.ok) throw new Error('Fetch failed');
                            const blob = await response.blob();
                            const arrayBuffer = await blob.arrayBuffer();
                            return Array.from(new Uint8Array(arrayBuffer));
                        }} catch (error) {{
                            console.error('Download error:', error);
                            throw error;
                        }}
                    }}
                ''')
                
                logger.info(f"Downloaded {len(video_content)} bytes")
                return bytes(video_content)
                
            finally:
                await browser.close()
                
        except Exception as e:
            logger.error(f"Playwright automation failed: {e}")
            return None

# Browser automation using HTTP API (Browserless)
class BrowserlessAutomation:
    def __init__(self):
        self.base_url = config.BROWSERLESS_URL or "http://localhost:3000"
        self.api_key = config.BROWSERLESS_API_KEY
        
    async def translate_text_to_sign(self, text: str) -> Optional[bytes]:
        """Use browserless.io or similar service for browser automation"""
        
        # Escape text for JavaScript
        escaped_text = text.replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n')
        
        script = f'''
        const page = await browser.newPage();
        
        try {{
            console.log("Navigating to sign.mt...");
            await page.goto('https://sign.mt/', {{ 
                waitUntil: 'networkidle2',
                timeout: 30000 
            }});
            
            console.log("Page loaded, waiting for elements...");
            await page.waitForTimeout(3000);
            
            // Find and fill text input with better selectors
            const textInput = await page.waitForSelector('textarea, input[type="text"], [contenteditable="true"]', {{
                timeout: 10000
            }});
            
            console.log("Found text input, filling with text...");
            await textInput.click();
            await page.keyboard.selectAll();
            await page.keyboard.type('{escaped_text}');
            
            // Try multiple ways to trigger translation
            console.log("Triggering translation...");
            try {{
                // Look for translate button with various selectors
                const translateBtn = await page.$('button[type="submit"], button:contains("Translate"), .translate-btn, #translate');
                if (translateBtn) {{
                    await translateBtn.click();
                }} else {{
                    // Fallback to Enter key
                    await page.keyboard.press('Enter');
                }}
            }} catch (e) {{
                console.log("Using Enter key fallback");
                await page.keyboard.press('Enter');
            }}
            
            console.log("Waiting for video generation...");
            // Wait longer for video content
            await page.waitForTimeout(15000);
            
            // Look for video elements more thoroughly
            const videoContent = await page.evaluate(() => {{
                // Check for various video-related elements
                const videoElements = document.querySelectorAll('video, [src*="blob:"], [href*="blob:"]');
                const blobUrls = [];
                
                videoElements.forEach(el => {{
                    if (el.src && el.src.startsWith('blob:')) {{
                        blobUrls.push(el.src);
                    }}
                    if (el.href && el.href.startsWith('blob:')) {{
                        blobUrls.push(el.href);
                    }}
                }});
                
                // Also check for data URLs or other video sources
                const allElements = document.querySelectorAll('*');
                allElements.forEach(el => {{
                    ['src', 'href', 'data-src'].forEach(attr => {{
                        const value = el.getAttribute(attr);
                        if (value && (value.startsWith('blob:') || value.startsWith('data:video/'))) {{
                            blobUrls.push(value);
                        }}
                    }});
                }});
                
                return {{ blobUrls, pageContent: document.body.innerHTML.substring(0, 1000) }};
            }});
            
            console.log(`Found ${{videoContent.blobUrls.length}} blob URLs`);
            
            if (videoContent.blobUrls.length === 0) {{
                console.log("Page content:", videoContent.pageContent);
                throw new Error('No video content found after translation');
            }}
            
            // Download the first blob
            console.log("Downloading video content...");
            const videoBuffer = await page.evaluate(async (blobUrl) => {{
                try {{
                    const response = await fetch(blobUrl);
                    if (!response.ok) {{
                        throw new Error(`Fetch failed: ${{response.status}}`);
                    }}
                    const blob = await response.blob();
                    const arrayBuffer = await blob.arrayBuffer();
                    return Array.from(new Uint8Array(arrayBuffer));
                }} catch (error) {{
                    console.error("Download error:", error);
                    throw error;
                }}
            }}, videoContent.blobUrls[0]);
            
            console.log(`Downloaded ${{videoBuffer.length}} bytes`);
            return Buffer.from(videoBuffer);
            
        }} catch (error) {{
            console.error("Browser automation error:", error);
            throw error;
        }} finally {{
            await page.close();
        }}
        '''
        
        try:
            async with aiohttp.ClientSession() as session:
                # Prepare headers
                headers = {'Content-Type': 'application/json'}
                
                # Prepare payload
                payload = {
                    "code": script,
                    "context": {"text": escaped_text}
                }
                
                # Build endpoint URL with token parameter
                endpoint = f"{self.base_url}/function"
                if self.api_key:
                    endpoint = f"{endpoint}?token={self.api_key}"
                
                logger.info(f"Making request to: {endpoint}")
                
                async with session.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=config.API_TIMEOUT)
                ) as response:
                    
                    response_text = await response.text()
                    logger.info(f"Browserless response status: {response.status}")
                    
                    if response.status == 200:
                        content = await response.read() if response.content_type == 'application/octet-stream' else response_text.encode()
                        return content
                    elif response.status == 403:
                        logger.error(f"Browserless 403 Forbidden. Response: {response_text[:500]}")
                        logger.error("This usually means:")
                        logger.error("1. Missing or invalid token parameter")
                        logger.error("2. Exceeded rate limits") 
                        logger.error("3. Incorrect endpoint URL")
                        return None
                    elif response.status == 401:
                        logger.error("Browserless 401 Unauthorized - Check your token")
                        return None
                    else:
                        logger.error(f"Browserless API error: {response.status} - {response_text[:500]}")
                        return None
                        
        except asyncio.TimeoutError:
            logger.error("Browserless request timed out")
            return None
        except Exception as e:
            logger.error(f"Browser automation failed: {e}")
            return None

# Selenium Grid fallback
class SeleniumGridAutomation:
    def __init__(self):
        self.grid_url = config.SELENIUM_GRID_URL or "http://localhost:4444/wd/hub"
    
    async def translate_text_to_sign(self, text: str) -> Optional[bytes]:
        """Use Selenium Grid for browser automation"""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.webdriver.common.keys import Keys
            
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            
            driver = webdriver.Remote(
                command_executor=self.grid_url,
                options=options
            )
            
            try:
                driver.get("https://sign.mt/")
                time.sleep(3)
                
                # Find text input
                text_input = driver.find_element(By.CSS_SELECTOR, "textarea, input[type='text']")
                text_input.clear()
                text_input.send_keys(text)
                text_input.send_keys(Keys.RETURN)
                
                # Wait for translation
                time.sleep(10)
                
                # Find blob URLs
                blob_script = """
                var elements = document.querySelectorAll('*');
                var blobs = [];
                for (var i = 0; i < elements.length; i++) {
                    var el = elements[i];
                    if (el.src && el.src.startsWith('blob:')) {
                        blobs.push(el.src);
                    }
                }
                return blobs;
                """
                
                blob_urls = driver.execute_script(blob_script)
                
                if not blob_urls:
                    return None
                
                # Download blob content
                download_script = f"""
                return new Promise((resolve) => {{
                    fetch('{blob_urls[0]}')
                        .then(r => r.blob())
                        .then(blob => {{
                            var reader = new FileReader();
                            reader.onload = () => resolve(reader.result);
                            reader.readAsDataURL(blob);
                        }})
                        .catch(() => resolve(null));
                }});
                """
                
                base64_data = driver.execute_async_script(download_script)
                
                if base64_data and base64_data.startswith('data:'):
                    header, data = base64_data.split(',', 1)
                    return base64.b64decode(data)
                
                return None
                
            finally:
                driver.quit()
                
        except Exception as e:
            logger.error(f"Selenium Grid automation failed: {e}")
            return None

# Initialize automation backend
automation_backend = None
playwright_instance = None

if config.USE_PLAYWRIGHT:
    playwright_instance = PlaywrightAutomation()
    automation_backend = playwright_instance
    logger.info("Using Playwright for browser automation")
elif config.BROWSERLESS_URL:
    automation_backend = BrowserlessAutomation()
    logger.info("Using Browserless for browser automation")
elif config.SELENIUM_GRID_URL:
    automation_backend = SeleniumGridAutomation()
    logger.info("Using Selenium Grid for browser automation")
else:
    logger.warning("No browser automation backend configured")

# Global instances
storage_manager = StorageManager()

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FastAPI Text-to-Sign API (Fixed Version)")
    
    # Initialize Playwright if using it
    if playwright_instance:
        await playwright_instance.initialize()
    
    yield
    
    # Cleanup
    if playwright_instance:
        await playwright_instance.cleanup()
    
    if hasattr(storage_manager, 'temp_dir'):
        shutil.rmtree(storage_manager.temp_dir, ignore_errors=True)

# FastAPI app instance
app = FastAPI(
    title="Text-to-Sign Translation API",
    description="Convert text to sign language videos - Fixed Version",
    version="3.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def process_translation_async(translation_id: str, text: str):
    """Process translation in background"""
    try:
        logger.info(f"Processing translation {translation_id}: {text}")
        
        if not automation_backend:
            raise Exception("No browser automation backend available")
        
        # Perform the translation
        video_content = await automation_backend.translate_text_to_sign(text)
        
        if not video_content:
            raise Exception("Failed to generate video content")
        
        # Validate video content
        if len(video_content) < 1000:
            raise Exception(f"Video content too small ({len(video_content)} bytes), likely corrupted")
        
        # Save the file
        safe_text = "".join(c for c in text if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_text = safe_text.replace(' ', '_')[:50]  # Limit filename length
        filename = f"{safe_text}_sign_{uuid.uuid4().hex[:8]}.webm"
        
        file_path = await storage_manager.save_file(video_content, filename)
        
        await StateManager.set_translation(translation_id, {
            'status': 'completed',
            'text': text,
            'file_path': file_path,
            'error': None,
            'created_at': time.time()
        })
        
        logger.info(f"Translation {translation_id} completed successfully ({len(video_content)} bytes)")
        
    except Exception as e:
        error_msg = str(e)
        await StateManager.set_translation(translation_id, {
            'status': 'failed',
            'text': text,
            'file_path': None,
            'error': error_msg,
            'created_at': time.time()
        })
        logger.error(f"Translation {translation_id} failed: {error_msg}")

# FastAPI Routes
@app.post("/api/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest, background_tasks: BackgroundTasks):
    """Start a new text-to-sign translation"""
    if not automation_backend:
        raise HTTPException(
            status_code=503, 
            detail="Browser automation service not available. Please configure BROWSERLESS_URL with API key, USE_PLAYWRIGHT, or SELENIUM_GRID_URL."
        )
    
    text = request.text
    translation_id = str(uuid.uuid4())
    
    await StateManager.set_translation(translation_id, {
        'status': 'processing',
        'text': text,
        'file_path': None,
        'error': None,
        'created_at': time.time()
    })

    background_tasks.add_task(process_translation_async, translation_id, text)

    return TranslationResponse(
        translation_id=translation_id,
        status='processing',
        message='Translation started'
    )

@app.get("/api/translation/{translation_id}", response_model=TranslationStatus)
async def get_translation_status(translation_id: str):
    """Get the status of a translation"""
    translation = await StateManager.get_translation(translation_id)
    
    if not translation:
        raise HTTPException(status_code=404, detail='Translation not found')

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
    translation = await StateManager.get_translation(translation_id)
    
    if not translation:
        raise HTTPException(status_code=404, detail='Translation not found')

    if translation['status'] != 'completed' or not translation['file_path']:
        raise HTTPException(status_code=400, detail='Translation not ready')

    file_content = await storage_manager.get_file(translation['file_path'])
    
    if not file_content:
        raise HTTPException(status_code=404, detail='File not found')

    def generate():
        yield file_content

    return StreamingResponse(
        generate(),
        media_type='video/webm',
        headers={"Content-Disposition": f"attachment; filename=sign_translation_{translation_id}.webm"}
    )

@app.get("/api/status", response_model=APIStatus)
async def api_status():
    """Get API status information"""
    all_translations = await StateManager.get_all_translations()
    
    return APIStatus(
        status='online',
        active_translations=len([t for t in all_translations.values() if t.get('status') == 'processing']),
        total_translations=len(all_translations),
        storage_backend=storage_backend,
        state_backend="redis" if redis_client else "memory"
    )

@app.get("/debug/browserless")
async def debug_browserless():
    """Debug Browserless configuration"""
    return {
        "browserless_url": config.BROWSERLESS_URL,
        "has_api_key": bool(config.BROWSERLESS_API_KEY),
        "api_key_preview": f"{config.BROWSERLESS_API_KEY[:10]}..." if config.BROWSERLESS_API_KEY else None,
        "use_playwright": config.USE_PLAYWRIGHT,
        "selenium_grid_url": config.SELENIUM_GRID_URL,
        "automation_backend": type(automation_backend).__name__ if automation_backend else None,
        "backend_available": automation_backend is not None
    }

@app.get("/debug/test-browserless")
async def test_browserless():
    """Test Browserless connection"""
    if not isinstance(automation_backend, BrowserlessAutomation):
        return {"error": "Browserless not configured"}
    
    try:
        async with aiohttp.ClientSession() as session:
            headers = {'Content-Type': 'application/json'}
            
            # Simple test script
            payload = {
                "code": "return 'Hello from Browserless!';"
            }
            
            # Build endpoint URL with token parameter
            endpoint = f"{config.BROWSERLESS_URL}/function"
            if config.BROWSERLESS_API_KEY:
                endpoint = f"{endpoint}?token={config.BROWSERLESS_API_KEY}"
            
            async with session.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                result = await response.text()
                return {
                    "status": response.status,
                    "response": result,
                    "success": response.status == 200,
                    "endpoint": endpoint.replace(config.BROWSERLESS_API_KEY or "", "***TOKEN***")
                }
    except Exception as e:
        return {"error": str(e), "success": False}

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status='healthy', 
        timestamp=time.time(),
        version="3.1.0"
    )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Text-to-Sign Translation API - Fixed Version",
        "version": "3.1.0",
        "docs": "/docs",
        "health": "/health",
        "debug": "/debug/browserless",
        "features": {
            "storage": storage_backend,
            "state_management": "redis" if redis_client else "memory",
            "browser_automation": type(automation_backend).__name__ if automation_backend else "none",
            "backends_available": {
                "browserless": bool(config.BROWSERLESS_URL),
                "playwright": config.USE_PLAYWRIGHT,
                "selenium_grid": bool(config.SELENIUM_GRID_URL)
            }
        }
    }

if __name__ == '__main__':
    import uvicorn
    logger.info("Starting FastAPI Text-to-Sign API (Fixed Version)...")
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        workers=int(os.getenv("WORKERS", "1"))
    )