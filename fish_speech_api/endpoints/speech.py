import logging
from pathlib import Path
from typing import Optional, Tuple
import base64
import os
import requests

from pydantic import BaseModel, ValidationError
from fastapi import APIRouter, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import FileResponse
from fastapi.exceptions import HTTPException

from fish_speech_api.services.tts_service import generate_tts
from fish_speech_api.services.voice_store import MAX_AUDIO_BYTES, VoiceStore

BUILTIN_VOICE_DIR = Path("./examples")
VOICE_DIR = Path(os.environ.get("FISH_SPEECH_VOICE_DIR", "./voices"))
VOICE_STORE = VoiceStore(VOICE_DIR, BUILTIN_VOICE_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("fish_speech_api.routes.speech")

router = APIRouter()


class TTSRequest(BaseModel):
    model: str
    input: str
    voice: Optional[str] = 'english-nice'
    top_p: float = 0.7
    repetition_penalty: float = 1.5
    temperature: float = 0.7
    chunk_length: int = 150
    max_new_tokens: int = 2048
    seed: Optional[int] = None
    reference_audio_base64: Optional[str] = None
    reference_text: Optional[str] = None


async def get_voice_sample(voice: str) -> Tuple[bytes, str]:
    """Get a voice sample and its text label."""
    try:
        return VOICE_STORE.get(voice)
    except ValueError as error:
        raise HTTPException(400, error.args[0])
    except FileNotFoundError as error:
        raise HTTPException(400, error.args[0])


async def process_tts_request(
    model: str,
    input_text: str,
    voice: Optional[str],
    top_p: float,
    repetition_penalty: float,
    temperature: float,
    chunk_length: int,
    max_new_tokens: int,
    seed: Optional[int],
    reference_audio_bytes: Optional[bytes],
    prompt_text: Optional[str],
) -> FileResponse:
    # Validate input length
    if len(input_text) > 4096:
        raise HTTPException(400, "Input too long (max 4096 chars)")

    audio_bytes = None

    # Priority 1: User-provided reference
    if reference_audio_bytes:
        if len(reference_audio_bytes) > 25 * 1024 * 1024:
            raise HTTPException(400, "Audio data too large (max 25MB)")
        if reference_audio_bytes[:4] != b'RIFF':
            raise HTTPException(400, "Invalid audio format (must be WAV)")
        audio_bytes = reference_audio_bytes

    # Priority 2: Voice sample from repository
    elif voice:
        audio_bytes, prompt_text = await get_voice_sample(voice)

    # Generate TTS
    output_path = generate_tts(
        text=input_text,
        model_name=model,
        voice_sample=audio_bytes,
        prompt_text=prompt_text,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        chunk_length=chunk_length,
        max_new_tokens=max_new_tokens,
        seed=seed
    )

    return FileResponse(
        path=output_path,
        media_type="audio/wav",
        filename=Path(output_path).name
    )


@router.post("/model", status_code=201)
async def create_voice_model(
    name: str = Form(...),
    reference_audio: UploadFile = File(...),
    reference_text: str = Form(...),
):
    if not reference_audio.filename or not reference_audio.filename.lower().endswith(".wav"):
        raise HTTPException(400, "Invalid audio format (must be WAV)")

    audio_bytes = await reference_audio.read(MAX_AUDIO_BYTES + 1)
    try:
        voice_name = VOICE_STORE.create(name, audio_bytes, reference_text)
    except FileExistsError as error:
        raise HTTPException(409, error.args[0])
    except ValueError as error:
        raise HTTPException(400, error.args[0])

    return {"id": voice_name, "voice": voice_name}


@router.post("/v1/tts")
@router.post("/audio/speech")
async def speech_endpoint(request: Request):
    content_type = request.headers.get('Content-Type', '')
    logger.info(f"Incoming request with Content-Type: {content_type}")

    try:
        if 'multipart/form-data' in content_type:
            form_data = await request.form()

            # Log form data
            log_data = {k: v.filename if isinstance(v, UploadFile) else v for k, v in form_data.items()}
            logger.info(f"Multipart form data: {log_data}")

            # Extract and validate form fields
            model = form_data.get('model')
            input_text = form_data.get('input')
            if not model or not input_text:
                raise HTTPException(400, "Missing required fields: model and input")

            # Convert numeric parameters with defaults
            top_p = float(form_data.get('top_p', 0.7))
            repetition_penalty = float(form_data.get('repetition_penalty', 1.5))
            temperature = float(form_data.get('temperature', 0.7))
            chunk_length = int(form_data.get('chunk_length', 150))
            max_new_tokens = int(form_data.get('max_new_tokens', 2048))
            seed = int(form_data['seed']) if 'seed' in form_data else None
            voice = form_data.get('voice')

            # Handle reference audio file
            reference_audio = form_data.get('reference_audio')
            reference_text = form_data.get('reference_text', None)
            reference_audio_bytes = None
            if reference_audio:
                if not reference_audio.filename.lower().endswith('.wav'):
                    raise HTTPException(400, "Invalid audio format (must be WAV)")
                reference_audio_bytes = await reference_audio.read(MAX_AUDIO_BYTES + 1)

            return await process_tts_request(
                model=model,
                input_text=input_text,
                voice=voice,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                chunk_length=chunk_length,
                max_new_tokens=max_new_tokens,
                seed=seed,
                reference_audio_bytes=reference_audio_bytes,
                prompt_text=reference_text
            )

        elif 'application/json' in content_type:
            json_data = await request.json()
            logger.info(f"JSON request data: {json_data}")

            # Explicit required parameter checks
            if 'model' not in json_data or 'input' not in json_data:
                raise HTTPException(400, "Missing required parameters: model and input")

            request_data = TTSRequest(**json_data)

            # Handle base64 audio with validation
            reference_audio_bytes = None
            if request_data.reference_audio_base64:
                try:
                    reference_audio_bytes = base64.b64decode(request_data.reference_audio_base64)
                except Exception:
                    raise HTTPException(400, "Invalid base64 data")

                # Size check (25MB limit)
                if len(reference_audio_bytes) > 25 * 1024 * 1024:
                    raise HTTPException(400, "Audio data too large (max 25MB)")

                # Format check via magic number
                if reference_audio_bytes[:4] != b'RIFF':
                    raise HTTPException(400, "Invalid audio format (must be WAV)")

            return await process_tts_request(
                model=request_data.model,
                input_text=request_data.input,
                voice=request_data.voice,
                top_p=request_data.top_p,
                repetition_penalty=request_data.repetition_penalty,
                temperature=request_data.temperature,
                chunk_length=request_data.chunk_length,
                max_new_tokens=request_data.max_new_tokens,
                seed=request_data.seed,
                reference_audio_bytes=reference_audio_bytes,
                prompt_text=request_data.reference_text,
            )
        else:
            logger.warning(f"Unsupported media type: {content_type}")
            raise HTTPException(415, "Unsupported media type")

    except HTTPException as he:
        logger.warning(f"HTTPException: {he.detail}")
        raise he
    except ValidationError as ve:
        logger.warning(f"Validation error: {str(ve)}")
        raise HTTPException(400, str(ve))
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        raise HTTPException(500, "Internal server error")


@router.get("/ping")
async def ping():
    return {"status": "ok"}
