import os
import sys
import time
import torch
import soundfile as sf
import torchaudio
import tempfile
import shutil
import gc
import logging
from contextlib import contextmanager
from fish_speech_api.utils.cudagraphs import mark_compiled_step_starts
from huggingface_hub import snapshot_download

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("fish_speech_infer")

FISH_SPEECH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fish_speech")
sys.path.append(FISH_SPEECH_DIR)

from fish_speech.tokenizer import FishTokenizer
from fish_speech.models.text2semantic.inference import main as text2semantic_main
from fish_speech.models.vqgan import inference as vqgan_inference


@contextmanager
def temporary_argv(new_argv):
    original_argv = sys.argv
    sys.argv = new_argv
    try:
        yield
    finally:
        sys.argv = original_argv


def download_models(cache_dir="./models/fish-speech-1.5", local_only=True):
    logger.info(f"Downloading model into: {cache_dir}")
    try:
        repo_dir = snapshot_download(
            repo_id="fishaudio/fish-speech-1.5",
            cache_dir=cache_dir,
            local_files_only=local_only
        )
        logger.info("Model download completed successfully.")
        return repo_dir
    except Exception as e:
        logger.error(f"Model download failed: {e}")
        return None


def encode_reference_audio(reference_audio_path, temp_dir, checkpoint_path, device="cuda"):
    logger.info("Encoding reference audio...")

    # Normalize waveform
    waveform, sample_rate = torchaudio.load(reference_audio_path)
    waveform = waveform.to(device)

    if waveform.abs().max() > 1.0:
        waveform = waveform / waveform.abs().max()

    audio_int16 = (waveform * 32767).to(torch.int16).cpu().numpy()
    normalized_path = os.path.join(temp_dir, "normalized_ref.wav")
    sf.write(normalized_path, audio_int16.T, sample_rate)
    logger.debug(f"Normalized audio saved at: {normalized_path}")

    # Ensure output base path has NO extension, but .npy will be added by inference.py
    reference_tokens_base = os.path.join(temp_dir, "reference_tokens")
    reference_tokens_path = reference_tokens_base + ".npy"

    # Run VQGAN inference to encode audio into tokens
    sys.argv = [
        "inference.py",
        "--input-path", normalized_path,
        "--output-path", reference_tokens_base + ".wav",  # Add .wav to satisfy soundfile
        "--checkpoint-path", checkpoint_path,
        "--device", device
    ]

    try:
        vqgan_inference.main()
    except SystemExit:
        logger.debug("vqgan_inference.main() exited with SystemExit (normal for CLI entrypoints).")

    # Ensure .npy was created
    if not os.path.exists(reference_tokens_path):
        raise RuntimeError(f"Reference tokens were not generated at: {reference_tokens_path}")

    logger.debug(f"Reference tokens saved at: {reference_tokens_path}")
    return reference_tokens_path


def generate_semantic_tokens(
    text,
    checkpoint_path,
    temp_dir,
    prompt_tokens=None,
    prompt_text=None,
    device="cuda",
    compile_model=False,
    num_samples=1,
    top_p=0.7,
    repetition_penalty=1.2,
    temperature=0.7,
    chunk_length=200,
    max_new_tokens=1024,
    seed=None,
):
    logger.info("Generating semantic tokens...")
    semantic_tokens_path = os.path.join(temp_dir, "codes_0.npy")

    args = [
        "--text", text,
        "--checkpoint-path", checkpoint_path,
        "--device", device,
        "--num-samples", str(num_samples),
        "--output-dir", temp_dir,
        "--top-p", str(top_p),
        "--repetition-penalty", str(repetition_penalty),
        "--temperature", str(temperature),
        "--chunk-length", str(chunk_length),
        "--max-new-tokens", str(max_new_tokens),
    ]

    if prompt_tokens:
        args.extend(["--prompt-tokens", prompt_tokens])
        args.extend(["--prompt-text", prompt_text or text])

    if compile_model:
        args.append("--compile")

    if seed is not None:
        args.extend(["--seed", str(seed)])

    logger.debug(f"Inference args: {args}")

    with temporary_argv(["inference.py"] + args):
        try:
            if compile_model:
                with mark_compiled_step_starts(torch):
                    text2semantic_main()
            else:
                text2semantic_main()
        except SystemExit:
            logger.debug("text2semantic_main() exited with SystemExit (normal for CLI entrypoints).")

    logger.debug(f"Semantic tokens saved at: {semantic_tokens_path}")
    return semantic_tokens_path


def generate_speech_from_tokens(tokens_path, checkpoint_path, output_path, device="cuda"):
    logger.info("Generating waveform from semantic tokens...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sys.argv = [
        "inference.py",
        "--input-path", tokens_path,
        "--output-path", output_path,
        "--checkpoint-path", checkpoint_path,
        "--device", device
    ]

    try:
        vqgan_inference.main()
    except SystemExit:
        logger.debug("vqgan_inference.main() exited with SystemExit (normal for CLI entrypoints).")

    logger.info(f"Waveform saved to: {output_path}")
    return output_path


def clear_gpu_memory():
    logger.debug("Clearing GPU memory...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'ipc_collect'):
            torch.cuda.ipc_collect()


def text_to_speech(
    text,
    output_path="output.wav",
    reference_audio_path=None,
    checkpoint_dir="./models/fish-speech-1.5",
    device="cuda",
    compile_model=False,
    top_p=0.7,
    repetition_penalty=1.2,
    temperature=0.7,
    chunk_length=200,
    max_new_tokens=1024,
    seed=None,
    prompt_text=None,
):
    logger.info("Starting text-to-speech pipeline...")
    start = time.time()

    llama_ckpt = checkpoint_dir
    decoder_ckpt = os.path.join(checkpoint_dir, "firefly-gan-vq-fsq-8x1024-21hz-generator.pth")

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            reference_tokens_path = None
            if reference_audio_path:
                reference_tokens_path = encode_reference_audio(
                    reference_audio_path=reference_audio_path,
                    temp_dir=temp_dir,
                    checkpoint_path=decoder_ckpt,
                    device=device
                )

            semantic_tokens_path = generate_semantic_tokens(
                text=text,
                checkpoint_path=llama_ckpt,
                temp_dir=temp_dir,
                prompt_tokens=reference_tokens_path,
                prompt_text=prompt_text,
                device=device,
                compile_model=compile_model,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                chunk_length=chunk_length,
                max_new_tokens=max_new_tokens,
                seed=seed,
            )

            generate_speech_from_tokens(
                tokens_path=semantic_tokens_path,
                checkpoint_path=decoder_ckpt,
                output_path=output_path,
                device=device
            )

            elapsed = time.time() - start
            logger.info(f"Inference completed in {elapsed:.2f} seconds")
            return output_path

        except Exception as e:
            logger.exception(f"TTS pipeline failed: {e}")
            raise

        finally:
            clear_gpu_memory()
