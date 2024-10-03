import os
import whisper_timestamped
import subprocess
from typing import Dict, Any
from cog import BasePredictor, Input, Path
import uuid

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = whisper_timestamped.load_model("openai/whisper-large-v3", device="cuda")

    def predict(
            self,
            audio_file: Path = Input(description="Audio file to transcribe"),
            language: str = Input(description="Language code (e.g., 'en') or 'auto' for auto-detect", default="auto"),
            task: str = Input(description="Task to perform", choices=["transcribe", "translate"], default="transcribe"),
            vad: bool = Input(description="Use Voice Activity Detection", default=False),
            detect_disfluencies: bool = Input(description="Detect speech disfluencies", default=False),
            compute_word_confidence: bool = Input(description="Compute word confidence scores", default=True),
            temperature: float = Input(description="Temperature for sampling", default=0.0),
            best_of: int = Input(description="Number of candidates when sampling with non-zero temperature", default=None),
            beam_size: int = Input(description="Number of beams in beam search, only applicable when temperature is zero", default=None),
            patience: float = Input(description="Optional patience value to use in beam decoding", default=None),
            length_penalty: float = Input(description="Optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144", default=None),
            suppress_tokens: str = Input(description="Comma-separated list of token ids to suppress during sampling", default="-1"),
            initial_prompt: str = Input(description="Optional text to provide as a prompt for the first window", default=None),
            condition_on_previous_text: bool = Input(description="Whether to condition on previous text", default=True),
            no_speech_threshold: float = Input(description="Threshold for no speech probability", default=0.6),
            compression_ratio_threshold: float = Input(description="Threshold for compression ratio", default=2.4),
            logprob_threshold: float = Input(description="Threshold for average log probability", default=-1.0),
            verbose: bool = Input(description="Whether to display the text being decoded", default=False)
    ) -> Dict[str, Any]:
        """Run a single prediction on the model"""
        try:
            # Prepare audio file
            audio = self.prepare_audio(audio_file)

            # Prepare transcription options
            options = {
                "language": None if language == "auto" else language,
                "task": task,
                "vad": vad,
                "detect_disfluencies": detect_disfluencies,
                "compute_word_confidence": compute_word_confidence,
                "temperature": temperature,
                "best_of": best_of,
                "beam_size": beam_size,
                "patience": patience,
                "length_penalty": length_penalty,
                "suppress_tokens": suppress_tokens,
                "initial_prompt": initial_prompt,
                "condition_on_previous_text": condition_on_previous_text,
                "no_speech_threshold": no_speech_threshold,
                "compression_ratio_threshold": compression_ratio_threshold,
                "logprob_threshold": logprob_threshold,
                "verbose": verbose
            }

            # Run transcription
            result = whisper_timestamped.transcribe(self.model, audio, **options)

            return result

        except Exception as e:
            raise RuntimeError(f"Error during transcription: {str(e)}")

        finally:
            # Cleanup temporary files
            if 'audio' in locals() and os.path.exists(audio):
                os.remove(audio)


    def prepare_audio(self, audio_file: Path) -> str:
        """Prepare audio file"""
        try:
            # Generate a unique filename using UUID
            unique_filename = f"{uuid.uuid4()}.wav"
            output_path = os.path.join("/tmp", unique_filename)

            # Convert audio to required format
            subprocess.run([
                "ffmpeg",
                "-i", str(audio_file),
                "-ar", "16000",
                "-ac", "1",
                "-c:a", "pcm_s16le",
                output_path
            ], check=True, capture_output=True, text=True)

            return output_path

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error preparing audio file: {e.stderr}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error preparing audio file: {str(e)}")