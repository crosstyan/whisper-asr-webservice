#!/usr/bin/env python3

import click
import httpx
import ffmpeg
import tempfile
import os
from pathlib import Path
from typing import Final, Optional, Tuple
import time
from datetime import timedelta

# from whisper import tokenizer

LANGUAGES: Final[dict[str, str]] = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}

LANGUAGE_CODES: Final[list[str]] = sorted(LANGUAGES.keys())

# https://ahmetoner.com/whisper-asr-webservice/endpoints/
SUPPORTED_OUTPUT_FORMATS: Final[tuple[str, ...]] = ('srt', 'vtt', 'text', 'json', 'tsv')

# Common video and audio formats
MEDIA_FORMATS: Final[tuple[str, ...]] = (
    # Video formats
    "*.mp4",
    "*.mkv",
    "*.avi",
    "*.mov",
    "*.wmv",
    "*.flv",
    "*.webm",
    "*.m4v",
    "*.mpg",
    "*.mpeg",
    "*.3gp",
    # Audio formats
    "*.mp3",
    "*.wav",
    "*.m4a",
    "*.aac",
    "*.ogg",
    "*.flac",
    "*.wma",
    "*.opus",
    "*.alac",
)

# Audio-only formats (no need to extract audio from these)
AUDIO_FORMATS: Final[set[str]] = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac', '.wma', '.opus', '.alac'}


def output_file_name(input_file: Path, output_suffix: str, language: Optional[str] = None) -> Path:
    if language is None:
        return input_file.with_suffix(f'.{output_suffix}')
    else:
        return input_file.with_suffix(f'.{language}.{output_suffix}')


def get_media_info(file_path: Path) -> dict:
    """Get media information using ffmpeg."""
    try:
        probe = ffmpeg.probe(str(file_path))
        return probe
    except ffmpeg.Error as e:
        click.echo(f"Error probing media file: {e.stderr.decode()}", err=True)
        return {}


def extract_audio(input_file: Path, temp_dir: Optional[Path] = None) -> Tuple[Optional[Path], Optional[dict]]:
    """Extract audio from video file using ffmpeg.

    Returns:
        Tuple containing:
        - Path to extracted audio file (or None if extraction failed)
        - Dictionary with audio information (duration, codec, etc.)
    """
    try:
        # Get media info
        probe = get_media_info(input_file)
        if not probe:
            return None, None

        # Find audio stream
        audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
        if not audio_stream:
            click.echo(f"No audio stream found in {input_file}")
            return None, None

        # Get audio codec
        audio_codec = audio_stream.get('codec_name', 'unknown')

        # Create temp file with appropriate extension
        temp_file = tempfile.NamedTemporaryFile(suffix=f".{audio_codec}", dir=temp_dir, delete=False)
        temp_file.close()
        temp_path = Path(temp_file.name)

        # Extract audio
        click.echo(f"Extracting audio from {input_file}...")

        # Get duration if available
        duration = float(probe.get('format', {}).get('duration', 0))
        duration_str = str(timedelta(seconds=int(duration))) if duration else "unknown"

        # Display audio information
        click.echo(f"  Duration: {duration_str}")
        click.echo(f"  Audio codec: {audio_codec}")
        click.echo(f"  Extracting to temporary file...")

        # Run ffmpeg extraction
        (
            ffmpeg.input(str(input_file))
            .output(str(temp_path), vn=None, acodec='copy')
            .run(quiet=True, overwrite_output=True)
        )

        # Get file size
        file_size = temp_path.stat().st_size
        click.echo(f"  Extracted audio size: {file_size / (1024*1024):.2f} MB")

        return temp_path, {'duration': duration, 'codec': audio_codec, 'size': file_size}

    except ffmpeg.Error as e:
        click.echo(f"Error extracting audio: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}", err=True)
        return None, None
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        return None, None


def process_single_file(
    input_file: Path,
    url: str,
    output_format: str,
    language: Optional[str] = None,
    output_file: Optional[Path] = None,
    extract_audio_only: bool = False,
) -> tuple[bool, float]:
    """Process a single video file and return success status and processing time."""
    # If output file is not specified, use input filename with new extension
    if output_file is None:
        output_file = output_file_name(input_file, output_format, language)

    # Skip if subtitle file already exists
    if output_file.exists():
        click.echo(f"Skipping {input_file} - subtitle file already exists at {output_file}")
        return False, 0.0

    start_time = time.time()
    temp_audio_file = None

    try:
        # Determine if we should extract audio
        file_to_send = input_file
        if extract_audio_only and input_file.suffix.lower() not in AUDIO_FORMATS:
            # Extract audio to temp file
            temp_audio_file, audio_info = extract_audio(input_file)
            if temp_audio_file:
                file_to_send = temp_audio_file
            else:
                click.echo(f"Warning: Audio extraction failed, using original file")

        # Prepare the files for upload
        click.echo(f"Transcribing {input_file}...")
        click.echo(f"Output will be saved to {output_file}")

        with open(file_to_send, 'rb') as f:
            files = {'audio_file': f}

            # Prepare query parameters
            if language is None:
                params = {'output': output_format}
            else:
                params = {'output': output_format, 'language': language}

            # Make the request
            with httpx.Client(timeout=None) as client:
                response = client.post(url, params=params, files=files)
                response.raise_for_status()

                # Save the response content to the output file
                output_file.write_bytes(response.content)
                elapsed_time = time.time() - start_time
                click.echo(
                    f"✓ Transcription completed in {timedelta(seconds=int(elapsed_time))} and saved to {output_file}"
                )
                return True, elapsed_time

    except httpx.HTTPError as e:
        click.echo(f"Error: Failed to make request: {e}", err=True)
        return False, 0.0
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        return False, 0.0
    finally:
        # Clean up temp file if it exists
        if temp_audio_file and temp_audio_file.exists():
            try:
                os.unlink(temp_audio_file)
            except Exception as e:
                click.echo(f"Warning: Failed to remove temporary file {temp_audio_file}: {e}", err=True)


@click.command()
@click.argument('input_path', type=click.Path(path_type=Path))
@click.option('--url', '-u', default='http://127.0.0.1:9000/asr', help='URL of the ASR service')
@click.option(
    '--language', '-l', default='en', type=click.Choice(LANGUAGE_CODES), help='Language code for transcription'
)
@click.option(
    '--auto-language',
    '-a',
    is_flag=True,
    help='Automatically detect language. Language code will be ignored if this flag is set.',
)
@click.option('--output-format', '-f', default='srt', type=click.Choice(SUPPORTED_OUTPUT_FORMATS), help='Output format')
@click.option(
    '--output-file',
    '-o',
    type=click.Path(path_type=Path),
    help='Output file path (default: input filename with new extension). Only used for single file input.',
)
@click.option(
    '--dry-run',
    '-d',
    is_flag=True,
    help='Show what would be processed without actually processing any files.',
)
@click.option(
    '--recursive',
    '-r',
    is_flag=True,
    help='Recursively scan subdirectories for media files.',
)
@click.option(
    '--extract-audio',
    '-e',
    is_flag=True,
    help='Extract audio from video files before sending to ASR service. Improves performance for large video files.',
)
def transcribe(
    input_path: Path,
    url: str,
    language: str,
    auto_language: bool,
    output_format: str,
    output_file: Path | None,
    dry_run: bool,
    recursive: bool,
    extract_audio: bool,
):
    """Transcribe audio/video file(s) using Whisper ASR Web Service.

    INPUT_PATH: Path to media file, directory, or glob pattern (e.g., "videos/*.mp4", "**/*.mp3")
    """
    # Handle glob patterns in input path
    if '*' in str(input_path):
        base_dir = Path(str(input_path).split('*')[0]).parent
        pattern = input_path.name
        if recursive:
            media_files = list(base_dir.glob('**/' + pattern))
        else:
            media_files = list(base_dir.glob(pattern))

        if not media_files:
            click.echo(f"No files found matching pattern: {input_path}")
            return

        # Sort files by name for consistent ordering
        media_files.sort()

    elif input_path.is_file():
        media_files = [input_path]
    else:
        # Scan directory for media files
        media_files = []
        click.echo(f"Scanning directory {input_path} for media files{'recursively' if recursive else ''}...")

        for pattern in MEDIA_FORMATS:
            if recursive:
                media_files.extend(input_path.glob(f"**/{pattern}"))
            else:
                media_files.extend(input_path.glob(pattern))

        if not media_files:
            click.echo("No media files found in the directory.")
            return

        # Sort files by name for consistent ordering
        media_files.sort()

    # Preview files to be processed
    click.echo(f"\nFound {len(media_files)} media files:")
    to_process: list[Path] = []
    to_skip: list[Path] = []

    actual_language = None if auto_language else language

    for media_file in media_files:
        output_file = output_file_name(media_file, output_format, actual_language)
        if output_file.exists():
            to_skip.append(media_file)
        else:
            to_process.append(media_file)

    if to_process:
        click.echo("\nFiles to be processed:")
        for idx, file in enumerate(to_process, 1):
            output_name = output_file_name(file, output_format, actual_language)
            click.echo(f"{idx}. {file.name} -> {output_name.name}")

    if to_skip:
        click.echo("\nFiles to be skipped (subtitles already exist):")
        for idx, file in enumerate(to_skip, 1):
            click.echo(f"{idx}. {file.name}")

    if not to_process:
        click.echo("\nNo files to process - all files already have subtitles.")
        return

    if dry_run:
        click.echo(f"\nDry run complete. Would process {len(to_process)} files.")
        return

    # Process files
    processed_files = 0
    total_time = 0.0

    total_to_process = len(to_process)
    click.echo(f"\nProcessing {total_to_process} files:")

    for idx, media_file in enumerate(to_process, 1):
        click.echo(f"[{idx}/{total_to_process}] Processing {media_file.name}...")
        success, elapsed_time = process_single_file(
            media_file, url, output_format, actual_language, None, extract_audio
        )
        if success:
            processed_files += 1
            total_time += elapsed_time
            click.echo(f"✓ Completed in {timedelta(seconds=int(elapsed_time))}")
        else:
            click.echo("✗ Failed")

    # Show summary
    avg_time = total_time / processed_files if processed_files > 0 else 0
    click.echo(f"\nBatch processing complete:")
    click.echo(f"Total files found: {len(media_files)}")
    click.echo(f"Files skipped: {len(to_skip)}")
    click.echo(f"Files processed: {processed_files} of {len(to_process)}")
    click.echo(f"Total processing time: {timedelta(seconds=int(total_time))}")
    if processed_files > 0:
        click.echo(f"Average processing time per file: {timedelta(seconds=int(avg_time))}")


if __name__ == '__main__':
    transcribe()  # pylint: disable=no-value-for-parameter
