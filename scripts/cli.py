#!/usr/bin/env python3

import click
import httpx
from pathlib import Path
from typing import Final
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

VIDEO_FORMATS: Final[tuple[str, ...]] = (
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
)


def process_single_file(
    input_file: Path,
    url: str,
    language: str,
    auto_language: bool,
    output_format: str,
    output_file: Path | None,
) -> tuple[bool, float]:
    """Process a single video file and return success status and processing time."""

    # If output file is not specified, use input filename with new extension
    if output_file is None:
        if auto_language:
            output_file = input_file.with_suffix(f'.{output_format}')
        else:
            output_file = input_file.with_suffix(f'.{language}.{output_format}')

    # Skip if subtitle file already exists
    if output_file.exists():
        click.echo(f"Skipping {input_file} - subtitle file already exists at {output_file}")
        return False, 0.0

    # Prepare the files for upload
    files = {'audio_file': input_file.open('rb')}

    # Prepare query parameters
    if auto_language:
        params = {'output': output_format}
    else:
        params = {'output': output_format, 'language': language}

    click.echo(f"Transcribing {input_file}...")
    click.echo(f"Output will be saved to {output_file}")

    start_time = time.time()
    try:
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


# https://ahmetoner.com/whisper-asr-webservice/endpoints/
SUPPORTED_OUTPUT_FORMATS: Final[tuple[str, ...]] = ('srt', 'vtt', 'text', 'json', 'tsv')


@click.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
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
def transcribe(
    input_path: Path,
    url: str,
    language: str,
    auto_language: bool,
    output_format: str,
    output_file: Path | None,
    dry_run: bool,
):
    """Transcribe audio/video file(s) using Whisper ASR Web Service.

    INPUT_PATH: Path to video file or directory containing video files
    """
    if input_path.is_file():
        if dry_run:
            output_name = (
                output_file
                if output_file
                else input_path.with_suffix(f'.{language if not auto_language else ""}{output_format}')
            )
            click.echo(f"Would process: {input_path}")
            click.echo(f"Would save to: {output_name}")
            return
        process_single_file(input_path, url, language, auto_language, output_format, output_file)
    else:
        # Scan directory and collect files
        video_files: list[Path] = []
        click.echo(f"Scanning directory {input_path} for video files...")

        for pattern in VIDEO_FORMATS:
            video_files.extend(input_path.glob(pattern))

        if not video_files:
            click.echo("No video files found in the directory.")
            return

        # Sort files by name for consistent ordering
        video_files.sort()

        # Preview files to be processed
        click.echo(f"\nFound {len(video_files)} video files:")
        to_process: list[Path] = []
        to_skip: list[Path] = []

        for video_file in video_files:
            output_file = video_file.with_suffix(f'.{language if not auto_language else ""}{output_format}')
            if output_file.exists():
                to_skip.append(video_file)
            else:
                to_process.append(video_file)

        if to_process:
            click.echo("\nFiles to be processed:")
            for idx, file in enumerate(to_process, 1):
                output_name = file.with_suffix(f'.{language if not auto_language else ""}{output_format}')
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

        with click.progressbar(
            to_process, label='Processing files', item_show_func=lambda p: p.name if p else ''
        ) as files:
            for video_file in files:
                success, elapsed_time = process_single_file(
                    video_file, url, language, auto_language, output_format, None
                )
                if success:
                    processed_files += 1
                    total_time += elapsed_time

        # Show summary
        avg_time = total_time / processed_files if processed_files > 0 else 0
        click.echo(f"\nBatch processing complete:")
        click.echo(f"Total files found: {len(video_files)}")
        click.echo(f"Files skipped: {len(to_skip)}")
        click.echo(f"Files processed: {processed_files} of {len(to_process)}")
        click.echo(f"Total processing time: {timedelta(seconds=int(total_time))}")
        if processed_files > 0:
            click.echo(f"Average processing time per file: {timedelta(seconds=int(avg_time))}")


if __name__ == '__main__':
    transcribe()  # pylint: disable=no-value-for-parameter
