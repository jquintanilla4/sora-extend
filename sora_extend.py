#!/usr/bin/env python3

"""
Pipeline:
1) Use an LLM (GPT-5 Thinking) to plan N scene prompts from a base idea
2) Render each segment with Sora 2; for continuity, pass the prior segment's final frame
3) Concatenate segments into a single MP4
"""

import sys
import subprocess
import os
import re
import json
import shutil
from pathlib import Path
import requests
import cv2
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.prompt import Prompt, IntPrompt, Confirm

from planner_prompt import PLANNER_SYSTEM_INSTRUCTIONS

console = Console()


# PLANNER FUNCTION

def plan_prompts_with_ai(client, planner_model, base_prompt, seconds_per_segment, num_generations):
    """
    Calls OpenRouter API (via OpenAI SDK) to produce a JSON object with segment prompts.
    
    Args:
        client: OpenAI client configured for OpenRouter
        planner_model: Model name to use for planning
        base_prompt: Base prompt describing the video concept
        seconds_per_segment: Duration per segment
        num_generations: Total number of segments to generate
    
    Returns:
        List of segment dicts with title, seconds, and prompt
    """
    user_input = f"""
    BASE PROMPT: {base_prompt}
    
    GENERATION LENGTH (seconds): {seconds_per_segment}
    TOTAL GENERATIONS: {num_generations}
    
    Return exactly {num_generations} segments.
    """.strip()
    
    print(f"\nPlanning {num_generations} segments with {planner_model}...")
    
    response = client.chat.completions.create(
        model=planner_model,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": user_input}
        ],
        max_tokens=4000,
        temperature=0.7,
        response_format={"type": "json_object"},
        extra_body={
            "reasoning": {
                "effort": "medium",
            }
        },
    )

    text = (response.choices[0].message.content or "").strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback to extracting JSON payload if the model returned extra text
        m = re.search(r'\{[\s\S]*\}', text)
        if not m:
            raise ValueError("Planner did not return JSON. Inspect response and adjust instructions.")
        data = json.loads(m.group(0))
    
    segments = data.get("segments", [])
    
    if len(segments) < num_generations:
        raise ValueError(
            f"Planner returned {len(segments)} segments but {num_generations} were requested. "
            f"LLM output needs fixing."
        )
    
    if len(segments) > num_generations:
        segments = segments[:num_generations]
    
    # Force durations to the requested number
    for seg in segments:
        seg["seconds"] = int(seconds_per_segment)
    
    print("✓ Planning complete\n")
    
    return segments


# SORA VIDEO GENERATION

def create_and_poll_video(prompt, resolution, aspect_ratio, duration, input_image=None, 
                          text_to_video_endpoint="fal-ai/sora-2/text-to-video/pro",
                          image_to_video_endpoint="fal-ai/sora-2/image-to-video/pro",
                          print_progress=True):
    """
    Create a video using Fal AI's Sora 2 Pro endpoint.
    
    Args:
        prompt: Text prompt for video generation
        resolution: Video resolution - "720p" or "1080p"
        aspect_ratio: Video aspect ratio - "16:9" or "9:16"
        duration: Duration in seconds - 4, 8, or 12
        input_image: Optional input image path for continuity
        text_to_video_endpoint: Fal AI text-to-video endpoint
        image_to_video_endpoint: Fal AI image-to-video endpoint
        print_progress: Whether to print progress updates
    
    Returns:
        Result dict from Fal AI
    """
    import fal_client
    image_path = Path(input_image) if input_image is not None else None
    
    def on_queue_update(update):
        if isinstance(update, fal_client.InProgress) and print_progress:
            for log in update.logs:
                print(f"  {log['message']}")
    
    # Select endpoint
    if image_path is not None:
        endpoint = image_to_video_endpoint
        print(f"  Using image-to-video endpoint (continuity mode)")
        print(f"  Input image: {image_path.name}")
    else:
        endpoint = text_to_video_endpoint
        print(f"  Using text-to-video endpoint (first segment)")
    
    # Build arguments
    arguments = {
        "prompt": prompt,
        "resolution": resolution,
        "aspect_ratio": aspect_ratio,
        "duration": duration
    }
    
    # Upload input image if provided
    if image_path is not None:
        image_url = fal_client.upload_file(str(image_path))
        arguments["image_url"] = image_url
    
    print(f"  Submitting to Fal AI ({endpoint})...")
    print(f"  Resolution: {resolution}, Aspect ratio: {aspect_ratio}, Duration: {duration}s")
    
    result = fal_client.subscribe(
        endpoint,
        arguments=arguments,
        with_logs=True,
        on_queue_update=on_queue_update,
    )
    
    return result


def download_fal_video(result, out_path):
    """Download video from Fal AI result"""
    video_url = result.get("video", {}).get("url")
    if not video_url:
        raise RuntimeError(f"No video URL in Fal AI result: {result}")
    
    print(f"  Downloading video from {video_url}...")
    
    with requests.get(video_url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    
    return out_path


def extract_last_frame(video_path, out_image_path):
    """Extract the last frame from a video file using OpenCV"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {video_path}")
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    success, frame = False, None
    
    if total > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
        success, frame = cap.read()
    if not success or frame is None:
        cap.release()
        cap = cv2.VideoCapture(str(video_path))
        while True:
            ret, f = cap.read()
            if not ret:
                break
            frame = f
            success = True
    cap.release()
    
    if not success or frame is None:
        raise RuntimeError(f"Could not read last frame from {video_path}")
    
    out_image_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_image_path), frame)
    if not ok:
        raise RuntimeError(f"Failed to write {out_image_path}")
    return out_image_path


# CHAIN GENERATION

def chain_generate_sora(segments, resolution, aspect_ratio, seconds_per_segment, out_dir, 
                       text_to_video_endpoint, image_to_video_endpoint, print_progress):
    """
    Generate video segments with continuity.
    
    Args:
        segments: List of segment dicts with title, seconds, and prompt
        resolution: Video resolution
        aspect_ratio: Video aspect ratio
        seconds_per_segment: Duration per segment
        out_dir: Output directory path
        text_to_video_endpoint: Fal AI text-to-video endpoint
        image_to_video_endpoint: Fal AI image-to-video endpoint
        print_progress: Whether to print progress updates
    
    Returns:
        List of video segment paths
    """
    input_ref = None
    segment_paths = []
    
    for i, seg in enumerate(segments, start=1):
        secs = int(seg["seconds"])
        prompt = seg["prompt"]
        
        print(f"\n=== Generating Segment {i}/{len(segments)} — {secs}s ===")
        
        # Generate video
        result = create_and_poll_video(
            prompt=prompt,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            duration=seconds_per_segment,
            input_image=input_ref,
            text_to_video_endpoint=text_to_video_endpoint,
            image_to_video_endpoint=image_to_video_endpoint,
            print_progress=print_progress
        )
        
        # Download video
        seg_path = out_dir / f"segment_{i:02d}.mp4"
        download_fal_video(result, seg_path)
        print(f"  Saved {seg_path}")
        segment_paths.append(seg_path)
        
        # Extract final frame for next segment
        if i < len(segments):
            frame_path = out_dir / f"segment_{i:02d}_last.jpg"
            extract_last_frame(seg_path, frame_path)
            print(f"  Extracted last frame -> {frame_path}")
            input_ref = frame_path
    
    return segment_paths


def concatenate_segments(segment_paths, out_path):
    """
    Concatenate video segments using MoviePy when available, otherwise fall back to ffmpeg.
    
    Args:
        segment_paths: List of video segment paths
        out_path: Output path for combined video
    
    Returns:
        Path to combined video
    """
    if not segment_paths:
        raise ValueError("No segments to concatenate")
    
    try:
        from moviepy.editor import VideoFileClip, concatenate_videoclips
        moviepy_available = True
    except ImportError:
        moviepy_available = False
    
    if moviepy_available:
        print("\nConcatenating segments with MoviePy...")
        clips = [VideoFileClip(str(p)) for p in segment_paths]
        target_fps = clips[0].fps or 24
        result = concatenate_videoclips(clips, method="compose")
        result.write_videofile(
            str(out_path),
            codec="libx264",
            audio_codec="aac",
            fps=target_fps,
            preset="medium",
            threads=0
        )
        for c in clips:
            c.close()
    else:
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            try:
                import imageio_ffmpeg

                ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception as exc:
                raise RuntimeError(
                    "ffmpeg is not available. Install MoviePy or make ffmpeg accessible in PATH."
                ) from exc
        
        print("\nConcatenating segments with ffmpeg...")
        
        # Create temporary file list
        list_file = out_path.parent / "concat_list.txt"
        with open(list_file, "w") as f:
            for seg_path in segment_paths:
                f.write(f"file '{seg_path.absolute()}'\n")
        
        # Run ffmpeg concat
        cmd = [
            ffmpeg_bin,
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(out_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {result.stderr}")
        
        # Clean up
        list_file.unlink()
    
    return out_path


# INTERACTIVE PROMPTS

def _format_prompt(prompt: str) -> str:
    """Consistent Rich formatting for interactive prompts."""
    return f"[bold]{prompt}[/bold]"


def _ask_text(prompt, default=None, required=False):
    """Prompt the user for text input with optional default."""
    formatted_prompt = _format_prompt(prompt)
    if isinstance(default, str):
        default = default.strip()
    while True:
        raw_value = Prompt.ask(formatted_prompt, default=default)
        value = (raw_value or "").strip()
        if value:
            return value
        if default is not None:
            return default
        if not required:
            return ""
        console.print("[red]Please enter a value.[/red]")


def _ask_int(prompt, default, minimum=None):
    """Prompt the user for an integer, enforcing optional minimum."""
    formatted_prompt = _format_prompt(prompt)
    while True:
        value = IntPrompt.ask(formatted_prompt, default=default)
        if minimum is None or value >= minimum:
            return value
        console.print(f"[red]Please enter a value >= {minimum}.[/red]")


def _ask_choice(prompt, options, default):
    """Prompt the user to select from enumerated options."""
    formatted_prompt = _format_prompt(prompt)
    return Prompt.ask(
        formatted_prompt,
        choices=options,
        default=default,
        show_choices=True,
        case_sensitive=False,
    )


def _ask_yes_no(prompt, default=True):
    """Prompt for a yes/no answer using Rich."""
    return Confirm.ask(_format_prompt(prompt), default=default)


def collect_user_configuration():
    """Collect run configuration from the user via interactive prompts."""
    console.print()
    console.rule("[bold cyan]SORA EXTEND - INTERACTIVE SETUP[/bold cyan]")
    console.print("[dim]Press Enter to accept the highlighted defaults.[/dim]\n")

    base_prompt = _ask_text("Base prompt describing the video concept", required=True)
    segments = _ask_int("Number of segments to generate", default=2, minimum=1)
    duration = int(_ask_choice("Duration per segment (seconds)", ["4", "8", "12"], default="8"))
    resolution = _ask_choice("Video resolution", ["720p", "1080p"], default="720p")
    aspect_ratio = _ask_choice("Video aspect ratio", ["16:9", "9:16"], default="16:9")
    output_dir = _ask_text("Output directory", default="sora_output")
    planner_model = _ask_text("Planner model", default="openai/gpt-5")
    show_progress = _ask_yes_no("Show progress output during generation?", default=True)

    return {
        "prompt": base_prompt,
        "segments": segments,
        "duration": duration,
        "resolution": resolution,
        "aspect_ratio": aspect_ratio,
        "output_dir": output_dir,
        "planner_model": planner_model,
        "print_progress": show_progress,
    }


# MAIN FUNCTION
def main():
    # Step 1: Setup configuration and load up environment variables
    config = collect_user_configuration()
    
    # Load environment variables
    load_dotenv()
    openrouter_key = os.getenv("OPENROUTER_KEY")
    fal_key = os.getenv("FAL_KEY")
    
    if not openrouter_key:
        print("Error: OPENROUTER_KEY not found in environment variables")
        print("Please set it in your .env file")
        sys.exit(1)
    
    if not fal_key:
        print("Error: FAL_KEY not found in environment variables")
        print("Please set it in your .env file")
        sys.exit(1)
    
    # Set up OpenRouter client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_key,
    )
    
    # Create output directory
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Sora endpoints
    text_to_video_endpoint = "fal-ai/sora-2/text-to-video/pro"
    image_to_video_endpoint = "fal-ai/sora-2/image-to-video/pro"
    
    print("\n" + "=" * 80)
    print("SORA EXTEND - AI-Planned Video Generation with Continuity")
    print("=" * 80)
    print(f"\nBase prompt: {config['prompt']}")
    print(f"Segments: {config['segments']}")
    print(f"Duration per segment: {config['duration']}s")
    print(f"Total duration: {config['segments'] * config['duration']}s")
    print(f"Resolution: {config['resolution']}")
    print(f"Aspect ratio: {config['aspect_ratio']}")
    print(f"Output directory: {out_dir}")
    print(f"Planner model: {config['planner_model']}")
    
    # Step 2: Plan prompts with AI
    segments_plan = plan_prompts_with_ai(
        client=client,
        planner_model=config["planner_model"],
        base_prompt=config["prompt"],
        seconds_per_segment=config["duration"],
        num_generations=config["segments"]
    )
    
    print("AI-planned segments:\n")
    for i, seg in enumerate(segments_plan, start=1):
        print(f"[{i:02d}] {seg['seconds']}s — {seg.get('title', '(untitled)')}")
        print(seg["prompt"])
        print("-" * 80)
    
    # Step 3: Generate segments with continuity
    segment_paths = chain_generate_sora(
        segments=segments_plan,
        resolution=config["resolution"],
        aspect_ratio=config["aspect_ratio"],
        seconds_per_segment=config["duration"],
        out_dir=out_dir,
        text_to_video_endpoint=text_to_video_endpoint,
        image_to_video_endpoint=image_to_video_endpoint,
        print_progress=config["print_progress"]
    )
    
    # Step 4: Concatenate segments
    final_path = out_dir / "combined.mp4"
    concatenate_segments(segment_paths, final_path)
    
    print("\n" + "=" * 80)
    print(f"✓ SUCCESS! Combined video saved to: {final_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
