# Sora Extend

This is a fork of a fork. **WIP**

Original https://github.com/mshumer/sora-extend

Fork https://github.com/stevemoraco/sora-extend

**Goal of this fork is to use OpenRouter and Fal.ai as the API endpoints on both the Notebook and the new TUI app**

**Seamlessly generate extended Sora 2 videos beyond OpenAI’s 12-second limit.**

OpenAI’s Sora video generation model currently restricts output to 12-second clips. By leveraging the final frame of each generation as context for the next, and intelligently breaking down your prompt into coherent segments that mesh well, Sora Extend enables the creation of high-quality, extended-duration videos with continuity.

---

## How it Works

1. **Prompt Deconstruction**

   * Your initial prompt is intelligently broken down into smaller, coherent segments suitable for Sora 2’s native generation limits, with additional context that allows each subsequent prompt to have a sense of what happened before it.

2. **Sequential Video Generation**

   * Each prompt segment is independently processed by Sora 2, sequentially, generating video clips that align smoothly both visually and thematically. The final frame of each generated clip is captured and fed into the subsequent generation step as contextual input, helping with visual consistency.

3. **Concatenation**

   * Generated video segments are concatenated automatically, resulting in a single continuous video output without noticeable transitions or interruptions.

---

## Setup

1. **Install dependencies**  
   The project targets Python **3.11**. We recommend using the fast `uv` package manager so installs stay reproducible:
   ```bash
   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```
   If you prefer stock tooling, replace the last line with `pip install -r requirements.txt`. Dependencies include the OpenRouter SDK (`openai`), Fal.ai client, OpenCV for frame extraction, Rich for the interactive TUI, and optional tooling for video concatenation (`moviepy` or ffmpeg via `imageio-ffmpeg`).

2. **Configure environment variables**  
   Create a `.env` file with your API keys:
   ```bash
   OPENROUTER_KEY=your_openrouter_api_key
   FAL_KEY=your_fal_ai_api_key
   ```
   These keys are loaded automatically at runtime. The script will exit with a helpful error if either key is missing.

---

## Usage

### Interactive TUI Workflow

```bash
python sora_extend.py
```

Running the script launches an interactive Rich-powered TUI. Each prompt shows its default value (press **Enter** to accept) or lets you provide custom input:

| Prompt | Default | Description |
|--------|---------|-------------|
| Base prompt | — | High-level description of the video you want. |
| Number of segments | 2 | Total clips to generate and stitch together. |
| Duration per segment | 8 | Clip length in seconds (4, 8, or 12). |
| Video resolution | 720p | Choose 720p or 1080p output. |
| Video aspect ratio | 16:9 | Swap to 9:16 for vertical content. |
| Output directory | `sora_output` | Folder for segment MP4s, last-frame JPGs, and the final combined video. |
| Planner model | `openai/gpt-5` | Any OpenRouter-accessible model that supports JSON responses. |
| Show progress output? | `Yes` | Toggle verbose Fal.ai progress logs. |

After you confirm your choices, the script:

1. Calls OpenRouter to plan detailed segment prompts from your base idea.
2. Uses Fal.ai’s Sora 2 Pro endpoints to render each segment, feeding the last frame forward for continuity.
3. Concatenates the generated clips into `combined.mp4` using MoviePy when available, otherwise ffmpeg.

All prompts and the resulting plan are echoed to the terminal so you can review them before rendering begins.

### Examples
```text
> python sora_extend.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SORA EXTEND - INTERACTIVE SETUP
Press Enter to accept the highlighted defaults.

Base prompt describing the video concept: A robot dancing in a neon city
Number of segments to generate [2]:
Duration per segment (seconds) (4, 8, 12) [8]:
Video resolution (720p, 1080p) [720p]:
Video aspect ratio (16:9, 9:16) [16:9]:
Output directory [sora_output]:
Planner model [openai/gpt-5]:
Show progress output during generation? [Y/n]:
```

Once you confirm, the planner output is displayed, segment rendering begins, and the combined video is produced automatically.

---

## Output

The script generates:
- Individual segment files: `segment_01.mp4`, `segment_02.mp4`, etc.
- Last frame images for continuity: `segment_01_last.jpg`, etc.
- Final combined video: `combined.mp4`

All files are saved in the specified output directory (default: `sora_output/`).

---

## Troubleshooting

### Missing API Keys
```
Error: OPENROUTER_KEY not found in environment variables
```
**Solution**: Create a `.env` file with your API keys (see Setup section)

### FFmpeg Not Found
```
RuntimeError: FFmpeg not found and MoviePy unavailable
```
**Solution**: Install ffmpeg on your system:
- macOS: `brew install ffmpeg`
- Ubuntu: `apt-get install ffmpeg`
- Or let the script install `imageio-ffmpeg` automatically

### Video Concatenation Fails
**Solution**: Ensure all segments were generated successfully and are valid MP4 files

---

## Notes

- Total video length = `segments × duration`
- First segment uses text-to-video (no input image)
- Subsequent segments use image-to-video with the last frame from the previous segment
- The planner model intelligently creates prompts with continuity context
- Progress updates can be disabled by responding "No" when asked about showing progress output

---

Generate long-form AI videos effortlessly with Sora Extend.
