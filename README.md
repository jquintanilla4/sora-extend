# Sora Extend

This is a fork of a fork. 

Original https://github.com/mshumer/sora-extend

Fork https://github.com/stevemoraco/sora-extend

**Goal of this fork is to use OpenRouter and Fal.ai as the API endpoints**

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

Generate long-form AI videos effortlessly with Sora Extend.
