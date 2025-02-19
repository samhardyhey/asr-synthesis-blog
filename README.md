# ASR Synthesis 🎙️

Generate synthetic stereo call transcripts using ASR and speech synthesis. Companion code for the blog post ["Poor Man's ASR (Part 2)"](https://www.samhardyhey.com/poor-mans-asr-pt-2).

## Features
- 🗣️ Synthetic conversation generation
- 🎧 Stereo audio synthesis
- 📝 Automated transcription
- 🤖 Large language model integration

## Setup
```bash
# Install dependencies and audio tools
./create_env.sh
```

## Usage
```bash
# Generate synthetic transcripts
python synthesise_transcript_audio.py ./output
```

## Structure
- 📓 `1_transcript_generation.ipynb` # Main synthesis notebook
- 🛠️ `synthesise_transcript_audio.py` # CLI tool
- ⚙️ `create_env.sh` # Environment setup

*Note: Some models require 5GB+ storage. See notebook for configuration options.*