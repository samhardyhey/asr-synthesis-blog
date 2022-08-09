## ASR Synthesis
Notebooks and scripts for the synthesis of stereo call transcripts. See the accompanying blog post here.

## Install
- Conda env creation, python dependencies, low-level audio tools via `create_env.sh`
- Note some of the models referenced within `1_transcript_generation.ipynb` can be quite large (5G+)

## Usage
- Via `python synthesise_transcript_audio.py ./output`
- Additional parameters (number of self-chats) can be adjusted within the main script