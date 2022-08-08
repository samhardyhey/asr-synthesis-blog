## ASR Synthesis
Notebooks and scripts for the synthesis of stereo call transcripts. See the accompanying blog post here.

## Install
Python libraries via:
- `pip install -e requirements.txt`

Additional libraries for audio file manipulation (required):
```
apt-get install ffmpeg
```

Additional libraries for pyttsx3 (optional):
```
apt-get update -y
apt-get install -y libespeak-dev
!apt-get install alsa-utils -y
!apt update && apt install espeak libespeak1 -y
```
- Note some of the models referenced within `1_transcript_generation.ipynb` can be quite large (5G+)

## Usage