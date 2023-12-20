from collections import namedtuple
from functools import partial
import gc
from pathlib import Path
from typing import Optional, Tuple
import warnings

from IPython.display import Audio
import openvino as ov
import numpy as np
import torch
from torch.jit import TracerWarning
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions

#import winsound
from pydub import AudioSegment
from pydub.playback import play

#def soundsetI():

# Ignore tracing warnings
warnings.filterwarnings("ignore", category=TracerWarning)

# Load the pipeline
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", torchscript=True, return_dict=False)


device = "cpu"
sample_length = 8  # seconds

n_tokens = sample_length * model.config.audio_encoder.frame_rate + 3
sampling_rate = model.config.audio_encoder.sampling_rate
print('Sampling rate is', sampling_rate, 'Hz')

model.to(device)
model.eval();

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

inputs = processor(
    text=["80s pop track with bassy drums and synth"],
    return_tensors="pt",
)

audio_values = model.generate(**inputs.to(device), do_sample=True, guidance_scale=3, max_new_tokens=n_tokens)

def play_audio_numpy(audio_data, rate):
    audio = AudioSegment(
        audio_data.tobytes(),
        frame_rate=rate,
        sample_width=audio_data.dtype.itemsize,
        channels=1  # Assuming mono audio
    )
    play(audio)

# 위의 play_audio_numpy 함수를 이용하여 오디오 재생 부분을 수정
play_audio_numpy(audio_values[0].cpu().numpy(), rate=sampling_rate)
# Audio(audio_values[0].cpu().numpy(), rate=sampling_rate)
#audio_values[0].cpu().numpy()
#Audio(output_path)
