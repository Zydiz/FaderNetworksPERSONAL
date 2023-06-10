import os
import argparse
import numpy as np
import torch
import librosa
import librosa.display
import soundfile as sf

from src.logger import create_logger
from src.loader import normalize_images

melspec_params = {
    'n_mels': 256,
    'duration': 64000,
    'hop_length': ((4*16000)//256+1),
    'n_fft': 2048,
    'fmin': 20
}

attr_keys = ['bright', 'dark', 'distortion', 'percussive']

#############
### functions
#############

def load_audio(params, file_path):
    y, sr = librosa.load(file_path, sr = None)

    # clip silence
    yt, index = librosa.effects.trim(y, top_db=60)
    # print(len(yt))

    # pad to a length of 4s
    if len(yt) > params['duration']:
        yt = yt[:params['duration']]
    else:
        padding = params['duration'] - len(yt)
        offset = padding // 2
        yt = np.pad(yt, (offset, padding - offset), 'constant')
        #print('size of Yt is')
        #print(len(yt))
    return yt, sr

# parse parameters
parser = argparse.ArgumentParser(description='Attributes swapping')
parser.add_argument("--model_path", type=str,
                    help="Trained model path", required=True)
parser.add_argument("--audio_path", type=str,
                    help="Input audio path", required=True)
parser.add_argument('--attrib_list', type=str,
                    help='List of attributes in the form of ex. 1,1,0,1', required=True)
parser.add_argument("--output_path", type=str, default="output.wav",
                    help="Output path (.wav)")
params = parser.parse_args()

# check parameters
assert os.path.isfile(params.model_path)
assert params.output_path.lower().endswith('.wav')

################
### Attr Parsing
################

attrib_list = [item for item in params.attrib_list.split(',')]
attribs = {k: 0 for k in attr_keys}
attribs["bright"]     = int(attrib_list[0])
attribs["dark"]       = int(attrib_list[1])
attribs["distortion"] = int(attrib_list[2])
attribs["percussive"] = int(attrib_list[3])

# parse attributes
attrs = []
for name in attr_keys:
    a_0 = 1 - attribs[name]
    attrs.append(a_0)
    a_1 = attribs[name]
    attrs.append(a_1)

attributes = torch.FloatTensor(attrs)
attributes = attributes.unsqueeze(0)

# create logger / load trained model
logger = create_logger(None)
ae = torch.load(params.model_path).eval()

######################
### Preprocessing step
######################

x, sr = load_audio(melspec_params, params.audio_path)
mels = []

melspec = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=melspec_params['n_mels'] ,hop_length=melspec_params['hop_length'])
melspec_pad = np.pad(melspec, ((0, 0), (0, 1)), mode='constant')
mels.append(melspec_pad)

mels_n = np.array(mels)
mels_n = np.expand_dims(mels_n, axis=1)

data_m = torch.from_numpy(mels_n)

# TODO check if true
image = normalize_images(data_m.cpu())

#####################
### Reconstruct image
#####################

enc_outputs = ae.encode(image)
# print(enc_outputs[2].size())
print(enc_outputs[0].size(0))
output = ae.decode(enc_outputs, attributes)[-1]

#####################
### Reconstruct audio
#####################
output_np = output.detach().numpy()
output_np = output_np.squeeze()
print(f"output shape:{output_np.shape}")
y_inv = librosa.feature.inverse.mel_to_audio(output_np, sr=sr, hop_length=((4*16000)//256+1))
print(f"y shape:{y_inv.shape}")
sf.write(params.output_path, y_inv, sr)