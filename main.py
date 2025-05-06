from kokoro import KModel, phonemize, set_lexicon
import torch
import torch.nn.utils.rnn as rnn
import json
import scipy.io.wavfile as wav

text = [
    "This is a test!",
    "This is a test! I'm going to the store.",
]
model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True, voice_name="af_heart").to("cpu").eval()
set_lexicon(json.load(open("./misaki_lexicons/us_gold.json")) | json.load(open("./misaki_lexicons/us_silver.json")))

input_id_tensors = []

for t in text:
    ps = phonemize(t)["ps"]
    toks = list(filter(lambda i: i is not None, map(lambda p: model.vocab.get(p), ps)))
    input_id_tensors.append(torch.tensor([0,*toks,0], dtype=torch.long))
input_lengths = torch.tensor([toks.shape[0] for toks in input_id_tensors], dtype=torch.long)
input_ids = rnn.pad_sequence(input_id_tensors, batch_first=True, padding_value=0)
print(input_lengths, input_ids)

audio, pred_dur = model.forward_with_tokens(input_ids, 1.0,input_lengths)

wav.write("./output.wav", 24000, audio[1][0].numpy())