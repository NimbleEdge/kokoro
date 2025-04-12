from .istftnet import Decoder
from .modules import CustomAlbert, ProsodyPredictor, TextEncoder
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from loguru import logger
from torch import nn
from transformers import AlbertConfig
from typing import Dict, Optional, Union
import json
import torch

class KModel(torch.nn.Module):
    '''
    KModel is a torch.nn.Module with 2 main responsibilities:
    1. Init weights, downloading config.json + model.pth from HF if needed
    2. forward(phonemes: str, ref_s: FloatTensor) -> (audio: FloatTensor)

    You likely only need one KModel instance, and it can be reused across
    multiple KPipelines to avoid redundant memory allocation.

    Unlike KPipeline, KModel is language-blind.

    KModel stores self.vocab and thus knows how to map phonemes -> input_ids,
    so there is no need to repeatedly download config.json outside of KModel.
    '''

    MODEL_NAMES = {
        'hexgrad/Kokoro-82M': 'kokoro-v1_0.pth',
        'hexgrad/Kokoro-82M-v1.1-zh': 'kokoro-v1_1-zh.pth',
    }

    def __init__(
        self,
        repo_id: Optional[str] = None,
        config: Union[Dict, str, None] = None,
        model: Optional[str] = None,
        disable_complex: bool = False
    ):
        super().__init__()
        if repo_id is None:
            repo_id = 'hexgrad/Kokoro-82M'
            print(f"WARNING: Defaulting repo_id to {repo_id}. Pass repo_id='{repo_id}' to suppress this warning.")
        self.repo_id = repo_id
        if not isinstance(config, dict):
            if not config:
                logger.debug("No config provided, downloading from HF")
                config = hf_hub_download(repo_id=repo_id, filename='config.json')
            with open(config, 'r', encoding='utf-8') as r:
                config = json.load(r)
                logger.debug(f"Loaded config: {config}")
        self.vocab = config['vocab']
        self.bert = CustomAlbert(AlbertConfig(vocab_size=config['n_token'], **config['plbert']))
        self.bert_encoder = torch.nn.Linear(self.bert.config.hidden_size, config['hidden_dim'])
        self.context_length = self.bert.config.max_position_embeddings
        self.predictor = ProsodyPredictor(
            style_dim=config['style_dim'], d_hid=config['hidden_dim'],
            nlayers=config['n_layer'], max_dur=config['max_dur'], dropout=config['dropout']
        )
        self.text_encoder = TextEncoder(
            channels=config['hidden_dim'], kernel_size=config['text_encoder_kernel_size'],
            depth=config['n_layer'], n_symbols=config['n_token']
        )
        self.decoder = Decoder(
            dim_in=config['hidden_dim'], style_dim=config['style_dim'],
            dim_out=config['n_mels'], disable_complex=disable_complex, **config['istftnet']
        )
        if not model:
            model = hf_hub_download(repo_id=repo_id, filename=KModel.MODEL_NAMES[repo_id])
        for key, state_dict in torch.load(model, map_location='cpu', weights_only=True).items():
            assert hasattr(self, key), key
            try:
                getattr(self, key).load_state_dict(state_dict)
            except:
                logger.debug(f"Did not load {key} from state_dict")
                state_dict = {k[7:]: v for k, v in state_dict.items()}
                getattr(self, key).load_state_dict(state_dict, strict=False)

    @property
    def device(self):
        return self.bert.device

    @dataclass
    class Output:
        audio: torch.FloatTensor
        pred_dur: Optional[torch.LongTensor] = None

    @torch.no_grad()
    def forward_with_tokens(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1,
        input_lengths: Optional[torch.LongTensor] = None
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        batch_size = input_ids.shape[0]
        # If input_lengths is not provided, assume all sequences use full length
        if input_lengths is None:
            input_lengths = torch.full(
                (batch_size,), 
                input_ids.shape[-1], 
                device=input_ids.device,
                dtype=torch.long
            )
        s = ref_s[:, :, 128:] # b x 1 x sty_dim
            
        # Create text mask where True means "this position is a padding token"
        max_len = input_ids.shape[1]
        text_mask = torch.arange(max_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        text_mask = (text_mask >= input_lengths.unsqueeze(1)).to(self.device)
        
        # Convert to attention mask where 1 means "attend to this token" and 0 means "ignore this token"
        attention_mask = (~text_mask).float()
        
        # Forward pass through BERT
        bert_dur = self.bert(input_ids, attention_mask=attention_mask) # b x seq_len x hidden
        d_en = self.bert_encoder(bert_dur) # b x seq_len x hidden
        
        # Pass through predictor
        d = self.predictor.text_encoder(d_en, s, input_lengths) # b x seq_len x (d_model + sty_dim)

        x = nn.utils.rnn.pack_padded_sequence(d, input_lengths, batch_first=True, enforce_sorted=False)
        self.predictor.lstm.flatten_parameters()
        x, _ = self.predictor.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True) # b x seq_len x d_model


        duration = self.predictor.duration_proj(x) # b x seq_len x max_dur
        duration = torch.sigmoid(duration).sum(axis=-1) / speed # b x seq_len
        # For each sequence, we only care about the non-padded tokens
        # Mask out durations for padded tokens
        duration = torch.round(duration * attention_mask).clamp(min=1).long() # b x seq_len
        updated_seq_lengths = torch.sum(duration, dim=-1) # b
        # Flatten the durations for each sequence
        pred_dur = duration.view(-1) # b*seq_len
        d_flat = d.view(-1, d.shape[-1]) # b*seq_len x (d_model + sty_dim)
        # Create alignment target
        indices = torch.repeat_interleave(torch.arange(0, pred_dur.shape[0], device=self.device), pred_dur) # b*max_dur
        pred_aln_trg = torch.zeros((pred_dur.shape[0], indices.shape[0]), device=self.device) # b*max_dur x b*seq_len
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        en = pred_aln_trg @ d_flat # b*max_dur x (d_model + sty_dim)
        en = en.view(batch_size, -1, en.shape[-1]) # b x max_dur x (d_model + sty_dim)

        F0_pred, N_pred = self.predictor.F0Ntrain(en, s, input_lengths) # b x 1 x max_dur, b x 1 x max_dur

        t_en = self.text_encoder(input_ids, input_lengths) # b x seq_len x d_model
        t_en_flat = t_en.view(-1, t_en.shape[-1]) # b*seq_len x d_model
        asr = pred_aln_trg @ t_en_flat # b*max_dur x d_model

        #TODO check if this is correct
        audio = self.decoder(asr, F0_pred, N_pred, s).squeeze()
        #TODO check end
        return audio, pred_dur

    def forward(
        self,
        phonemes: str,
        ref_s: torch.FloatTensor,
        speed: float = 1,
        return_output: bool = False
    ) -> Union['KModel.Output', torch.FloatTensor]:
        input_ids = list(filter(lambda i: i is not None, map(lambda p: self.vocab.get(p), phonemes)))
        logger.debug(f"phonemes: {phonemes} -> input_ids: {input_ids}")
        assert len(input_ids)+2 <= self.context_length, (len(input_ids)+2, self.context_length)
        input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(self.device)
        ref_s = ref_s.to(self.device)
        audio, pred_dur = self.forward_with_tokens(input_ids, ref_s, speed)
        audio = audio.squeeze().cpu()
        pred_dur = pred_dur.cpu() if pred_dur is not None else None
        logger.debug(f"pred_dur: {pred_dur}")
        return self.Output(audio=audio, pred_dur=pred_dur) if return_output else audio

class KModelForONNX(torch.nn.Module):
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.kmodel = kmodel

    def forward(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        waveform, duration = self.kmodel.forward_with_tokens(input_ids, ref_s, speed)
        return waveform, duration
