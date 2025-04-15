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
        speed: float,
        input_lengths: Optional[torch.LongTensor]
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        s = ref_s[:, :, 128:] # b x 1 x sty_dim
    
        max_len = input_ids.shape[1]
        text_mask = torch.arange(max_len, device=self.device).unsqueeze(0)
        sequence_mask = (text_mask.expand(input_ids.shape[0], -1) >= input_lengths.unsqueeze(1)).to(self.device) # b x seq_len
        # Convert to attention mask where 1 means "attend to this token" and 0 means "ignore this token"
        attention_mask = (~sequence_mask).float()
        
        # Forward pass through BERT
        bert_dur = self.bert(input_ids, attention_mask=attention_mask) # b x seq_len x hidden
        d_en = self.bert_encoder(bert_dur) # b x seq_len x hidden
        
        # Pass through predictor
        d = self.predictor.text_encoder(d_en, s, input_lengths, sequence_mask) # b x seq_len x (d_model + sty_dim)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x) # b x seq_len x max_dur
        duration = torch.sigmoid(duration).sum(axis=-1) / speed # b x seq_len
        # For each sequence, we only care about the non-padded tokens
        # Mask out durations for padded tokens
        duration = torch.round(duration * attention_mask).clamp(min=1).long() # b x seq_len
        updated_seq_lengths = torch.sum(duration, dim=-1) # b
        max_frames = updated_seq_lengths.max()
        
        frame_indices = torch.arange(max_frames, device=self.device).view(1,1,-1) # 1 x 1 x max_dur
        duration_cumsum = duration.cumsum(dim=1).unsqueeze(-1) # b x seq_len x 1
        mask1 = duration_cumsum > frame_indices # b x seq_len x max_dur
        mask2 = frame_indices >= torch.cat([torch.zeros(duration.shape[0],1, 1), duration_cumsum[:,:-1,:]],dim=1) # b x seq_len x max_dur
        pred_aln_trg = (mask1 & mask2).float().transpose(1, 2) # b x max_dur x seq_len 
        en = torch.bmm(pred_aln_trg, d) # b x max_dur x (d_model + sty_dim)

        updated_frame_mask = (frame_indices.squeeze(1).expand(en.shape[0], -1) >= updated_seq_lengths.unsqueeze(1)).to(self.device)
        updated_frame_mask = (~updated_frame_mask).float()
        F0_pred, N_pred, _ = self.predictor.F0Ntrain(en, s, updated_seq_lengths, updated_frame_mask) # b x 1 x 2*max_dur, b x 1 x 2*max_dur
        t_en = self.text_encoder(input_ids, input_lengths, attention_mask) # b x seq_len x d_model
        asr = torch.bmm(pred_aln_trg, t_en) * updated_frame_mask.unsqueeze(-1) # b x max_dur x d_model
        audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :, :128], updated_frame_mask) # b x T
        return audio#, pred_dur

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
