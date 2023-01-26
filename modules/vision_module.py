# Code for the vision module of HVV

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from config.config import *
from .attention_module import *
from otk.layers import OTKernel
    
# ------------------------------------------ Vision Module ------------------------------------------ #

class OTKVisionModule(nn.Module):
    
    def __init__(
        self,
        model_dim: int
    ):
        super(OTKVisionModule, self).__init__() 
        
        print("\n\nUsing OTKVisionModule...\n\n")
                
        if VISION_ENCODER == 'VIT':
            print("\nUsing {}: {}\n".format(VISION_ENCODER, VISION_ENCODER_DICT[VISION_ENCODER]))
            self.vision_encoder = AutoModel.from_pretrained(VISION_ENCODER_DICT[VISION_ENCODER])
            self.vision_len_transform = nn.Linear(197, LM_MAX_LEN)
            
        elif VISION_ENCODER == 'SWIN':
            print("\nUsing {}: {}\n".format(VISION_ENCODER, VISION_ENCODER_DICT[VISION_ENCODER]))
            self.vision_encoder = AutoModel.from_pretrained(VISION_ENCODER_DICT[VISION_ENCODER])
            self.vision_len_transform = nn.Linear(49, LM_MAX_LEN)
        
        elif VISION_ENCODER == 'BEiT':
            print("\nUsing {}: {}\n".format(VISION_ENCODER, VISION_ENCODER_DICT[VISION_ENCODER]))
            self.vision_encoder = AutoModel.from_pretrained(VISION_ENCODER_DICT[VISION_ENCODER])
            self.vision_len_transform = nn.Linear(197, LM_MAX_LEN)
            
        elif VISION_ENCODER == 'ImageGPT':
            print("\nUsing {}: {}\n".format(VISION_ENCODER, VISION_ENCODER_DICT[VISION_ENCODER]))
            self.vision_encoder = AutoModel.from_pretrained(VISION_ENCODER_DICT[VISION_ENCODER])
            self.vision_len_transform = nn.Linear(1024, LM_MAX_LEN)
            
        self.vision_transform = nn.Linear(self.vision_encoder.config.hidden_size, VISION_DIM)
        self.vision_attention_layer = ScaledDotProductAttention(dim=VISION_DIM)
        
        self.lm_transform = nn.Linear(model_dim, LM_DIM)
        self.lm_dropout = nn.Dropout(DROPOUT_RATE)
        self.lm_attention_layer = ScaledDotProductAttention(dim=LM_DIM)

        self.otk_layer = nn.Sequential(
            nn.Linear(LM_DIM+VISION_DIM, LM_DIM),
            nn.ReLU(),
            OTKernel(in_dim=LM_DIM, out_size=LM_MAX_LEN, heads=1)
        )
        self.layer_norm = nn.LayerNorm(LM_DIM)
        
        
             
    def forward(
        self,
        image_inputs,
        lm_input,
        caption_input: Optional[torch.Tensor] = None
    ):
        image_output = self.vision_encoder(**image_inputs)['last_hidden_state']
        
        image_output = image_output.permute(0, 2, 1)
        image_output = F.relu(self.vision_len_transform(image_output))
        image_output = image_output.permute(0, 2, 1)
        image_output = F.relu(self.vision_transform(image_output))
        image_output, _ = self.vision_attention_layer(query=image_output, 
                                                      key=image_output, 
                                                      value=image_output)
        
        lm_input = self.lm_dropout(lm_input)
        lm_input = F.relu(self.lm_transform(lm_input))
        lm_input, _ = self.lm_attention_layer(query=lm_input, 
                                              key=lm_input, 
                                              value=lm_input)
        
        output = self.otk_layer(torch.cat([lm_input, image_output], dim=-1))
        output = torch.nan_to_num(output)
        output = self.layer_norm(output)
        return output
