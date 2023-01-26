# Code for the fusion modules of HVV

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from ot.da import (
    LinearTransport,
    EMDTransport,
    EMDLaplaceTransport,
    JCPOTTransport,
    UnbalancedSinkhornTransport
)
from otk.layers import OTKernel

from .attention_module import *
from .vision_module import *
from .kg_module import *
from config.config import *


# ------------------------------------------ Concat Fusion Module ------------------------------------------ #

class ConcatFusionModule(nn.Module):
    
    def __init__(
        self,
        model_dim: int,
        use_attention: bool
    ):
        super(ConcatFusionModule, self).__init__()
        
        print("\n\nUsing ConcatFusionModule..., use_attention: {}\n\n".format(use_attention))
        extra_dim = model_dim
 
        if USE_IMAGE:
            extra_dim = extra_dim + LM_DIM
            
        if USE_KG:
            extra_dim = extra_dim + LM_DIM
        
        print("\nextra_dim: " ,extra_dim, "\n")
        self.transform = nn.Linear(extra_dim, model_dim)    
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.use_attention = use_attention
        if use_attention:
            self.attention_layer = MultiHeadAttention(d_model=model_dim, num_heads=8)
        
        
    def forward(
        self,
        lm_input: torch.Tensor,
        image_output: Optional[torch.Tensor] = None,
        kg_output: Optional[torch.Tensor] = None,
    ):   
        output = lm_input
        
        if USE_IMAGE and image_output is not None:
            output = torch.cat([output, image_output], dim=-1)
        
        if USE_KG and kg_output is not None:
            output = torch.cat([output, kg_output], dim=-1)
                
        output = self.dropout(output)
        output = F.relu(self.transform(output))
        if self.use_attention:
            output, _ = self.attention_layer(query=output, key=output, value=output)
        output = self.layer_norm(output)
        
        return output
 