# Code for the Knowledge graph module of HVV

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dgl_nn

from config.config import *
from .attention_module import *
from otk.layers import OTKernel
    
# ------------------------------------------ Graph Module ------------------------------------------ #
    
class OTKKnowledgeGraphModule(nn.Module):
    def __init__(
        self, 
        model_dim: int
    ):
        super(OTKKnowledgeGraphModule, self).__init__()
        
        print("\n\nUsing OTKKnowledgeGraphModule...\n\n")
        
        self.conv1 = dgl_nn.GraphConv(768, (768+KG_DIM)//2)
        self.conv2 = dgl_nn.GraphConv((768+KG_DIM)//2, KG_DIM)

        self.pooling = dgl_nn.MaxPooling()
        self.graph_len_transform = nn.Linear(GRAPH_SEQ_DIM, LM_MAX_LEN)
        self.graph_attention_layer = ScaledDotProductAttention(dim=KG_DIM)
        
        self.lm_transform = nn.Linear(model_dim, LM_DIM)
        self.lm_dropout = nn.Dropout(DROPOUT_RATE)
        self.lm_attention_layer = ScaledDotProductAttention(dim=LM_DIM)
        
        self.otk_layer = nn.Sequential(
            nn.Linear(LM_DIM+KG_DIM, LM_DIM),
            nn.ReLU(),
            OTKernel(in_dim=LM_DIM, out_size=LM_MAX_LEN, heads=1)
        )

        self.layer_norm = nn.LayerNorm(LM_DIM)

        
        
                
    def forward(
        self, 
        graph, 
        node_feat,
        graph_edge_weight,
        lm_input
    ):
        
        # 1. Obtain node embeddings 
        graph_output = self.conv1(graph, node_feat, edge_weight=graph_edge_weight)
        graph_output = self.conv2(graph, graph_output, edge_weight=graph_edge_weight)
        
        # 2. Apply pooling
        graph_output = self.pooling(graph, graph_output)
        
        # 3. Convert to 3D tensor
        graph_output = graph_output.unsqueeze(1).repeat(1, GRAPH_SEQ_DIM, 1)
        graph_output = graph_output.permute(0, 2, 1)
        graph_output = F.relu(self.graph_len_transform(graph_output))
        graph_output = graph_output.permute(0, 2, 1)
        
        graph_output, _ = self.graph_attention_layer(query=graph_output, 
                                                     key=graph_output, 
                                                     value=graph_output)
        
        # 4. Fuse with LM output
        lm_input = self.lm_dropout(lm_input)
        lm_input = F.relu(self.lm_transform(lm_input))
        lm_input, _ = self.lm_attention_layer(query=lm_input, 
                                              key=lm_input, 
                                              value=lm_input)
        
        output = self.otk_layer(torch.cat([lm_input, graph_output], dim=-1))
        output = torch.nan_to_num(output)
        output = self.layer_norm(output)
        return output
