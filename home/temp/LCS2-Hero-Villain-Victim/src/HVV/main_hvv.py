# Code for LM baselines to classify HVV
# File: main_hvv.py
            
import argparse
import pandas as pd
import numpy as np
import warnings
import time
from datetime import datetime
import gzip
import pickle
import json
import torch
from torch.utils.data import (
    DataLoader, 
    WeightedRandomSampler
)

import transformers
from transformers import (
    AutoTokenizer,
    AutoFeatureExtractor
)

import dgl
from dgl.dataloading import GraphDataLoader

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()    

print('Pytorch and CUDA Version: ', torch.__version__)
print('DGL Version: ', dgl.__version__)
print('Transformers Version: ', transformers.__version__)

from config.config import *

from utils.data_utils import (
    get_total_dataset_logically, 
    get_total_dataset_constraint,
    get_graph_data,
    HVVCustomDataset
)

from utils.training_utils import train_hvv

from modeling_transformers.modeling_deberta import (
    HVVDebertaClassifier
)

# ------------------------------------------------------------ MAIN FUNCTION ------------------------------------------------------------ #

if __name__ == "__main__":
    
    print("\n\nProgram PID: ", os.getpid(), "\n\n")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type",type=str, default='large')
    parser.add_argument("--data_augmentation",type=str, default='no')
    parser.add_argument("--weighted_sampling",type=str, default='no')
    args = parser.parse_args()
    
    if args.data_augmentation == 'yes':
        print("\nDATA_AUGMENTATION\n")
        DATA_AUGMENTATION = True
    else:
        DATA_AUGMENTATION = False
        
    if USE_COVID and USE_POLITICS:
        dataset_type = 'combined'
        
    elif USE_COVID:
        dataset_type = 'covid'
        
    elif USE_POLITICS:
        dataset_type = 'uspolitics'
        
    print("\ndataset_type: {}\n".format(dataset_type))
    
    # ---------------------------------------------- MODEL INITIALIZATION ---------------------------------------------- #
    
    if args.model_type == 'small':
        MODEL_NAME = 'microsoft/deberta-v3-small'
        modelparams = 141304320
    elif args.model_type == 'base':
        MODEL_NAME = 'microsoft/deberta-v3-base'
        modelparams = 183831552
    else:
        MODEL_NAME = 'microsoft/deberta-v3-large'
        modelparams = 434012160
    print("\n\nLoading Model: ", MODEL_NAME)
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, additional_special_tokens=["[MEME TEXT]", "[MEME DESCRIPTION]", "[MEME CONTEXT]", "[MEME ENTITY]", "[MEME FACE]"])
    print("Tokenizer loaded...\n")
    MODEL = HVVDebertaClassifier.from_pretrained(MODEL_NAME, num_labels=4)
    print("Model loaded...\n")
    CAPTION_FILE = EMBEDDING_PATH + 'caption_DeBERTa_ST_embeddings.gz'
    CONTEXT_FILE = EMBEDDING_PATH + 'context_DeBERTa_ST_embeddings.gz'
    KNOWLEDGE_PATH = [
        'GPT3_prompts_train_DeBERTa_ST_embeddings.gz',
        'GPT3_prompts_val_DeBERTa_ST_embeddings.gz',
        'GPT3_prompts_test_DeBERTa_ST_embeddings.gz'
    ]
    KG_EMBEDDING_PATH = KG_PATH + 'final_nodes_dict_DeBERTa_mean.gz'
        
    MODEL.resize_token_embeddings(len(TOKENIZER))    
    MODEL.to(DEVICE)
    
    if FREEZE_VISION_ENCODER:
        for name, param in MODEL.named_parameters():
            if 'vision_encoder.embeddings' in name:
                param.requires_grad = False
            if 'vision_encoder.encoder' in name:
                if 'vision_encoder.encoder.layer.11' not in name:
                    param.requires_grad = False

                    
    pytorch_total_params = sum(p.numel() for p in MODEL.parameters())
    print("\n\nTotal parameters: ", pytorch_total_params)

    pytorch_total_train_params = sum(p.numel() for p in MODEL.parameters() if p.requires_grad)
    print("Total trainable parameters: {}\n\n".format(pytorch_total_train_params))
    
    extra = pytorch_total_train_params-modelparams
    print("Extra trainable parameters added:", extra)
    print("Percentage parameter change: ", extra*100/modelparams)
    
    print("\nTrainable Parameters: \n")
    for name, param in MODEL.named_parameters():
        if param.requires_grad:
            print(name) 
           
    # ---------------------------------------------- VISION TOKENIZER ---------------------------------------------- #
    

    print("Using : ", VISION_ENCODER)
    VISION_TOKENIZER = AutoFeatureExtractor.from_pretrained(VISION_ENCODER_DICT[VISION_ENCODER])
    print(VISION_ENCODER, " Tokenizer loaded...\n") 
    
    
    # ----------------------------------------- Helper Data Loading -----------------------------------------
    
    with open(FACE_DATA_PATH, 'r') as file:
        FACE_DATA = json.load(file)
    file.close()
    print("\nFace data size: ", len(FACE_DATA))
    
    if USE_KG:
        print("\nUSE_KG\n")
        start = datetime.now()
        with gzip.open(KG_PATH + 'final_nodes_dict_inverse.gz', "rb") as f:
            INVERSE_NODE2ID = pickle.load(f)
        f.close()
        print("INVERSE_NODE2ID size: ", len(INVERSE_NODE2ID))

        with gzip.open(KG_PATH + 'final_nodes_dict_DeBERTa_mean.gz', 'rb') as file:
            NODE_EMBEDDINGS = pickle.load(file)
        file.close()
        print("\n\nTimeline embedding size: ", len(NODE_EMBEDDINGS))
    
        TRAIN_DGL_GRAPH_DATA_DICT = get_graph_data(
            graph_path=KG_PATH+'knowledge_graph_train_weight_dict.gz',
            graph_embedding=NODE_EMBEDDINGS,  
            node_mapping=INVERSE_NODE2ID,
        )
        print("\nTRAIN_DGL_GRAPH_DATA_DICT size", len(TRAIN_DGL_GRAPH_DATA_DICT))
        
        VAL_DGL_GRAPH_DATA_DICT = get_graph_data(
            graph_path=KG_PATH+'knowledge_graph_val_weight_dict.gz',
            graph_embedding=NODE_EMBEDDINGS,  
            node_mapping=INVERSE_NODE2ID,
        )
        print("\nVAL_DGL_GRAPH_DATA_DICT size", len(VAL_DGL_GRAPH_DATA_DICT))
        
        TEST_DGL_GRAPH_DATA_DICT = get_graph_data(
            graph_path=KG_PATH+'knowledge_graph_test_weight_dict.gz',
            graph_embedding=NODE_EMBEDDINGS,  
            node_mapping=INVERSE_NODE2ID,
        )
        end = datetime.now()
        print("\nTEST_DGL_GRAPH_DATA_DICT size", len(TEST_DGL_GRAPH_DATA_DICT))
        print('Time taken for graph loading: ', end-start)
        
    else:
        TRAIN_DGL_GRAPH_DATA_DICT = None
        VAL_DGL_GRAPH_DATA_DICT = None
        TEST_DGL_GRAPH_DATA_DICT = None
    
    
    # ----------------------------------------- Data Loading -----------------------------------------
    
    if DATASET_TYPE == 'LOGICALLY':
        train_data, val_data, test_data = get_total_dataset_logically(face_data=False, augmentation=args.data_augmentation)
        print("\ntrain_data shape: ", train_data.shape)
        print("val_data: ", val_data.shape)
        print("test_data: ", test_data.shape)
        
    elif DATASET_TYPE == 'CONSTRAINT':
        train_data, val_data, test_data = get_total_dataset_constraint(face_data=False, augmentation=args.data_augmentation)
        print("\ntrain_data shape: ", train_data.shape)
        print("val_data: ", val_data.shape)
        print("test_data: ", test_data.shape)
    
    train_dataset = HVVCustomDataset(
        df=train_data, 
        data_type='train', 
        lm_tokenizer=TOKENIZER,
        vision_tokenizer=VISION_TOKENIZER,
        face_data=FACE_DATA,
        kg_data=TRAIN_DGL_GRAPH_DATA_DICT
    )
    if USE_KG:
        train_dataloader = GraphDataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        
    print("\n\nTrain data loaded, length:", len(train_dataset))
    
    
    val_dataset = HVVCustomDataset(
        df=val_data,
        data_type='val', 
        face_data=FACE_DATA,
        lm_tokenizer=TOKENIZER,
        vision_tokenizer=VISION_TOKENIZER,
        kg_data=VAL_DGL_GRAPH_DATA_DICT
    )
    if USE_KG:
        val_dataloader = GraphDataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )
        
    else:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )
    print("\n\nVal data loaded, length:", len(val_dataset))


    test_dataset = HVVCustomDataset(
        df=test_data,
        data_type='test', 
        lm_tokenizer=TOKENIZER, 
        vision_tokenizer=VISION_TOKENIZER,
        face_data=FACE_DATA,
        kg_data=TEST_DGL_GRAPH_DATA_DICT
    )
    if USE_KG:
        test_dataloader = GraphDataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )
    else:
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )
    
    
    print("\n\nTest data loaded, length:", len(test_dataset))
    

    # ------------------------------ TRAINING SETUP ------------------------------ #
   
    start = datetime.now()
    print("\n\nTraining started started at ", start)
    
    model_name = 'DEBERTA_' + '_' + args.model_type + '_' + VISION_ENCODER + '_aug'
            
    model_info = {
        'model_name': model_name,
        'timestamp': start,
        'dataset_type': dataset_type
    }
    
    
    train_hvv(
        model_info=model_info,
        model=MODEL,
        train_data_loader=train_dataloader,
        val_data_loader=val_dataloader,
        test_data_loader=test_dataloader,
        learning_rate=LEARNING_RATE
    )
    
    end = datetime.now()
    print("\n\nTraining ended ended at ", end)
    print("\n\nTotal time taken ", end-start)