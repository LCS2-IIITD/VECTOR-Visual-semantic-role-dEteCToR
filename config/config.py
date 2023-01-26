# Config file for HVV

import os
import numpy as np
import random
import torch
import dgl


def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999
    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    os.environ["OMP_NUM_THREADS"] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)

SEED = 4
set_random_seed(SEED)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
    
    

if torch.cuda.is_available():
#     DEVICE = torch.device("cuda:0")
    DEVICE = torch.device("cuda:1")
    print("Using GPU: ", DEVICE)
else:
    DEVICE = torch.device("cpu")
    print("Using CPU: ", DEVICE)

    
    
LABEL2ID = {
    'hero' : 3, 
    'villain' : 2, 
    'victim' : 1, 
    'other' : 0
}


TARGET_NAMES = ['other', 'victim', 'villain', 'hero']  



MODEL_PATH = 'Hero-Villain-Victim/models/HVV/'
RESULT_PATH = 'Hero-Villain-Victim/results/HVV/'
TEXT_RESULT_PATH = 'Hero-Villain-Victim/results/text-baselines/'
VISION_RESULT_PATH = 'Hero-Villain-Victim/results/vision-baselines/'


DATASET_TYPE = 'LOGICALLY'
# DATASET_TYPE = 'CONSTRAINT'

DATA_DIR = 'Hero-Villain-Victim/dataset/images/'

if DATASET_TYPE == 'LOGICALLY':
    TRAIN_COVID = 'Hero-Villain-Victim/dataset/covid/train_final.jsonl'
    TRAIN_POLITICS = 'Hero-Villain-Victim/dataset/uspolitics/train_final.jsonl'

    VAL_COVID = 'Hero-Villain-Victim/dataset/covid/val_final.jsonl'
    VAL_POLITICS = 'Hero-Villain-Victim/dataset/uspolitics/val_final.jsonl'

    TEST = 'Hero-Villain-Victim/dataset/test/test_final.jsonl'

elif DATASET_TYPE == 'LOGICALLY':
    TRAIN_COVID = 'Hero-Villain-Victim/dataset/covid/train_covid.json'
    TRAIN_POLITICS = 'Hero-Villain-Victim/dataset/uspolitics/train_pol.json'

    VAL_COVID = 'Hero-Villain-Victim/dataset/covid/val_covid.json'
    VAL_POLITICS = 'Hero-Villain-Victim/dataset/uspolitics/val_pol.json'

    TEST_COVID = 'Hero-Villain-Victim/dataset/covid/test_covid.json'
    TEST_POLITICS = 'Hero-Villain-Victim/dataset/uspolitics/test_pol.json'

EMBEDDING_PATH = 'Hero-Villain-Victim/dataset/additional-features/embeddings/'
KG_PATH = 'Hero-Villain-Victim/dataset/additional-features/KG-data/'


SAVE_RESULTS = True
USE_COVID = True
USE_POLITICS = True

LM_MAX_LEN = 64
MAX_EPOCHS = 15
BATCH_SIZE = 8

DROPOUT_RATE = 0.2
LEARNING_RATE = 1e-5
ACCUMULATION_STEPS = 2
UPSAMPLES = 2000

# FUSION_LAYER = 11
FUSION_LAYER = 23
FUSION_TYPE = '2d_concat'
# FUSION_TYPE = '3d_concat'
# FUSION_TYPE = 'OT'

USE_FACE = True
USE_CAPTION_EARLY = False
USE_CONTEXT_EARLY = False
USE_EVIDENCE_EARLY = False

USE_CAPTION_EMBEDDINGS = False
USE_CONTEXT_EMBEDDINGS = True
USE_EVIDENCE_EMBEDDINGS = False
USE_KG = True
USE_IMAGE = True
FREEZE_VISION_ENCODER = False
GRAPH_SEQ_DIM = 8

IMAGE_INTERACTION_TYPE = 'attention'
# IMAGE_INTERACTION_TYPE = 'otke'

KG_INTERACTION_TYPE = 'attention'
# KG_INTERACTION_TYPE = 'otke'

CAPTION_INTERACTION_TYPE = 'attention'
# CAPTION_INTERACTION_TYPE = 'otke'

CONTEXT_INTERACTION_TYPE = 'attention'
# CONTEXT_INTERACTION_TYPE = 'otke'

EVIDENCE_INTERACTION_TYPE = 'attention'
# EVIDENCE_INTERACTION_TYPE = 'otke'

LM_DIM = 512
VISION_DIM = 256
KG_DIM = VISION_DIM
CAPTION_DIM = VISION_DIM
CONTEXT_DIM = VISION_DIM
EVIDENCE_DIM = VISION_DIM
USE_ATTENTION_FUSION = True

# VISION_ENCODER = 'CLIP'
# VISION_ENCODER = 'RESNET'
# VISION_ENCODER = 'CONVXNET'
VISION_ENCODER = 'VIT'
# VISION_ENCODER = 'SWIN'
# VISION_ENCODER = 'BEiT'
# VISION_ENCODER = 'ImageGPT'

VISION_ENCODER_DICT = {
    'CLIP': 'openai/clip-vit-base-patch32',
    'VIT': 'google/vit-base-patch16-224-in21k',
    'SWIN': 'microsoft/swin-base-patch4-window7-224-in22k',
    'RESNET': 'microsoft/resnet-50',
    'CONVXNET': 'facebook/convnext-base-224',
    'BEiT': 'microsoft/beit-base-patch16-224-pt22k-ft22k',
    'ImageGPT': 'openai/imagegpt-small'
}

print("\nDATASET_TYPE: ", DATASET_TYPE)
print("TRAIN_COVID: ", TRAIN_COVID)
print("TRAIN_POLITICS: ", TRAIN_POLITICS)

print("\nLM_MAX_LEN: ", LM_MAX_LEN)
print("MAX_EPOCHS: ", MAX_EPOCHS)
print("BATCH_SIZE: ", BATCH_SIZE)
print("DROPOUT_RATE: ", DROPOUT_RATE)
print("LEARNING_RATE: ", LEARNING_RATE)
print("ACCUMULATION_STEPS: ", ACCUMULATION_STEPS)
print("UPSAMPLES: ", UPSAMPLES)
print("FUSION_LAYER: ", FUSION_LAYER)
print("FUSION_TYPE: ", FUSION_TYPE)
print("IMAGE_INTERACTION_TYPE: ", IMAGE_INTERACTION_TYPE)
print("KG_INTERACTION_TYPE: ", KG_INTERACTION_TYPE)
print("EVIDENCE_INTERACTION_TYPE: ", EVIDENCE_INTERACTION_TYPE)
print("CAPTION_INTERACTION_TYPE: ", CAPTION_INTERACTION_TYPE)
print("CONTEXT_INTERACTION_TYPE: ", CONTEXT_INTERACTION_TYPE)
print("USE_FACE: ", USE_FACE)
print("USE_CAPTION_EARLY: ", USE_CAPTION_EARLY)
print("USE_CONTEXT_EARLY: ", USE_CONTEXT_EARLY)
print("USE_EVIDENCE_EARLY: ", USE_EVIDENCE_EARLY)
print("USE_CAPTION_EMBEDDINGS: ", USE_CAPTION_EMBEDDINGS)
print("USE_CONTEXT_EMBEDDINGS: ", USE_CONTEXT_EMBEDDINGS)
print("USE_EVIDENCE_EMBEDDINGS: ", USE_EVIDENCE_EMBEDDINGS)
print("USE_KG: ", USE_KG)
print("USE_IMAGE: ", USE_IMAGE)
print("FREEZE_VISION_ENCODER: ", FREEZE_VISION_ENCODER)
print("GRAPH_SEQ_DIM: ", GRAPH_SEQ_DIM)
print("LM_DIM: ", LM_DIM)
print("VISION_DIM: ", VISION_DIM)
print("KG_DIM: ", KG_DIM)
print("EVIDENCE_DIM: ", EVIDENCE_DIM)
print("CAPTION_DIM: ", CAPTION_DIM)
print("CONTEXT_DIM: ", CONTEXT_DIM)
print("VISION_ENCODER: ", VISION_ENCODER)
print("USE_ATTENTION_FUSION: ", USE_ATTENTION_FUSION)
print("SAVE_RESULTS: ", SAVE_RESULTS)
print("USE_COVID: ", USE_COVID)
print("USE_POLITICS: ", USE_POLITICS)