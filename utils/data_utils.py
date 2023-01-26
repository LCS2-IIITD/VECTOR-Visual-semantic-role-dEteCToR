# Code for data utils for HVV

import numpy as np
import pandas as pd
import json
import gzip
import pickle
from tqdm import tqdm
from sklearn.utils import resample
from PIL import Image
import networkx as nx
import dgl 
import gc
from torch.utils.data import Dataset
from torchvision.transforms import (
    Compose, 
    Resize, 
    CenterCrop, 
    ToTensor,
    Normalize
)

from config.config import *

# -------------------------------------------------------------- DATA UTILS -------------------------------------------------------------- 

def expand_entity_data(json_data):
    original = []
    text = []
    image_name = []
    word = []
    role = []
    meme_context = []
    
    for vals in tqdm(json_data):
        
        ocr = vals['OCR']
        if ocr is not None:
            ocr = ocr.lower().replace('\n', ' ')
        else:
            ocr = None
            
        context = vals['Context']
        if context is not None:
            context = context.lower().replace('\n', ' ')
        else:
            context = None
        
        image = vals['image'].lower()
        
        for keys in ['hero', 'villain', 'victim', 'other']:
            for word_val in vals[keys]:
                original.append(vals['OCR'])
                text.append(ocr)
                word.append(word_val)
                role.append(keys)
                image_name.append(image)
                meme_context.append(context)

    df = pd.DataFrame()

    df['sentence'] = text 
    df['original'] = original 
    df['image'] = image_name
    df['entity'] = word 
    df['entity_type'] = role
    df['label'] = [LABEL2ID[r] for r in role]
    df['context'] = meme_context

    return df



def get_total_dataset_constraint(face_data=False, augmentation='no'):

    print("\nLoading CONSTRAINT Dataset...\n")
    
    # ----------------------------------------------- READ TRAIN DATA -----------------------------------------------
    
    json_data_train = []
    
    if USE_COVID:
        with open(TRAIN_COVID, 'r') as json_file:
            for json_str in list(json_file):
                json_data_train.append(json.loads(json_str))
        json_file.close()
    
    if USE_POLITICS:
        with open(TRAIN_POLITICS, 'r') as json_file:
            for json_str in list(json_file):
                json_data_train.append(json.loads(json_str))
        json_file.close()
    
    df_train = expand_entity_data(json_data_train)
    

    # Up sampling of the minority classes  
    if augmentation == 'yes':
        
        df_other = df_train[df_train.entity_type=='other']
        df_hero = df_train[df_train.entity_type=='hero']
        df_villian = df_train[df_train.entity_type=='villain']
        df_victim = df_train[df_train.entity_type=='victim']
    
        df_hero_upsampled = resample(df_hero, 
                                     replace=True,     # sample with replacement
                                     n_samples=UPSAMPLES,    # to match majority class
#                                      random_state=4) # reproducible results
                                     random_state=42) # reproducible results

        df_other = pd.concat([df_other, df_hero])
        df_other = pd.concat([df_other, df_hero_upsampled])

        df_villian_upsampled = resample(df_villian, 
                                        replace=True,     # sample with replacement
                                        n_samples=UPSAMPLES,    # to match majority class
#                                         random_state=4) # reproducible results
                                        random_state=42) # reproducible results

        df_other = pd.concat([df_other, df_villian])
        df_other = pd.concat([df_other, df_villian_upsampled])

        df_victim_upsampled = resample(df_victim, 
                                        replace=True,     # sample with replacement
                                        n_samples=UPSAMPLES,    # to match majority class
#                                        random_state=4) # reproducible results
                                        random_state=42) # reproducible results

        df_train_final = pd.concat([df_other, df_victim])
        df_train_final = pd.concat([df_train_final, df_victim_upsampled])
#         df_train_final = df_train_final.reset_index(drop_index=False)
        
    else:
        df_train_final = df_train
    
    df_train_final['ID'] = np.arange(len(df_train_final))
    df_train_final['ID'] = df_train_final['ID'].apply(lambda x: 'train_' + str(x))
    
    # ----------------------------------------------- READ VAL DATA -----------------------------------------------
    
    json_data_val = []

    if USE_COVID:
        with open(VAL_COVID, 'r') as json_file:
            for json_str in list(json_file):
                json_data_val.append(json.loads(json_str))
        json_file.close()
    
    if USE_POLITICS:
        with open(VAL_POLITICS, 'r') as json_file:
            for json_str in list(json_file):
                json_data_val.append(json.loads(json_str))
        json_file.close()
    
    df_val = expand_entity_data(json_data_val)
    
    df_val['ID'] = np.arange(len(df_val))
    df_val['ID'] = df_val['ID'].apply(lambda x: 'val_' + str(x))
    
    # ----------------------------------------------- READ TEST DATA -----------------------------------------------
    
    json_data_test = []
    
    if USE_COVID:
        with open(TEST_COVID, 'r') as json_file:
            for json_str in list(json_file):
                json_data_test.append(json.loads(json_str))
        json_file.close()
    
    if USE_POLITICS:
        with open(TEST_POLITICS, 'r') as json_file:
            for json_str in list(json_file):
                json_data_test.append(json.loads(json_str))
        json_file.close()
    
    df_test = expand_entity_data(json_data_test)
    
    df_test['ID'] = np.arange(len(df_test))
    df_test['ID'] = df_test['ID'].apply(lambda x: 'test_' + str(x))
    
    return df_train_final, df_val, df_test




def get_total_dataset_logically(face_data=False, augmentation='no'):
    
    print("\nLoading LOGICALLY Dataset...\n")
    
    # ----------------------------------------------- READ TRAIN DATA -----------------------------------------------
    
    json_data_train = []
    
    if USE_COVID:
        with open(TRAIN_COVID, 'r') as json_file:
            for json_str in list(json_file):
                json_data_train.append(json.loads(json_str))
        json_file.close()
    
    if USE_POLITICS:
        with open(TRAIN_POLITICS, 'r') as json_file:
            for json_str in list(json_file):
                json_data_train.append(json.loads(json_str))
        json_file.close()
    
    original = []
    text = []
    image_name = []
    word = []
    role = []
    
    for vals in tqdm(json_data_train):
        sentence = vals['OCR'].lower().replace('\n', ' ')
        image = vals['image'].lower()
        for keys in ['hero', 'villain', 'victim', 'other']:
            for word_val in vals[keys]:
                original.append(vals['OCR'])
                if face_data:
                    text.append(sentence + '\n' + ' '.join(train_face_dict[vals['image']]))
                else:
                    text.append(sentence)
                word.append(word_val)
                role.append(keys)
                image_name.append(image)

    df_train = pd.DataFrame()

    df_train['sentence'] = text 
    df_train['original'] = original 
    df_train['image'] = image_name
    df_train['entity'] = word 
    df_train['entity_type'] = role
    df_train['label'] = [LABEL2ID[r] for r in role]
    
    df_train['ID'] = np.arange(len(df_train))
    df_train['ID'] = df_train['ID'].apply(lambda x: 'train_' + str(x))
    

    # Up sampling of the minority classes  
    
    if augmentation == 'yes':
        
        df_other = df_train[df_train.entity_type=='other']
        df_hero = df_train[df_train.entity_type=='hero']
        df_villian = df_train[df_train.entity_type=='villain']
        df_victim = df_train[df_train.entity_type=='victim']
    
        df_hero_upsampled = resample(df_hero, 
                                     replace=True,     # sample with replacement
                                     n_samples=UPSAMPLES,    # to match majority class
                                     random_state=42) # reproducible results

        df_other = pd.concat([df_other, df_hero])
        df_other = pd.concat([df_other, df_hero_upsampled])

        df_villian_upsampled = resample(df_villian, 
                                        replace=True,     # sample with replacement
                                        n_samples=UPSAMPLES,    # to match majority class
                                        random_state=42) # reproducible results

        df_other = pd.concat([df_other, df_villian])
        df_other = pd.concat([df_other, df_villian_upsampled])

        df_victim_upsampled = resample(df_victim, 
                                        replace=True,     # sample with replacement
                                        n_samples=UPSAMPLES,    # to match majority class
                                        random_state=42) # reproducible results

        df_train_final = pd.concat([df_other, df_victim])
        df_train_final = pd.concat([df_train_final, df_victim_upsampled])
        
    else:
        df_train_final = df_train

    
    # ----------------------------------------------- READ VAL DATA -----------------------------------------------
    
    json_data_val = []

    if USE_COVID:
        with open(VAL_COVID, 'r') as json_file:
            for json_str in list(json_file):
                json_data_val.append(json.loads(json_str))
        json_file.close()
    
    if USE_POLITICS:
        with open(VAL_POLITICS, 'r') as json_file:
            for json_str in list(json_file):
                json_data_val.append(json.loads(json_str))
        json_file.close()
    
    original = []
    text = []
    image_name = []
    word = []
    role = []

    for vals in tqdm(json_data_val):
        sentence = vals['OCR'].lower().replace('\n', ' ')
        image = vals['image'].lower()
        for keys in ['hero', 'villain', 'victim', 'other']:
            for word_val in vals[keys]:
                original.append(vals['OCR'])
                if face_data:
                    text.append(sentence + '\n' + ' '.join(val_face_dict[vals['image']]))
                else:
                    text.append(sentence)
                word.append(word_val)
                role.append(keys)
                image_name.append(image)

    df_val = pd.DataFrame()

    df_val['sentence'] = text 
    df_val['original'] = original 
    df_val['image'] = image_name
    df_val['entity'] = word 
    df_val['entity_type'] = role
    df_val['label'] = [LABEL2ID[r] for r in role]

    df_val['ID'] = np.arange(len(df_val))
    df_val['ID'] = df_val['ID'].apply(lambda x: 'val_' + str(x))
    
    
    # ----------------------------------------------- READ TEST DATA -----------------------------------------------
    
    json_data_test = []
    
    with open(TEST, 'r') as json_file:
        for json_str in list(json_file):
            json_data_test.append(json.loads(json_str))
    json_file.close()
    
    original = []
    text = []
    image_name = []
    word = []
    role = []

    for vals in tqdm(json_data_test):
        sentence = vals['OCR'].lower().replace('\n', ' ')
        image = vals['image'].lower()
        for keys in ['hero', 'villain', 'victim', 'other']:
            for word_val in vals[keys]:
                original.append(vals['OCR'])
                if face_data:
                    text.append(sentence + '\n' + ' '.join(val_face_dict[vals['image']]))
                else:
                    text.append(sentence)
                word.append(word_val)
                role.append(keys)
                image_name.append(image)

    df_test = pd.DataFrame()

    df_test['sentence'] = text 
    df_test['original'] = original 
    df_test['image'] = image_name
    df_test['entity'] = word 
    df_test['entity_type'] = role
    df_test['label'] = [LABEL2ID[r] for r in role]
    
    df_test['ID'] = np.arange(len(df_test))
    df_test['ID'] = df_test['ID'].apply(lambda x: 'test_' + str(x))
    
    return df_train_final, df_val, df_test





def load_faces_data():
    train_face_dict = {}
    val_face_dict = {}
    test_face_dict = {}
    
    json_data_train = []

    with open(FACE_PATH_COVID_TRAIN, 'r') as json_file:
        for json_str in list(json_file):
            json_data_train.append(json.loads(json_str))
    json_file.close()
    
    with open(FACE_PATH_POLITICS_TRAIN, 'r') as json_file:
        for json_str in list(json_file):
            json_data_train.append(json.loads(json_str))
    json_file.close()
    
    for trains in json_data_train:
        face = trains['faces']
        if len(face) != 0:
            face = face[0].split('[')[0].replace('_', ' ').strip()
        train_face_dict[trains['image']] = face


    json_data_val = []

    with open(FACE_PATH_COVID_VAL, 'r') as json_file:
        for json_str in list(json_file):
            json_data_val.append(json.loads(json_str))
    json_file.close()
    
    with open(FACE_PATH_POLITICS_VAL, 'r') as json_file:
        for json_str in list(json_file):
            json_data_val.append(json.loads(json_str))
    json_file.close()
    
    for vals in json_data_val:
        face = vals['faces']
        if len(face) != 0:
            face = face[0].split('[')[0].replace('_', ' ').strip()
        val_face_dict[vals['image']] = face
        
    json_data_test = []

    with open(FACE_PATH_TEST, 'r') as json_file:
        for json_str in list(json_file):
            json_data_test.append(json.loads(json_str))
    json_file.close()
    
    for tests in json_data_test:
        face = tests['faces']
        if len(face) != 0:
            face = face[0].split('[')[0].replace('_', ' ').strip()
        test_face_dict[tests['image']] = face

    return train_face_dict, val_face_dict, test_face_dict





def get_graph_data(
    graph_path,
    graph_embedding,  
    node_mapping
):
    # ------------------------------------- GRAPH EMBEDDINGS -------------------------------------
    
    with gzip.open(graph_path, 'rb') as file:
        graph_data_dict = pickle.load(file)
    file.close()
    print("\ngraph_data_dict size", len(graph_data_dict))

    dgl_graph_data_dict = dict()
    for uid in graph_data_dict.keys():
        temp = graph_data_dict[uid]
        temp = nx.relabel_nodes(temp, node_mapping)
        for node in temp.nodes():
            temp.nodes[node]['node_feat'] = graph_embedding[node]
        temp_dgl = dgl.from_networkx(temp, node_attrs=['node_feat'], edge_attrs=['weight'])
        temp_dgl = dgl.add_self_loop(temp_dgl)
        dgl_graph_data_dict[uid] = temp_dgl
   
    del graph_data_dict
    gc.collect()
    
    return dgl_graph_data_dict



def get_entity_mapping(dfs):
    total_entities = []
    for df in dfs:
        total_entities.extend(df['entity'].unique().tolist())
    total_entities = [x.lower() for x in total_entities]
    total_entities = np.unique(total_entities).tolist()   
    entity_dict = dict()
    for ix, item in enumerate(total_entities):
        entity_dict[item] = ix
        
    return entity_dict
    
    
# --------------------------------------------------- CUSTOM DATASET CLASS --------------------------------------------------- 

class HVVCustomDataset(Dataset):
    
    def __init__(
        self,
        df: pd.DataFrame,
        data_type: str,
        lm_tokenizer,
        vision_tokenizer,
        face_data,
        kg_data
    ): 
        self.df = df
        print("\nDataset length: ", self.df.shape)
        print("\nLabel Distribution: \n", self.df['label'].value_counts())
        self.data_type = data_type
        self.lm_tokenizer = lm_tokenizer
        self.vision_tokenizer = vision_tokenizer
        self.face_data = face_data
        self.kg_data = kg_data
        
        
        
    def __len__(
        self
    ):
        return len(self.df)
    


    
    def process_image(
        self, 
        image_path
    ):
        image_mean = torch.tensor([0.485, 0.456, 0.406])
        image_std = torch.tensor([0.229, 0.224, 0.225])
        preprocess = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor()
        ])    
        image = preprocess(Image.open(image_path).convert('RGB'))
        image_input = torch.tensor(np.stack(image))
        image_input -= image_mean[:, None, None]
        image_input /= image_std[:, None, None]
        return image_input
    
    
    
    def __getitem__(
        self,
        index: int
    ):
        data_point = self.df.iloc[index]
        img = str(data_point['image'])
        ID = str(data_point['ID'])
        
        
        if USE_FACE:
            faces = self.face_data[img]
            if len(faces)>0:
                faces = " [MEME FACE] " + str(faces)
            else:
                faces = ""
        else:
            faces = ""
            
            
        context = "[MEME ENTITY]" + str(data_point['entity']) + faces + caption
        
        text_inputs = self.lm_tokenizer.encode_plus(
            str(data_point['sentence']),
            context,
            max_length=LM_MAX_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_token_type_ids=True,            
            return_tensors='pt'
        )
            
            
        data_inputs = {
            'input_ids': text_inputs['input_ids'].flatten(),
            'attention_mask': text_inputs['attention_mask'].flatten(),
            'token_type_ids': text_inputs["token_type_ids"].flatten(),
            'targets': torch.tensor(self.df.iloc[index]['label'], dtype=torch.long)
        }
        
        if USE_IMAGE:
            image_inputs = self.vision_tokenizer(
                images=self.process_image(DATA_DIR+str(img)),    
                return_tensors='pt'
            )
            data_inputs['image_inputs'] = {'pixel_values': torch.squeeze(image_inputs['pixel_values'], dim=0).to(dtype=torch.float32)}
        
        if USE_KG:
            data_inputs['kg_input'] = self.kg_data[ID]
                   
        return data_inputs