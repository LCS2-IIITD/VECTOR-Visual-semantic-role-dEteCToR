# Code for training utils for HVV

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, 
    recall_score, 
    precision_score, 
    accuracy_score,
    classification_report,
    confusion_matrix
)

import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from config.config import *
            
# ----------------------------------------------------- HVV TRAINING UTILS -----------------------------------------------------

def train_epoch_hvv(
    model,
    data_loader,
    optimizer,
    scheduler
):
    model.train()
    epoch_train_loss = 0.0
    pred_list = []
    gold_list = []
    for step, batch in enumerate(tqdm(data_loader, desc="Training Iteration")):
        
        model_kwargs = {
            'input_ids' : batch['input_ids'].to(DEVICE, dtype = torch.long),
            'attention_mask' : batch['attention_mask'].to(DEVICE, dtype = torch.long),
            'token_type_ids' : batch['token_type_ids'].to(DEVICE, dtype = torch.long),
            'labels' : batch['targets'].to(DEVICE, dtype = torch.long)   
        }
        
        if USE_IMAGE:
            model_kwargs['image_inputs'] = {k:v.to(DEVICE) for k,v in batch['image_inputs'].items()}

        if USE_KG:
            model_kwargs['kg_input'] = batch['kg_input'].to(DEVICE)
            
        outputs = model(**model_kwargs)

        loss = outputs['loss']
        epoch_train_loss += loss.item()
        loss.backward()
        
        if (step+1) % ACCUMULATION_STEPS == 0 :
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        preds = torch.argmax(outputs['logits'], axis=1).detach().cpu().numpy().tolist()
        gold = model_kwargs['labels'].cpu().tolist()

        pred_list.extend(preds)
        gold_list.extend(gold)
    
    return round(epoch_train_loss/ step, 4), pred_list, gold_list




def val_epoch_hvv(
    model,
    data_loader
):
    model.eval()
    
    epoch_val_loss = 0.0
    pred_list=[]
    gold_list = []
    for step, batch in enumerate(tqdm(data_loader, desc="Validation Iteration")):  
        
        with torch.no_grad():
            
            model_kwargs = {
                'input_ids' : batch['input_ids'].to(DEVICE, dtype = torch.long),
                'attention_mask' : batch['attention_mask'].to(DEVICE, dtype = torch.long),
                'token_type_ids' : batch['token_type_ids'].to(DEVICE, dtype = torch.long),
                'labels' : batch['targets'].to(DEVICE, dtype = torch.long)   

            }
            
            if USE_IMAGE:
                model_kwargs['image_inputs'] = {k:v.to(DEVICE) for k,v in batch['image_inputs'].items()}
                
            if USE_KG:
                model_kwargs['kg_input'] = batch['kg_input'].to(DEVICE)

            outputs = model(**model_kwargs)
        
            loss = outputs['loss']
            epoch_val_loss += loss.item()

            preds = torch.argmax(outputs['logits'], axis=1).detach().cpu().numpy().tolist()
            gold = model_kwargs['labels'].cpu().tolist()

            pred_list.extend(preds)
            gold_list.extend(gold)

        
    return round(epoch_val_loss/ step, 4), pred_list, gold_list




def test_epoch_hvv(
    model,
    data_loader
):
    model.eval()
    
    pred_list=[]
    gold_list = []
    for step, batch in enumerate(tqdm(data_loader, desc="Test Iteration")):  
        
        with torch.no_grad():
            
            model_kwargs = {
                'input_ids' : batch['input_ids'].to(DEVICE, dtype = torch.long),
                'attention_mask' : batch['attention_mask'].to(DEVICE, dtype = torch.long),
                'token_type_ids' : batch['token_type_ids'].to(DEVICE, dtype = torch.long)
            }
            labels = batch['targets'].to(DEVICE, dtype = torch.long)

            if USE_IMAGE:
                model_kwargs['image_inputs'] = {k:v.to(DEVICE) for k,v in batch['image_inputs'].items()}
                
            if USE_KG:
                model_kwargs['kg_input'] = batch['kg_input'].to(DEVICE)
                
            outputs = model(**model_kwargs)
            
            preds = torch.argmax(outputs['logits'], axis=1).detach().cpu().numpy().tolist()
            gold = labels.cpu().tolist()

            pred_list.extend(preds)
            gold_list.extend(gold)
        
    return pred_list, gold_list




def train_hvv(
    model_info,
    model,
    train_data_loader,
    val_data_loader,
    test_data_loader,
    learning_rate
):
    
    optimizer, scheduler = prepare_for_training(model=model,
                                                learning_rate=learning_rate, 
                                                num_training_steps=len(train_data_loader)* MAX_EPOCHS)
    
    train_losses = []
    val_losses = []
    val_metric = []
    patience = 0
    
    for epoch in range(MAX_EPOCHS):
        
        # Train Set
        train_loss, train_preds, train_gold = train_epoch_hvv(model,
                                                              train_data_loader, 
                                                              optimizer, 
                                                              scheduler=scheduler)
        train_losses.append(train_loss)
        train_results = get_prediction_scores(train_preds, train_gold)
        train_cr = classification_report(y_true=train_gold, y_pred=train_preds, output_dict=True, target_names=TARGET_NAMES)
        train_cm = confusion_matrix(y_true=train_gold, y_pred=train_preds)
        
        
        # Val Set
        val_loss, val_preds, val_gold = val_epoch_hvv(model,
                                                      val_data_loader)
        val_losses.append(val_loss)
        val_results = get_prediction_scores(val_preds, val_gold)
        val_cr = classification_report(y_true=val_gold, y_pred=val_preds, output_dict=True, target_names=TARGET_NAMES)
        val_cm = confusion_matrix(y_true=val_gold, y_pred=val_preds)
        
        
        # Test Set
        test_preds, test_gold = test_epoch_hvv(model,
                                               test_data_loader)
        test_results = get_prediction_scores(test_preds, test_gold)
        test_cr = classification_report(y_true=test_gold, y_pred=test_preds, output_dict=True, target_names=TARGET_NAMES)
        test_cm = confusion_matrix(y_true=test_gold, y_pred=test_preds)

        
        print("\nEpoch: {}\ttrain_loss: {}\tval_loss: {}\tmin_val_loss: {}".format(epoch+1, 
                                                                                   train_loss, 
                                                                                   val_loss, 
                                                                                   min(val_losses)))
        
         # ----------------------------------------- Train Results -----------------------------------------
        
        print("\n\ntrain_acc: {}\ttrain_precision: {}\ttrain_recall: {}\ttrain_f1: {}".format(train_results['Accuracy'], 
                                                                                            train_results['Precision'], 
                                                                                            train_results['Recall'], 
                                                                                            train_results['Macro-F1']))
        
        print("\ntrain_hero_precision: {}\ttrain_hero_recall: {}\ttrain_hero_f1: {}".format(train_cr['hero']['precision'], 
                                                                                            train_cr['hero']['recall'],
                                                                                            train_cr['hero']['f1-score']))
        
        print("\ntrain_villain_precision: {}\ttrain_villain_recall: {}\ttrain_villain_f1: {}".format(train_cr['villain']['precision'], 
                                                                                                     train_cr['villain']['recall'],
                                                                                                     train_cr['villain']['f1-score']))
        
        print("\ntrain_victim_precision: {}\ttrain_victim_recall: {}\ttrain_victim_f1: {}".format(train_cr['victim']['precision'],
                                                                                                  train_cr['victim']['recall'],
                                                                                                  train_cr['victim']['f1-score']))
        
        print("\ntrain_other_precision: {}\ttrain_other_recall: {}\ttrain_other_f1: {}".format(train_cr['other']['precision'], 
                                                                                               train_cr['other']['recall'],
                                                                                               train_cr['other']['f1-score']))
        print("\ntrain confusion matrix: ", train_cm)
        
        # ----------------------------------------- Val Results -----------------------------------------
        
        print("\n\nval_acc: {}\tval_precision: {}\tval_recall: {}\tval_f1: {}".format(val_results['Accuracy'], 
                                                                                      val_results['Precision'], 
                                                                                      val_results['Recall'], 
                                                                                      val_results['Macro-F1']))
        
        print("\nval_hero_precision: {}\tval_hero_recall: {}\tval_hero_f1: {}".format(val_cr['hero']['precision'], 
                                                                                      val_cr['hero']['recall'],
                                                                                      val_cr['hero']['f1-score']))
        
        print("\nval_villain_precision: {}\tval_villain_recall: {}\tval_villain_f1: {}".format(val_cr['villain']['precision'], 
                                                                                               val_cr['villain']['recall'],
                                                                                               val_cr['villain']['f1-score']))
        
        print("\nval_victim_precision: {}\tval_victim_recall: {}\tval_victim_f1: {}".format(val_cr['victim']['precision'],
                                                                                            val_cr['victim']['recall'],
                                                                                            val_cr['victim']['f1-score']))
        
        print("\nval_other_precision: {}\tval_other_recall: {}\tval_other_f1: {}".format(val_cr['other']['precision'], 
                                                                                         val_cr['other']['recall'],
                                                                                         val_cr['other']['f1-score']))
        print("\nvalidation confusion matrix: ", val_cm)
        
        # ----------------------------------------- test Results -----------------------------------------
        
        print("\n\ntest_acc: {}\ttest_precision: {}\ttest_recall: {}\ttest_f1: {}".format(test_results['Accuracy'], 
                                                                                          test_results['Precision'], 
                                                                                          test_results['Recall'], 
                                                                                          test_results['Macro-F1']))
        
        print("\ntest_hero_precision: {}\ttest_hero_recall: {}\ttest_hero_f1: {}".format(test_cr['hero']['precision'], 
                                                                                         test_cr['hero']['recall'],
                                                                                         test_cr['hero']['f1-score']))
        
        print("\ntest_villain_precision: {}\ttest_villain_recall: {}\ttest_villain_f1: {}".format(test_cr['villain']['precision'], 
                                                                                                  test_cr['villain']['recall'],
                                                                                                  test_cr['villain']['f1-score']))
        
        print("\ntest_victim_precision: {}\ttest_victim_recall: {}\ttest_victim_f1: {}".format(test_cr['victim']['precision'],
                                                                                               test_cr['victim']['recall'],
                                                                                               test_cr['victim']['f1-score']))
        
        print("\ntest_other_precision: {}\ttest_other_recall: {}\ttest_other_f1: {}".format(test_cr['other']['precision'], 
                                                                                            test_cr['other']['recall'],
                                                                                            test_cr['other']['f1-score']))
        print("\ntest confusion matrix: ", test_cm)
        
        # --------------------------------------------- Storing Val Results ---------------------------------------------
        
        if SAVE_RESULTS:
            val_result_df = pd.DataFrame(list(zip(val_gold, val_preds)), columns=['gold', 'preds'])

            val_folder = RESULT_PATH + str(model_info['model_name']) + '/' + str(model_info['timestamp']) + '/val/'

            if not os.path.exists(val_folder):
                print("\n\nCreating folder at: ", val_folder)
                os.makedirs(val_folder)

            val_path = val_folder + str(model_info['model_name']) + '_' + str(model_info['dataset_type']) + '_timestamp_' + str(model_info['timestamp']) +'_val_epoch_' + str(epoch+1) + '.csv'

            val_result_df.to_csv(val_path, index=False)
            print("\nStored val data at: ", val_path)

            # --------------------------------------------- Storing Test Results ---------------------------------------------

            test_result_df = pd.DataFrame(list(zip(test_gold, test_preds)), columns=['gold', 'preds'])

            test_folder = RESULT_PATH + str(model_info['model_name']) + '/' + str(model_info['timestamp']) + '/test/'

            if not os.path.exists(test_folder):
                print("\n\nCreating folder at: ", test_folder)
                os.makedirs(test_folder)

            test_path = test_folder + str(model_info['model_name']) + '_' + str(model_info['dataset_type']) + '_timestamp_' + str(model_info['timestamp']) +'_test_epoch_' + str(epoch+1) + '.csv'

            test_result_df.to_csv(test_path, index=False)
            print("\nStored test data at: ", test_path)

        