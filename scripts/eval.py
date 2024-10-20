
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import gc
from tqdm import tqdm


import torch

import pdb


@torch.no_grad()
def test_one_epoch(model, dataloader, criterion, epoch):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    running_auroc = 0.0
    running_f1 = 0.0 
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        
        gt, preds, targets = data
        gt, preds, targets = gt.to(CONFIG['device']), preds.to(CONFIG['device']), targets.to(CONFIG['device'])
        batch_size = gt.shape[0]
        
        with autocast():
            outputs = model(gt, preds)
            loss = criterion(outputs, targets)
            loss = loss / CONFIG['n_accumulate']
        
        probabilities = torch.softmax(outputs, dim=1).detach().cpu().numpy() if outputs.shape[1] > 1 else torch.sigmoid(outputs).detach().cpu().numpy()
        preds = np.eye(outputs.shape[1])[np.argmax(probabilities, axis=1)] if outputs.shape[1] > 1 else (probabilities > 0.5).astype(float)
        
        auroc = average_precision_score(targets.cpu().numpy(), probabilities, average='weighted')
        f1 = f1_score(targets.cpu().numpy(), preds, average='weighted')
        
        running_loss += (loss.item() * batch_size)
        running_auroc  += (auroc * batch_size)
        running_f1 += (f1 * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        epoch_auroc = running_auroc / dataset_size
        epoch_f1 = running_f1 / dataset_size
        
        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss, Valid_Auroc=epoch_auroc, 
                        Valid_F1=epoch_f1,
                        )   
    gc.collect()
    
    return epoch_loss, epoch_auroc, epoch_f1

@torch.no_grad()
def test_ensemble(models, dataloaders, criterion, epoch):
    # Set models to evaluation mode
    for model in models:
        model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    running_auroc = 0.0
    running_f1 = 0.0 

    # Initialize a list to accumulate model outputs
    all_outputs = []

    # Initialize a tensor for targets (assuming they are consistent across all dataloaders)
    targets_list = []
    model_list_names = ['ecfp', 'molformer', 'fp', 'grover']
    # Iterate over each dataloader and evaluate the corresponding model
    for i, (dataloader, model) in enumerate(zip(dataloaders, models)):
        bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Model {model_list_names[i]}')
        
        for step, data in bar:
            gt, preds, targets = data  # Get targets from each batch
            gt, preds, targets = gt.to(CONFIG['device']), preds.to(CONFIG['device']), targets.to(CONFIG['device'])
            
            with autocast():
                outputs = model(gt, preds)
                all_outputs.append(outputs)  # Store outputs on CPU for later averaging
            
            if i == 0:  # Only collect targets from the first dataloader
                targets_list.append(targets)

    # After processing all batches for all models, concatenate the outputs
    all_outputs_tensor = torch.cat(all_outputs, dim=0)  # Shape will be (total_samples, 5)

    # Average the outputs from all models
    ensemble_outputs = torch.mean(all_outputs_tensor.view(len(models), -1, 5), dim=0)  # Shape should be (88, 5)

    # Concatenate targets into a single tensor (assuming they are consistent)
    targets_tensor = torch.cat(targets_list, dim=0)  # Shape should be (88, 5)

    # Calculate loss and metrics using the ensemble outputs
    loss = criterion(ensemble_outputs, targets_tensor)

    probabilities = torch.softmax(ensemble_outputs, dim=1).detach().cpu().numpy() if ensemble_outputs.shape[1] > 1 else torch.sigmoid(ensemble_outputs).detach().cpu().numpy()
    preds = np.eye(ensemble_outputs.shape[1])[np.argmax(probabilities, axis=1)] if ensemble_outputs.shape[1] > 1 else (probabilities > 0.5).astype(float)

    auroc = average_precision_score(targets_tensor.cpu().numpy(), probabilities, average='weighted')
    f1 = f1_score(targets_tensor.cpu().numpy(), preds, average='weighted')

    running_loss += loss.item() * len(targets_tensor)
    running_auroc += auroc * len(targets_tensor)
    running_f1 += f1 * len(targets_tensor)
    dataset_size += len(targets_tensor)

    epoch_loss = running_loss / dataset_size
    epoch_auroc = running_auroc / dataset_size
    epoch_f1 = running_f1 / dataset_size

    print(f'Epoch: {epoch}, Valid Loss: {epoch_loss}, Valid AUROC: {epoch_auroc}, Valid F1: {epoch_f1}')
    
    gc.collect()
    
    return epoch_loss, epoch_auroc, epoch_f1
