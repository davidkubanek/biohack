"""
This script is for the Enveda Team to supply their test parquet file
to evaluate our submitted model 
"""
from dataset import *
from config import CONFIG, CONFIG_ECFP, CONFIG_fp, CONFIG_grover, CONFIG_molformer
import pandas as pd
import torch
from torch.utils.data import DataLoader
from models import SiameseNetwork
import torch.nn as nn
from eval import test_one_epoch, test_ensemble
from args import test_args
import pdb

def eval_testdf(args):
    df = pd.read_pickle(args.test_dir)
    
    ### fp extraction
    df['ground_truth_fp'] = df['ground_truth_smiles'].apply(smiles_to_fingerprint)
    df['predicted_fp'] = df['predicted_smiles'].apply(smiles_to_fingerprint)

    ### GROVER extraction
    df['ground_truth_grover_fp'] = df['ground_truth_smiles'].apply(lambda x: find_grover_fp(x))
    df['predicted_grover_fp'] = df['predicted_smiles'].apply(lambda x: find_grover_fp(x))

    ### Graph embedding extraction
    df['ground_truth_graph_emb'] = prepare_datalist(df['ground_truth_smiles'])
    df['predicted_graph_emb'] = prepare_datalist(df['predicted_smiles'])

    ### MolFormer embedding extraction
    df['ground_truth_embeddings'] = df['ground_truth_smiles'].apply(extract_features)
    df['predicted_embeddings'] = df['predicted_smiles'].apply(extract_features)

    ### Save test set for later reuse and avoid recomputing features
    df.to_pickle('../data/enveda_test_split.pkl')
    
    # Initialize testset and test dataloader for each configuration
    df_ECFP = EnvedaDataset(CONFIG_ECFP, dataframe=df, labels=CONFIG_ECFP['labels'])
    df_molformer = EnvedaDataset(CONFIG_molformer, dataframe=df, labels=CONFIG_molformer['labels'])
    df_fp = EnvedaDataset(CONFIG_fp, dataframe=df, labels=CONFIG_fp['labels'])
    df_grover = EnvedaDataset(CONFIG_grover, dataframe=df, labels=CONFIG_grover['labels'])
    
    testloader_ECFP = DataLoader(df_ECFP, batch_size=CONFIG_ECFP['valid_batch_size'])
    testloader_molformer = DataLoader(df_molformer, batch_size=CONFIG_molformer['valid_batch_size'])
    testloader_fp = DataLoader(df_fp, batch_size=CONFIG_fp['valid_batch_size'])
    testloader_grover = DataLoader(df_grover, batch_size=CONFIG_grover['valid_batch_size'])

    # Load trained models for each configuration
    model_ECFP = SiameseNetwork(input_dim=CONFIG_ECFP['input_size'], output_dim=len(CONFIG_ECFP['labels'])).to(CONFIG['device'])
    model_molformer = SiameseNetwork(input_dim=CONFIG_molformer['input_size'], output_dim=len(CONFIG_molformer['labels'])).to(CONFIG['device'])
    model_fp = SiameseNetwork(input_dim=CONFIG_fp['input_size'], output_dim=len(CONFIG_fp['labels'])).to(CONFIG['device'])
    model_grover = SiameseNetwork(input_dim=CONFIG_grover['input_size'], output_dim=len(CONFIG_grover['labels'])).to(CONFIG['device'])
    model_ECFP.load_state_dict(torch.load('../results/best_F1_model_siamese_multiclass_sub_ECFP.bin', map_location=CONFIG['device']))
    model_molformer.load_state_dict(torch.load('../results/best_F1_model_siamese_multiclass_sub_molformer.bin', map_location=CONFIG['device']))
    model_fp.load_state_dict(torch.load('../results/best_F1_model_siamese_multiclass_sub_remove_duplicate.bin', map_location=CONFIG['device']))
    model_grover.load_state_dict(torch.load('../results/best_F1_model_siamese_multiclass_sub_grover.bin', map_location=CONFIG['device']))
    
    criterion = nn.CrossEntropyLoss()

    print('Evaluating each siamese network individually: ')
    print("Siamese with ECFP")
    _, _, _ = test_one_epoch(
        model=model_ECFP,
        dataloader=testloader_ECFP,
        criterion=criterion,
        epoch=1
    )
    print("Siamese with Molformer")
    _, _, _ = test_one_epoch(
        model=model_molformer,
        dataloader=testloader_molformer,
        criterion=criterion,
        epoch=1
    )
    print("Siamese with fp")
    _, _, _ = test_one_epoch(
    model=model_fp,
    dataloader=testloader_fp,
    criterion=criterion,
    epoch=1
    )
    print("Siamese with GROVER")
    _, _, _ = test_one_epoch(
        model=model_grover,
        dataloader=testloader_grover,
        criterion=criterion,
        epoch=1
    )
    
    print("Evaluating with an ensemble of siamese with different SMILE encoders:")
    models = [model_ECFP, model_molformer, model_fp, model_grover]
    dataloaders = [testloader_ECFP, testloader_molformer, testloader_fp, testloader_grover]

    # Call the ensemble evaluation function for a specific epoch
    _, _, _ = test_ensemble(models, dataloaders, criterion, 1)

    
if __name__ == "__main__":
    args = test_args()
    eval_testdf(args)