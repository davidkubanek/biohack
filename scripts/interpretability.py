import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
import matplotlib.pyplot as plt
import pdb
from PIL import Image
import io
from config import CONFIG

def visualize_molecule(smile1, smile2, grads1, grads2):
    """
    Visualizes two molecules from SMILES strings with integrated gradient scores.
    
    Parameters:
        smile1 (str): The first SMILES string.
        smile2 (str): The second SMILES string.
        grads1 (np.ndarray): Integrated gradients for the first molecule.
        grads2 (np.ndarray): Integrated gradients for the second molecule.
    """
    
    # Convert SMILES to RDKit molecules
    mol1 = Chem.MolFromSmiles(smile1)
    mol2 = Chem.MolFromSmiles(smile2)

    # Normalize integrated gradient scores for better visualization
    normed_scores1 = (grads1[0] - np.min(grads1[0])) / (np.max(grads1[0]) - np.min(grads1[0]))
    normed_scores2 = (grads2[0] - np.min(grads2[0])) / (np.max(grads2[0]) - np.min(grads2[0]))

    # Create the first figure for Molecule 1
    fig1, axs1 = plt.subplots(2, 1, figsize=(8, 12))  # Two rows, one column

    # Molecule 1 without colors
    img1_no_color = Draw.MolToImage(mol1,
                                     size=(300, 300),
                                     kekulize=True)
    
    axs1[0].imshow(img1_no_color)
    axs1[0].axis('off')
    axs1[0].set_title('Ground Truth Molecule')

    # Molecule 1 with bonds highlighted based on integrated gradients
    draw2d_1 = Draw.rdMolDraw2D.MolDraw2DCairo(300, 300)
    
    # Generate a similarity map based on the weights for bonds
    SimilarityMaps.GetSimilarityMapFromWeights(mol1, normed_scores1.tolist(), 'coolwarm')
    
    draw2d_1.FinishDrawing()
    
    # Convert drawing text to an image using PIL
    img_data_1 = draw2d_1.GetDrawingText()
    img1_colored = Image.open(io.BytesIO(img_data_1))
    
    axs1[1].imshow(img1_colored)
    axs1[1].axis('off')
    axs1[1].set_title('Ground Truth Molecule with Bond Highlighting with \n Integrated Gradient Attention Scores')
    
    pred = CONFIG['labels'][torch.argmax(target_one_hot).item()]
    # fig.suptitle(f'Visualizing prediction of {pred} \n with Integrated Gradients', fontsize=16, fontweight='bold')
    
    fig1.suptitle(f'Interpreting prediction of {pred} \n \n Visualization of Ground Truth Molecule', fontsize=16, fontweight='bold')
    
    plt.tight_layout(pad=3.0)  # Adjust padding to prevent overlap
    plt.show()

    
    # Create the second figure for Molecule 2
    fig2, axs2 = plt.subplots(2, 1, figsize=(8, 12))  # Two rows, one column

    # Molecule 2 without colors
    img2_no_color = Draw.MolToImage(mol2,
                                     size=(300, 300),
                                     kekulize=True)
    
    axs2[0].imshow(img2_no_color)
    axs2[0].axis('off')
    axs2[0].set_title('Predicted Molecule')

    # Molecule 2 with bonds highlighted based on integrated gradients
    draw2d_2 = Draw.rdMolDraw2D.MolDraw2DCairo(300, 300)

    # Generate a similarity map based on the weights for bonds
    SimilarityMaps.GetSimilarityMapFromWeights(mol2, normed_scores2.tolist(), 'coolwarm')

    draw2d_2.FinishDrawing()
    
    # Convert drawing text to an image using PIL
    img_data_2 = draw2d_2.GetDrawingText()
    img2_colored = Image.open(io.BytesIO(img_data_2))
    
    axs2[1].imshow(img2_colored)
    axs2[1].axis('off')
    axs2[1].set_title('Predicted Molecule with Bond Highlighting with \n Integrated Gradient Attention Scores')

    fig2.suptitle('Visualization of Predicted Molecule', fontsize=16, fontweight='bold')
    
    plt.tight_layout(pad=1.0)  # Adjust padding to prevent overlap
    plt.show()
