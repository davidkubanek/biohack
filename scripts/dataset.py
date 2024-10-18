import torch
from torch.utils.data import Dataset
from config import CONFIG
from transformers import AutoModel, AutoTokenizer

# smiles to graphs
import numpy as np

# RDkit
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import DataStructs

# Pytorch and Pytorch Geometric
import torch
from torch_geometric.data import Data

# A file with the datasets defined over the notebooks, including
# - EnvedaDataset for loading dataset with embeddings
# - PyTorch Geometric class for computing graph embeddings

class EnvedaDataset(Dataset):
    def __init__(self, CONFIG, dataframe, labels = []):
        """
        Args:
            dataframe (pd.DataFrame): A DataFrame containing 'ground_truth_embeddings', 
                                       'predicted_embeddings', and output columns.
        """
        self.dataframe = dataframe
        
        if CONFIG['FP'] == 'molformer':
            # Convert Molformer embeddings to tensors
            self.ground_truth_embeddings = torch.tensor(dataframe['ground_truth_embeddings'].tolist(), dtype=torch.float32)
            self.predicted_embeddings = torch.tensor(dataframe['predicted_embeddings'].tolist(), dtype=torch.float32)
    
        # fingerprints
        elif CONFIG['FP'] == 'ECFP':
            self.ground_truth_embeddings = torch.tensor(dataframe['ground_truth_ECFP'].tolist(), dtype=torch.float32)
            self.predicted_embeddings = torch.tensor(dataframe['predicted_ECFP'].tolist(), dtype=torch.float32)
        elif CONFIG['FP'] == 'fp':
            self.ground_truth_embeddings = dataframe['ground_truth_fp']
            self.predicted_embeddings = dataframe['predicted_fp']

        elif CONFIG['FP'] == 'grover':
            self.ground_truth_embeddings = torch.tensor(dataframe['ground_truth_grover_fp'].tolist(), dtype=torch.float32)
            self.predicted_embeddings = torch.tensor(dataframe['predicted_grover_fp'].tolist(), dtype=torch.float32)

        elif CONFIG['FP'] == 'graph':
            self.ground_truth_embeddings = dataframe['ground_truth_graph_emb']
            self.predicted_embeddings = dataframe['predicted_graph_emb']
            if CONFIG['fp_concat'] is True:
                if CONFIG['FP2'] == 'molformer':
                    # Convert Molformer embeddings to tensors
                    self.ground_truth_fps = torch.tensor(dataframe['ground_truth_embeddings'].tolist(), dtype=torch.float32)
                    self.predicted_fps = torch.tensor(dataframe['predicted_embeddings'].tolist(), dtype=torch.float32)
            
                # fingerprints
                elif CONFIG['FP2'] == 'ECFP':
                    self.ground_truth_fps = torch.tensor(dataframe['ground_truth_ECFP'].tolist(), dtype=torch.float32)
                    self.predicted_fps = torch.tensor(dataframe['predicted_ECFP'].tolist(), dtype=torch.float32)
                elif CONFIG['FP2'] == 'fp':
                    self.ground_truth_fps = dataframe['ground_truth_fp']
                    self.predicted_fps = dataframe['predicted_fp']

                elif CONFIG['FP2'] == 'grover':
                    self.ground_truth_fps = torch.tensor(dataframe['ground_truth_grover_fp'].tolist(), dtype=torch.float32)
                    self.predicted_fps = torch.tensor(dataframe['predicted_grover_fp'].tolist(), dtype=torch.float32)
            
        
        self.labels_text = labels
        # Convert labels to tensor
        self.labels = torch.tensor(dataframe[labels].values, dtype=torch.float32)

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.dataframe)

    def __getitem__(self, idx):
        """Generates one sample of data."""
        if CONFIG['FP'] == 'molformer':
            return self.ground_truth_embeddings[idx].squeeze(0), self.predicted_embeddings[idx].squeeze(0), self.labels[idx]
        elif CONFIG['FP'] == 'fp':
            return self.ground_truth_embeddings.iloc[idx].squeeze(0), self.predicted_embeddings.iloc[idx].squeeze(0), self.labels[idx]
        elif CONFIG['FP'] == 'ECFP':
            return self.ground_truth_embeddings[idx], self.predicted_embeddings[idx], self.labels[idx]
        elif CONFIG['FP'] == 'grover':
            return self.ground_truth_embeddings[idx].squeeze(0), self.predicted_embeddings[idx].squeeze(0), self.labels[idx]
        elif CONFIG['FP'] == 'graph':
            if CONFIG['fp_concat'] is True:
                return self.ground_truth_embeddings.iloc[idx], self.predicted_embeddings.iloc[idx], self.labels[idx], self.ground_truth_fps.iloc[idx].squeeze(0), self.predicted_fps.iloc[idx].squeeze(0)
            else:
                return self.ground_truth_embeddings.iloc[idx], self.predicted_embeddings.iloc[idx], self.labels[idx]
        


class GraphDatasetClass:
    '''
    Convert a dataframe of SMILES into a Pytorch Geometric Graph Dataset.
    https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/
    '''

    def __init__(self):
        pass

    def one_hot_encoding(self, x, permitted_list):
        """
        Maps input elements x which are not in the permitted list to the last element
        of the permitted list.
        """

        if x not in permitted_list:
            x = permitted_list[-1]

        binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]

        return binary_encoding


    def get_atom_features(self, atom,
                        use_chirality = True,
                        hydrogens_implicit = True):
        """
        Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
        """

        # define list of permitted atoms

        permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I', 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']

        if hydrogens_implicit == False:
            permitted_list_of_atoms = ['H'] + permitted_list_of_atoms

        # compute atom features

        atom_type_enc = self.one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)

        n_heavy_neighbors_enc = self.one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])

        formal_charge_enc = self.one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])

        hybridisation_type_enc = self.one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])

        is_in_a_ring_enc = [int(atom.IsInRing())]

        is_aromatic_enc = [int(atom.GetIsAromatic())]

        atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]

        vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]

        covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]

        atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled

        if use_chirality == True:
            chirality_type_enc = self.one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
            atom_feature_vector += chirality_type_enc

        if hydrogens_implicit == True:
            n_hydrogens_enc = self.one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
            atom_feature_vector += n_hydrogens_enc

        return np.array(atom_feature_vector)

    def get_bond_features(self, bond,
                        use_stereochemistry = True):
        """
        Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
        """

        permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

        bond_type_enc = self.one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)

        bond_is_conj_enc = [int(bond.GetIsConjugated())]

        bond_is_in_ring_enc = [int(bond.IsInRing())]

        bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc

        if use_stereochemistry == True:
            stereo_type_enc = self.one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
            bond_feature_vector += stereo_type_enc

        return np.array(bond_feature_vector)

    def create_pytorch_geometric_graph_data_list_from_smiles_and_labels(self, x_smiles):
        """
        Inputs:

        x_smiles = [smiles_1, smiles_2, ....] ... a list of SMILES strings
        y = [y_1, y_2, ...] ... a list of numerial labels for the SMILES strings (such as associated pKi values)

        Outputs:

        data_list = [G_1, G_2, ...] ... a list of torch_geometric.data.Data objects which represent labeled molecular graphs that can readily be used for machine learning

        """

        data_list = []

        for smiles in x_smiles:

            # convert SMILES to RDKit mol object
            mol = Chem.MolFromSmiles(smiles)

            # get feature dimensions
            n_nodes = mol.GetNumAtoms()
            n_edges = 2*mol.GetNumBonds()
            unrelated_smiles = "O=O"
            unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
            n_node_features = len(self.get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
            n_edge_features = len(self.get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))

            # construct node feature matrix X of shape (n_nodes, n_node_features)
            X = np.zeros((n_nodes, n_node_features))

            for atom in mol.GetAtoms():
                X[atom.GetIdx(), :] = self.get_atom_features(atom)

            X = torch.tensor(X, dtype = torch.float)

            # construct edge index array E of shape (2, n_edges)
            (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
            torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
            torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
            E = torch.stack([torch_rows, torch_cols], dim = 0)

            # construct edge feature array EF of shape (n_edges, n_edge_features)
            EF = np.zeros((n_edges, n_edge_features))

            for (k, (i,j)) in enumerate(zip(rows, cols)):

                EF[k] = self.get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))

            EF = torch.tensor(EF, dtype = torch.float)


            # construct Pytorch Geometric data object and append to data list
            data_list.append(Data(x = X, edge_index = E, edge_attr = EF))

        return data_list


### Helper functions 

# Get Graph embeddings
def prepare_datalist(smiles):
    '''
    Convert matrix dataframe to a data_list with pytorch geometric graph data, fingerprints and labels.
    Inputs:
        matrix_df: dataframe of SMILES, assays and bioactivity labels
        args: arguments
        graph_fp: if True, includes graph embedding fingerprints into data_list
        grover_fp: if True, includes GROVER graph transformer embedding fingerprints into data_list
    Outputs:
        data_list: list of data objects
    '''
    # get SMILES strings
    data = smiles

    # add graph fingerprint

    GraphDataset = GraphDatasetClass()
    # create pytorch geometric graph data list
    data_list = GraphDataset.create_pytorch_geometric_graph_data_list_from_smiles_and_labels(data)

    print(f'Example of a graph data object: {data_list[0]}')

    return data_list



# Get MolFormer features
model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True).to(CONFIG['device'])
tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
def extract_features(smiles):
    # Tokenize the SMILES string
    inputs = tokenizer(smiles, return_tensors='pt', padding=True, truncation=True).to(CONFIG['device'])
    
    # Disable gradient calculation for inference
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # pdb.set_trace()
    # Extract the last hidden state (embeddings)
    embeddings = outputs.pooler_output.cpu().numpy()
    return embeddings

# Get FP
def smiles_to_fingerprint(smile):
    """
    Convert SMILES string to concatenation of RDFKIT, Morgan and MACCSS fingerprints.
    """
    # RDFKIT (fingerprint dim fpSize=1024)
    x = Chem.MolFromSmiles(smile)
    fp1 = Chem.RDKFingerprint(x, fpSize=1024)

    # MACCSS substructure (fingerprint dim 167)
    fp2 = MACCSkeys.GenMACCSKeys(x)

    # Morgan (fingerprint dim fpSize=1024)
    fp_hashes = []
    fp3 = AllChem.GetHashedMorganFingerprint(x, 2, nBits=1024)
    fp3_array = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp3, fp3_array)
    fp_hashes.append(fp3_array)

    # combine all fingerprints
    fp = fp1.ToBitString() + fp2.ToBitString()
    fp = np.array(list(fp)).astype(np.int8)
    fp = np.concatenate([fp] + fp_hashes)
    fp = torch.tensor(fp).to(torch.float32)
    return fp

# Get GROVER embeddings
grover_fp = np.load('../data/fp_large_unique_smiles.npz', allow_pickle=True)
def find_grover_fp(smile):
    # Get the corresponding grover fingerprint
    # find the index of the SMILES in the grover_fp
    idx = np.where(grover_fp['smiles'] == smile)[0][0]

    return grover_fp['fps'][idx]

# Get ECFP embeddings
def ecfp_embeddings(df):
    # represent ground truth smiles and predicted smiles as ECFP4 fingerprints
    df['ground_truth_mol'] = df['ground_truth_smiles'].apply(Chem.MolFromSmiles)
    df['predicted_mol'] = df['predicted_smiles'].apply(Chem.MolFromSmiles)
    fpgen = AllChem.GetMorganGenerator(radius=2)
    df['ground_truth_ECFP'] = df['ground_truth_mol'].apply(lambda x: fpgen.GetFingerprint(x))
    df['predicted_ECFP'] = df['predicted_mol'].apply(lambda x: fpgen.GetFingerprint(x))
    return df