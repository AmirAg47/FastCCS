import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdFreeSASA, rdFingerprintGenerator
from rdkit.Chem import rdmolops
import uuid
import re
from rdkit.Chem.Descriptors import CalcMolDescriptors
from typing import List, Optional
import pickle
from pathlib import Path  
from django.conf import settings 
import tensorflow as tf  # Import TensorFlow
from sklearn.preprocessing import MinMaxScaler


def load_encoder():
    with open(Path(settings.BASE_DIR) / 'models/one_hot_encoder.pkl', 'rb') as f:
        return pickle.load(f)




def load_model():
    # Path to your .keras model (update the path as necessary)
    model_path = Path(settings.BASE_DIR) / 'models/dnn_model.keras'  # Update to your model's path

    # Load the model using TensorFlow's Keras API
    model = tf.keras.models.load_model(model_path)
    
    return model

def load_scaler():
    """
    Loads the scaler from a pickle file located in the 'models' directory.
  
    """
    # Load the scaler from the pickle file using the path to 'models'
    with open(Path(settings.BASE_DIR) / 'models/scaler.pkl', 'rb') as f:
        return pickle.load(f)

def load_cluster():
    with open(Path(settings.BASE_DIR) / 'models/kmeans_model.pkl', 'rb') as f:
        return pickle.load(f) 

def safe_compute_properties(smiles, adduct):  # Add adduct here
    try:
        if pd.isna(smiles) or isinstance(smiles, float):
            return [np.nan] * 320
        return compute_molecular_properties(smiles, adduct)  # Pass adduct to this function
    except Exception as e:
        print(f"Error processing SMILES: {smiles}, Error: {e}")
        return [np.nan] * 320
def compute_molecular_properties(smiles, adduct):
    # Create the molecule from the SMILES string
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return [np.nan] * 320  # Return NaN for all properties if molecule cannot be processed

    # Add explicit hydrogens
    #mol = Chem.AddHs(mol)

    # Ensure a conformer exists
    if mol.GetNumConformers() == 0:
        ps = AllChem.ETKDG()
        ps.randomSeed = 42
        ps.maxAttempts = 1
        ps.numThreads = 0
        ps.useRandomCoords = True
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=1, params=ps)
        if len(conf_ids) == 0:
            print(f"Failed to embed molecule for SMILES: {smiles}")
            return [np.nan] * 33  # Return NaN for all properties if embedding fails


    # Optimize the conformation
    if mol.GetNumConformers() > 0:
        res = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)
        if len(res) == 0:
            print(f"Failed to optimize molecule for SMILES: {smiles}")

    # Calculate General molecular properties
    calc_descriptors = CalcMolDescriptors(mol)
    general_values = list(calc_descriptors.values())
    #Calculate mqn
    def compute_mqns_from_mol(mol) -> Optional[List[float]]:
        try:
            mqns = Descriptors.rdMolDescriptors.MQNs_(mol)
            return mqns
        except Exception as e:  
            return None   
    mqn_values = compute_mqns_from_mol(mol)
    # mqn_values = list(mqn_features.values())
    # Special properties
    mol_aromatic_count = sum(atom.GetIsAromatic() for atom in mol.GetAtoms())
    
    # Calculate SASA
    try:
        radii = rdFreeSASA.classifyAtoms(mol)
        sasa = rdFreeSASA.CalcSASA(mol, radii)
    except RuntimeError:
        sasa = np.nan

    # Shannon entropy
    atom_degrees = [atom.GetDegree() for atom in mol.GetAtoms()]
    total_degree = sum(atom_degrees)
    shannon_entropy = 0.0
    if total_degree > 0:
        for degree in atom_degrees:
            probability = degree / total_degree
            if probability > 0:
                shannon_entropy -= probability * np.log(probability)

    pbf = rdMolDescriptors.CalcPBF(mol)

    # Custom functions for Zagreb indices
    def calculate_zagreb1(mol):
        deltas = [x.GetDegree() for x in mol.GetAtoms()]
        return sum(np.array(deltas) ** 2)

    def calculate_zagreb2(mol):
        ke = [x.GetBeginAtom().GetDegree() * x.GetEndAtom().GetDegree() for x in mol.GetBonds()]
        return sum(ke)

    def calculate_mzagreb1(mol):
        deltas = [x.GetDegree() for x in mol.GetAtoms()]
        while 0 in deltas:
            deltas.remove(0)
        deltas = np.array(deltas, 'd')
        return sum((1. / deltas) ** 2)

    def calculate_mzagreb2(mol):
        cc = [x.GetBeginAtom().GetDegree() * x.GetEndAtom().GetDegree() for x in mol.GetBonds()]
        while 0 in cc:
            cc.remove(0)
        cc = np.array(cc, 'd')
        return sum(1. / cc)

    zagreb1 = calculate_zagreb1(mol)
    zagreb2 = calculate_zagreb2(mol)
    mzagreb1 = calculate_mzagreb1(mol)
    mzagreb2 = calculate_mzagreb2(mol)

    def wiener_index(mol):
        n_atoms = mol.GetNumAtoms()
        if n_atoms < 2:
            return 0
        path_lengths = np.zeros((n_atoms, n_atoms))
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                path_length = Chem.GetDistanceMatrix(mol)[i, j]
                path_lengths[i, j] = path_length
                path_lengths[j, i] = path_length
        return int(np.sum(path_lengths) / 2)

    wiener_idx = wiener_index(mol)

    asphericity = rdMolDescriptors.CalcAsphericity(mol)
    radius_of_gyration = rdMolDescriptors.CalcRadiusOfGyration(mol)

    def count_heterocycles_by_type(mol):
    
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        o_heterocycles, n_heterocycles, s_heterocycles, total_heterocycles = 0, 0, 0, 0
        for ring in atom_rings:
            heteroatoms_in_ring = set(mol.GetAtomWithIdx(atom_idx).GetAtomicNum() for atom_idx in ring)
            if any(atom_num != 6 for atom_num in heteroatoms_in_ring):
                total_heterocycles += 1
                if 8 in heteroatoms_in_ring: o_heterocycles += 1
                if 7 in heteroatoms_in_ring: n_heterocycles += 1
                if 16 in heteroatoms_in_ring: s_heterocycles += 1
        return o_heterocycles, n_heterocycles, s_heterocycles, total_heterocycles

    o_heterocycles, n_heterocycles, s_heterocycles, total_heterocycles = count_heterocycles_by_type(mol)

    def count_halogens(mol):
        halogens = {9, 17, 35, 53, 85}  # Atomic numbers for F, Cl, Br, I, At
        count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in halogens)
        return count
    halogen_count = count_halogens(mol)
 
    def calculate_mz(adduct):
        adduct = str(adduct)  # Ensure 'Adduct' is a string
        M = Chem.Descriptors.MolWt(mol) # Accessing the ExactMolWt column for M weights
        base_mass = M  # Start with base mass as M
        mass_dict = {
                        'H': 1.0078, 'Na': 22.9898, 'K': 39.0983, 'Li': 6.941,
                        'H2O': 18.015, 'NH3': 17.031, 'Cl': 35.453, 'CH3COO': 59.013,
                        'SO3': 80.063, 'HCOO': 45.017, 'CO2': 44.01, 'Cs': 132.905,
                        'Rb': 85.468
                    }
        # Process for multiplier (e.g., "2M")
        if "2M" in adduct:
            base_mass *= 2
    
        # Process additions and subtractions using regular expressions
        for key, value in mass_dict.items():
            # Find all occurrences of +X or -X for the molecule
            add_matches = re.findall(rf"\+(\d*){key}", adduct)
            sub_matches = re.findall(rf"\-(\d*){key}", adduct)
    
            # Subtract the masses for each match (use default 1 if no number is given)
            for match in sub_matches:
                multiplier = int(match) if match else 1  # Default to 1 if no number is given
                base_mass -= multiplier * value
            
            # Add the masses for each match (use default 1 if no number is given)
            for match in add_matches:
                multiplier = int(match) if match else 1  # Default to 1 if no number is given
                base_mass += multiplier * value
    
        # Ignore charge parsing until after the bracket
        charge = 1  # Default charge
        if "]" in adduct:
            # Extract the part of the adduct after the closing bracket "]"
            charge_part = adduct.split("]")[-1]
            
            # Determine charge (based on presence of + or - charge in adduct)
            if "2+" in charge_part:
                charge = 2
            elif "3+" in charge_part:
                charge = 3
            elif "4+" in charge_part:
                charge = 4
            elif "5+" in charge_part:
                charge = 5
            elif "2-" in charge_part:
                charge = -2
            elif "3-" in charge_part:
                charge = -3
            elif "-" in charge_part and "+" not in charge_part:
                charge = -1
    
        # Calculate m/z
        mz = abs(base_mass / abs(charge))
        return mz

    mZ = calculate_mz(adduct)
    special_values = [ mol_aromatic_count,sasa, shannon_entropy, pbf, wiener_idx, zagreb1, zagreb2,
                      mzagreb1, mzagreb2, asphericity,radius_of_gyration, o_heterocycles, n_heterocycles, 
                      s_heterocycles, total_heterocycles, halogen_count, mZ]

    return general_values + special_values  + mqn_values
    
    
#def predict_data(model, data):
    #return model.predict(data)
# utils.py
# utils.py

general_columns = [
    'MaxAbsEStateIndex', 'MaxEStateIndex', 'MinAbsEStateIndex', 'MinEStateIndex', 'qed', 'SPS', 'MolWt',
    'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', 'NumRadicalElectrons', 'MaxPartialCharge',
    'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'FpDensityMorgan1', 'FpDensityMorgan2',
    'FpDensityMorgan3', 'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI',
    'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'AvgIpc', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v',
    'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Ipc',
    'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13',
    'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8',
    'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7',
    'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3',
    'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'EState_VSA1',
    'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6',
    'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3',
    'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'FractionCSP3',
    'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
    'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors',
    'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles',
    'NumSaturatedRings', 'RingCount', 'MolLogP', 'MolMR', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN',
    'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S',
    'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2',
    'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide',
    'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene',
    'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether',
    'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan',
    'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine',
    'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime',
    'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine',
    'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd',
    'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene',
    'fr_unbrch_alkane', 'fr_urea'
]

special_columns = [
    'Aromatic_Count','sasa','Shannon_Entropy', 'PBF', 'Wiener_Index',
    'Zagreb1', 'Zagreb2', 'MZagreb1', 'MZagreb2', 'asphericity', 'radius_of_gyration',
    'o_heterocycles', 'n_heterocycles', 's_heterocycles', 'total_heterocycles', 'halogen_count',
    'm/z'
]

mqn_columns = [f"MQN_{i+1}" for i in range(42)]

property_columns = general_columns + special_columns + mqn_columns


def predict_data(model, data, scaler, kmeans, feature_indices=None) -> pd.DataFrame:
    """
    Predicts using a trained TensorFlow model, applying Min-Max scaling to the data inside the function,
    and selects features based on given indices. Returns a DataFrame with predictions added.
    
    :param model: The trained TensorFlow model for running inference.
    :param data: The input data to predict, should be a pandas DataFrame.
    :param scaler: The scaler used for scaling the data.
    :param kmeans: The trained k-means clustering model.
    :param feature_indices: A list of indices of features to be selected (optional).
    """
    
    # Create a copy of the data to avoid modifying the original
    data_copy = data.copy()

    # Save 'Smiles' and 'Adduct' columns for later (they will not be used in scaling/prediction)
    smiles_and_adduct = data_copy[['Smiles', 'Adduct', 'm/z']]
    data_copy = data_copy.drop(columns=['Smiles', 'Adduct'])  # Drop them for scaling and prediction

    # Apply the pre-fitted scaler to all columns in the data
    data_copy = scaler.transform(data_copy)
    print("Scaling applied using pre-fitted scaler.")

    # Check the number of columns after scaling
    print(f"Scaled data shape: {data_copy.shape}")
    
    # Clustering
    selected_features = data_copy[:, 51:320]
    cluster = kmeans.predict(selected_features)
    data_copy = pd.DataFrame(data_copy)
    data_copy['Cluster'] = cluster
    data_copy = np.array(data_copy)
    
    # If feature selection is specified, apply it to the data
    if feature_indices is not None:
        feature_indices = np.array(feature_indices)  # Ensure feature_indices is a NumPy array
        data_copy = data_copy[:, feature_indices]  # Select the required features, using slicing for NumPy arrays
        print(f"Features selected using indices: {feature_indices}")

    # Ensure that the number of columns after selection matches the model's expected input
    input_shape = model.input_shape  # Get the expected input shape from the model
    if data_copy.shape[1] != input_shape[1]:
        raise ValueError(f"Expected {input_shape[1]} features, but received {data_copy.shape[1]} features. "
                         "Please check the feature selection logic.")

    # Initialize prediction list
    predictions = model.predict(data_copy)  # Using the regular `predict()` method for TensorFlow model

    # Reattach the 'Smiles' and 'Adduct' columns back to the DataFrame
    result_df = pd.DataFrame(data_copy, columns=[f"Feature_{i}" for i in range(data_copy.shape[1])])  # Create columns for the features
    
    result_df['CCS Å²'] = predictions.flatten()

    result_df['Smiles'] = smiles_and_adduct['Smiles']
    result_df['Adduct'] = smiles_and_adduct['Adduct']
    result_df['m/z'] = smiles_and_adduct['m/z']
    result_df['Cluster'] = cluster  # Add the cluster column from the KMeans model
    # Select the required columns ('Smiles', 'Adduct', and 'Predictions')
    result_df = result_df[['Smiles', 'Adduct', 'CCS Å²', 'm/z', 'Cluster']]
    
    return result_df
