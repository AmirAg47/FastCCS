import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdFreeSASA
import re
from rdkit.Chem.Descriptors import CalcMolDescriptors
from typing import List, Optional
import pickle 
from pathlib import Path
import seaborn as sns
import sys
sys.path.append('.')
from utils.utils import *
from tqdm import tqdm

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
        return [np.nan] * 269  # Return NaN for all properties if molecule cannot be processed

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
            return [np.nan] * 320  # Return NaN for all properties if embedding fails

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

    return general_values + special_values + mqn_values



def load_encoder():
    encoder_path = 'Encoder/one_hot_encoder.pkl'  # Adjust this if the file is in a subdirectory

    with open(encoder_path, 'rb') as f:
        return pickle.load(f)

        
def descriptor(file_path, output_dir):
    try:
        # Load the new data
        new_df = pd.read_csv(file_path)  # Use file_path instead of csv_file_path

        
        # Load encoder and preprocess
        encoder = load_encoder()
        new_adducts = new_df["Adduct"].values.reshape(-1, 1)
        
        # Ensure encoder is not None
        if encoder is None:
            raise ValueError("Encoder is None. Please check the encoder loading process.")
        
        # One-hot encode the 'Adduct' column
        one_hot_encoded_new_adducts = encoder.transform(new_adducts)
        new_one_hot_df = pd.DataFrame(one_hot_encoded_new_adducts, columns=encoder.get_feature_names_out(["Adduct"]))
        new_df_encoded_adduct = pd.concat([new_df, new_one_hot_df], axis=1)

        # Compute molecular properties
        tqdm.pandas()
        properties_df = new_df_encoded_adduct.progress_apply(
        lambda row: pd.Series(safe_compute_properties(row['Smiles'], row['Adduct'])),axis=1  )
        properties_df.columns = property_columns

        # Concatenate the computed properties with the original DataFrame
        new_df_encoded_adduct = pd.concat([new_df_encoded_adduct, properties_df], axis=1)

        # Remove the prediction step and save the results instead
        output_dir = Path(output_dir)  # 
        output_file = output_dir / "descripted.csv"
        new_df_encoded_adduct.to_csv(output_file, index=False)

        print(f"Descripting complete. File saved at {output_file}")
        # Return the output file path
        return output_file
        
    except Exception as e:
        print(f"Error during descripting: {e}")
        return None
