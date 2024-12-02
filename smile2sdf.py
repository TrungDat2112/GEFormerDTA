from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

df = pd.read_csv('davis/split/trainer.csv')  
smilesList = df['compound_iso_smiles']

i=0
for smile in smilesList:
    mol = Chem.MolFromSmiles(smile)
    i=i+1
    writer = Chem.SDWriter('davis/sdf/'+smile+'.sdf')

    writer.SetProps(['LOGP', 'MW', 'TPSA', '价电子数'])  
    mw = Descriptors.ExactMolWt(mol)
    logP = Descriptors.MolLogP(mol)
    TPSA = Descriptors.TPSA(mol)
    ValueElectronsNum = Descriptors.NumValenceElectrons(mol)
    name = Descriptors.names
    mol.SetProp('MW', '%.2f' % mw) 
    mol.SetProp('LOGP', '%.2f' % logP) 
    mol.SetProp('_Name', 'No_%s' % i) 
    mol.SetProp('TPSA', '%s' % TPSA)  
    mol.SetProp('价电子数', '%s' % ValueElectronsNum)  
    writer.write(mol)
    writer.close() 