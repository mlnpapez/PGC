import os
import json
import torch
import urllib
import pandas

from rdkit import Chem
from tqdm import tqdm
from rdkit import RDLogger

from utils.molecular import mol2g, g2mol
from utils.graphs import permute_graph, flatten_tril
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, depth_first_order, reverse_cuthill_mckee


BASE_DIR = '/mnt/data/density_learning/molspn/'

MOLECULAR_DATASETS = {
    'qm9': {
        'dataset': 'qm9',
        'max_atoms': 9,
        'max_types': 5,
        'atom_list': [0, 6, 7, 8, 9],
        'valency_dict': {6:4, 7:3, 8:2, 9:1}
    },
    'zinc250k': {
        'dataset': 'zinc250k',
        'max_atoms': 38,
        'max_types': 10,
        'atom_list': [0, 6, 7, 8, 9, 15, 16, 17, 35, 53],
        'valency_dict': {6:4, 7:3, 8:2, 9:1, 15:3, 16:2, 17:1, 35:1, 53:1}
    },
    'moses': {
        'dataset': 'moses',
        'max_atoms': 27,
        'max_types': 8,
        'atom_list': [0, 6, 7, 8, 9, 16, 17, 35],
        'valency_dict': {6:4, 7:3, 8:2, 9:1, 16:2, 17:1, 35:1}
    },
    'guacamol': {
        'dataset': 'guacamol',
        'max_atoms': 88,
        'max_types': 13,
        'atom_list': [0, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53]
    },
    'polymer': {
        'dataset': 'polymer',
        'max_atoms': 122,           # measured on train_data
        'max_types': 8,
        'atom_list': [0, 6, 7, 8, 9, 14, 15, 16]
    }
}

# Moses Atoms - C:6, N:7, S:16, O:8, F:9, Cl:17, Br:35, H:1
# Guacamol Atoms - C:6, N:7, O:8, F:9, B:5, Br:35, Cl:17, I:53, P:15, S:16, Se:34, Si:14, H:1

def download_qm9(dir='data/', order='canonical'):
    if os.path.isdir(dir) != True:
        os.makedirs(dir)

    file = f'{dir}qm9'
    url = 'https://raw.githubusercontent.com/divelab/DIG_storage/main/ggraph/qm9_property.csv'

    print('Downloading and preprocessing the QM9 dataset.')

    urllib.request.urlretrieve(url, f'{file}.csv')
    preprocess(file, 'smile', MOLECULAR_DATASETS['qm9']['max_atoms'], MOLECULAR_DATASETS['qm9']['atom_list'], order)
    os.remove(f'{file}.csv')

    print('Done.')

def download_zinc250k(dir='data/', order='canonical'):
    if os.path.isdir(dir) != True:
        os.makedirs(dir)

    file = f'{dir}zinc250k'
    url = 'https://raw.githubusercontent.com/divelab/DIG_storage/main/ggraph/zinc250k_property.csv'

    print('Downloading and preprocessing the Zinc250k dataset.')

    urllib.request.urlretrieve(url, f'{file}.csv')
    preprocess(file, 'smile', MOLECULAR_DATASETS['zinc250k']['max_atoms'], MOLECULAR_DATASETS['zinc250k']['atom_list'], order)
    os.remove(f'{file}.csv')

    print('Done.')

def download_moses(dir='data/', order='canonical'):
    # https://github.com/cvignac/DiGress/blob/main/src/datasets/moses_dataset.py
    if os.path.isdir(dir) != True:
        os.makedirs(dir)

    file = f'{dir}moses'
    train_url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/train.csv'
    test_url  = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/test.csv'
    # test_url  = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/test_scaffolds.csv'

    # NOTE: Downloading just train split so far.
    urllib.request.urlretrieve(train_url, f'{file}.csv')
    preprocess(file, 'SMILES', MOLECULAR_DATASETS['moses']['max_atoms'], MOLECULAR_DATASETS['moses']['atom_list'], order)
    os.remove(f'{file}.csv')

    print('Done.')

def download_guacamol(dir='data/', order='canonical'):
    # https://github.com/cvignac/DiGress/blob/main/src/datasets/guacamol_dataset.py
    if os.path.isdir(dir) != True:
        os.makedirs(dir)

    file = f'{dir}guacamol'
    train_url = 'https://figshare.com/ndownloader/files/13612760'
    valid_url = 'https://figshare.com/ndownloader/files/13612766'
    test_url = 'https://figshare.com/ndownloader/files/13612757'

    # NOTE: Downloading just train split so far.
    urllib.request.urlretrieve(train_url, f'{file}.csv')
    preprocess(file, None, MOLECULAR_DATASETS['guacamol']['max_atoms'], MOLECULAR_DATASETS['guacamol']['atom_list'], order)
    os.remove(f'{file}.csv')

    print('Done.')

def download_polymer(dir='data/', order='canonical'):
    # https://github.com/wengong-jin/hgraph2graph/tree/master/data/polymers
    if os.path.isdir(dir) != True:
        os.makedirs(dir)

    file = f'{dir}polymer'
    train_url = 'https://raw.githubusercontent.com/wengong-jin/hgraph2graph/refs/heads/master/data/polymers/train.txt'
    valid_url = 'https://raw.githubusercontent.com/wengong-jin/hgraph2graph/refs/heads/master/data/polymers/valid.txt'
    test_url  = 'https://raw.githubusercontent.com/wengong-jin/hgraph2graph/refs/heads/master/data/polymers/test.txt'

    # NOTE: Downloading just train split so far.
    urllib.request.urlretrieve(train_url, f'{file}.csv')
    preprocess(file, None, MOLECULAR_DATASETS['polymer']['max_atoms'], MOLECULAR_DATASETS['polymer']['atom_list'], order)
    os.remove(f'{file}.csv')

    print('Done.')

def perm_molecule(mol, p, max_atom, atom_list):
    Chem.Kekulize(mol)
    x, a = mol2g(mol, max_atom, atom_list)
    x, a = permute_graph(x, a, p)
    return x, a, g2mol(x, a, atom_list)

def reorder_molecule(x, a, mol, order, max_atom, atom_list):
    n = mol.GetNumAtoms()
    match order:
        case 'unordered':
            s = Chem.MolToSmiles(mol, kekuleSmiles=True, canonical=False)

        case 'canonical':
            s = Chem.MolToSmiles(mol, kekuleSmiles=True, canonical=True)
            mol = Chem.MolFromSmiles(s)
            Chem.Kekulize(mol)
            x, a = mol2g(mol, max_atom, atom_list)

        case 'bft':
            p = breadth_first_order(csr_matrix((a > 0).to(torch.int8)), 0, directed=False, return_predecessors=False).tolist() + list(range((x > 0).sum(), max_atom))
            x, a, mol = perm_molecule(mol, p, max_atom, atom_list)
            s = Chem.MolToSmiles(mol, kekuleSmiles=True, canonical=False)

        case 'dft':
            p = depth_first_order(csr_matrix((a > 0).to(torch.int8)), 0, directed=False, return_predecessors=False).tolist() + list(range((x > 0).sum(), max_atom))
            x, a, mol = perm_molecule(mol, p, max_atom, atom_list)
            s = Chem.MolToSmiles(mol, kekuleSmiles=True, canonical=False)

        case 'rcm':
            p = reverse_cuthill_mckee(csr_matrix((a > 0).to(torch.int8))).tolist()
            x, a, mol = perm_molecule(mol, p, max_atom, atom_list)
            s = Chem.MolToSmiles(mol, kekuleSmiles=True, canonical=False)

        case 'rand':
            rand_perm = torch.randperm(max_atom)
            p = torch.cat((rand_perm[rand_perm < n], torch.arange(n, max_atom)))
            x, a, mol = perm_molecule(mol, p, max_atom, atom_list)
            s = Chem.MolToSmiles(mol, kekuleSmiles=True, canonical=False)

        case _:
            os.error('Unknown order')
        
    return x, a, mol, s

def preprocess(path, smile_col, max_atom, atom_list, order='canonical'):
    if smile_col is not None:
        input_df = pandas.read_csv(f'{path}.csv', sep=',', dtype='str')
        smls_list = list(input_df[smile_col])
    else:
        input_df = pandas.read_csv(f'{path}.csv', header=None, sep=',', dtype='str')
        smls_list = list(input_df[0])

    data_list = []

    for smls in tqdm(smls_list):
        mol = Chem.MolFromSmiles(smls)
        n = mol.GetNumAtoms()

        if n > 1:
            p = torch.cat((torch.randperm(n), torch.arange(n, max_atom)))
            x, a, mol = perm_molecule(mol, p, max_atom, atom_list)
            x, a, mol, s = reorder_molecule(x, a, mol, order, max_atom, atom_list)

            data_list.append({'x': x, 'a': flatten_tril(a, max_atom), 'n': n, 's': s})

    torch.save(data_list, f'{path}_{order}.pt')

class DictDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def load_dataset(name, batch_size, split, seed=0, dir='data/', order='canonical'):
    x = DictDataset(torch.load(f'{dir}{name}_{order}.pt', weights_only=True))

    torch.manual_seed(seed)
    x_trn, x_val, x_tst = torch.utils.data.random_split(x, split)

    loader_trn = torch.utils.data.DataLoader(x_trn, batch_size=batch_size, num_workers=2, shuffle=False, pin_memory=True)
    loader_val = torch.utils.data.DataLoader(x_val, batch_size=batch_size, num_workers=2, shuffle=False, pin_memory=True)
    loader_tst = torch.utils.data.DataLoader(x_tst, batch_size=batch_size, num_workers=2, shuffle=False, pin_memory=True)

    smiles_trn = [x['s'] for x in loader_trn.dataset]
    smiles_val = [x['s'] for x in loader_val.dataset]
    smiles_tst = [x['s'] for x in loader_tst.dataset]

    return {
        'loader_trn': loader_trn,
        'loader_val': loader_val,
        'loader_tst': loader_tst,
        'smiles_trn': smiles_trn,
        'smiles_val': smiles_val,
        'smiles_tst': smiles_tst
    }


if __name__ == '__main__':
    RDLogger.DisableLog('rdApp.*')
    torch.set_printoptions(threshold=10_000, linewidth=200)

    download = True
    dataset = 'zinc250k'
    orders = ['bft']

    for order in orders:
        if download:
            match dataset:
                case 'qm9':
                    download_qm9(order=order)
                case 'zinc250k':
                    download_zinc250k(order=order)
                case 'moses':
                    download_moses(order=order)
                case 'guacamol':
                    download_guacamol(order=order)
                case 'polymer':
                    download_polymer(order=order)
                case _:
                    os.error('Unsupported dataset.')

        loaders = load_dataset(dataset, 100, split=[0.8, 0.1, 0.1], order=order)

        # x = [e['x'] for e in loader_trn.dataset]
        # a = [e['a'] for e in loader_trn.dataset]
        # s = [e['s'] for e in loader_trn.dataset]

        # x = torch.stack(x)
        # a = torch.stack(a)
        # a = unflatt_tril(a, MOLECULAR_DATASETS[dataset]['max_atoms'])

        # print(evaluate_molecules(x, a, s, MOLECULAR_DATASETS[dataset]['atom_list'], metrics_only=True, canonical=True))

    # loader_trn, loader_val = load_dataset(dataset, 100, split=[0.8, 0.2], canonical=False)

    # x = [e['x'] for e in loader_trn.dataset]
    # a = [e['a'] for e in loader_trn.dataset]
    # s = [e['s'] for e in loader_trn.dataset]

    # print(evaluate_molecules(x, a, s, MOLECULAR_DATASETS[dataset]['atom_list'], metrics_only=True, canonical=False))
