import os
import torch
import pandas as pd

from pylatex import Document, TikZ, TikZNode, TikZOptions, NoEscape

from utils.datasets import MOLECULAR_DATASETS, BASE_DIR
from utils.plot import plot_grid_unconditional

from rdkit import rdBase
rdBase.DisableLog("rdApp.error")


EVALUATION_DIR = f'{BASE_DIR}gs0/eval/'

ORDER_NAMES = {
    'bft': 'BFT',
    'canonical': 'MCA',
    'dft': 'DFT',
    'rcm': 'RCM',
    'unordered': 'Random'
}
BACKEND_NAMES = {
    'btree': 'BT',
    'vtree': 'LT',
    'rtree': 'RT',
    'ptree': 'RT-S',
    'ctree': 'HCLT'
}

IGNORE = [
    'device',
    'seed',
    'sam_valid',
    'sam_unique',
    'sam_unique_abs',
    'sam_novel',
    'sam_novel_abs',
    'sam_score',
    'sam_m_stab',
    'sam_a_stab',
    'sam_fcd_trn',
    'sam_kldiv_trn',
    'sam_nspdk_trn',
    'sam_fcd_val',
    'sam_kldiv_val',
    'sam_nspdk_val',
    'sam_fcd_tst',
    'sam_kldiv_tst',
    'sam_nspdk_tst',
    'res_valid',
    'res_unique',
    'res_unique_abs',
    'res_novel',
    'res_novel_abs',
    'res_score',
    'res_m_stab',
    'res_a_stab',
    'res_fcd_trn',
    'res_kldiv_trn',
    'res_nspdk_trn',
    'res_fcd_val',
    'res_kldiv_val',
    'res_nspdk_val',
    'res_fcd_tst',
    'res_kldiv_tst',
    'res_nspdk_tst',
    'cor_valid',
    'cor_unique',
    'cor_unique_abs',
    'cor_novel',
    'cor_novel_abs',
    'cor_score',
    'cor_m_stab',
    'cor_a_stab',
    'cor_fcd_trn',
    'cor_kldiv_trn',
    'cor_nspdk_trn',
    'cor_fcd_val',
    'cor_kldiv_val',
    'cor_nspdk_val',
    'cor_fcd_tst',
    'cor_kldiv_tst',
    'cor_nspdk_tst',
    'nll_trn_approx',
    'nll_val_approx',
    'nll_tst_approx',
    'time_sam',
    'time_res',
    'time_cor',
    'atom_list',
    'num_params',
    'model_path'
    ]


def find_best(evaluation_dir, dataset, model, metric='sam_fcd_val', maximize=False):
    path = evaluation_dir + f'metrics/{dataset}/{model}/'
    b_frame = pd.concat([pd.read_csv(path + f) for f in os.listdir(path)])
    g_frame = b_frame.groupby(['backend', 'order'])

    f_frame = []
    for df in g_frame:
        df[1].dropna(axis=1, how='all', inplace=True)
        gf = df[1].groupby(list(filter(lambda x: x not in IGNORE, df[1].columns)))
        af = gf.agg({metric: 'mean'})
        if maximize:
            ff = gf.get_group(af[metric].idxmax())
        else:
            ff = gf.get_group(af[metric].idxmin())
        f_frame.append(ff[['backend', 'order', 'model_path']])

    f_frame = pd.concat(f_frame).groupby(['backend', 'order'])
    f_frame = f_frame.first()

    return f_frame

def create_grid(df_paths, dataset, model_name, nrows=30, ncols=2, seed=0):
    atom_list = MOLECULAR_DATASETS[dataset]['atom_list']
    max_atoms = MOLECULAR_DATASETS[dataset]['max_atoms']

    for backend in BACKEND_NAMES.keys():
        for order in ORDER_NAMES.keys():
            if (backend, order) not in df_paths.index:
                print(f'Missing {model_name} {backend} model for {order} order for {dataset} dataset.')
                continue
            path_model = df_paths.loc[(backend, order)]['model_path']
            model = torch.load(path_model, weights_only=False)

            dname = f'results/gridsearch_unco_grid/{dataset}/{model_name}/{backend}/'
            fname = f'{dataset}_{model_name}_{backend}_{order}'
            os.makedirs(dname, exist_ok=True)

            torch.manual_seed(seed)
            plot_grid_unconditional(model, nrows, ncols, max_atoms, atom_list, dname=dname, fname=fname, useSVG=True)


def latexify_grid(dataset, model_name, backend):
    doc = Document(documentclass='standalone', document_options=('preview'), geometry_options={'margin': '0cm'})
    doc.packages.append(NoEscape(r'\usetikzlibrary{positioning}'))
    doc.packages.append(NoEscape(r'\usepackage{svg}'))

    m_width = 80

    with doc.create(TikZ(options=TikZOptions({}))) as pic:
        previous_position = {}
        for i, (order, order_clean) in enumerate(ORDER_NAMES.items()):
            dname = f'plots/unconditional/{dataset}/{model_name}/{backend}/'
            fname = f'{dataset}_{model_name}_{backend}_{order}.svg'
            samples_path = f'{dname}{fname}'

            pic.append(TikZNode(
                text=f'\includesvg[width={m_width}px]{{{samples_path}}}',
                options=TikZOptions({**previous_position, 'label': '{[yshift=0px, font=\large]{' + f'{order_clean}' '}}'}),
                handle=f'n{i}'))

            previous_position = {'right': f'15 px of n{i}'}

    doc.generate_tex(f'results/gridsearch_unco_grid/{dataset}_{model_name}_{backend}_grid')


if __name__ == "__main__":
    path_frame = find_best(EVALUATION_DIR, 'qm9', 'marg_sort')
    create_grid(path_frame, 'qm9', 'marg_sort', seed=0)
    for backend in BACKEND_NAMES.keys():
        latexify_grid('qm9', 'marg_sort', backend)

    path_frame = find_best(EVALUATION_DIR, 'zinc250k', 'marg_sort')
    create_grid(path_frame, 'zinc250k', 'marg_sort', seed=0)
    for backend in BACKEND_NAMES.keys():
        latexify_grid('zinc250k', 'marg_sort', backend)
