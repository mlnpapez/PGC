import os
import numpy as np
import pandas as pd

from pylatex import Document, Package, NoEscape
from gridsearch_cond import PATT_CONFIG

from rdkit import rdBase
rdBase.DisableLog("rdApp.error")


BACKEND_NAMES = {
    'btree': 'BT',
    'vtree': 'LT',
    'rtree': 'RT',
    'ptree': 'RT-S',
    'ctree': 'HCLT'
}

COLUMN_NAMES = [
        'Occurence (Train)', 'Valid', 'NSPDK', 'FCD', 'Unique', 'Novel', 'nAt', 'nBo']

def highlight_top3(x, type='max'):
    styles = np.array(len(x)*[None])
    if x.isnull().sum().any():
        return styles
    match type:
        case 'min':
            i = np.argsort(x)[:3]
        case 'max':
            i = np.argsort(x)[-3:][::-1]
        case _:
            os.error('Unsupported type.')

    styles[i] = [
        'color:{c1};textbf:--rwrap;',
        'color:{c2};textbf:--rwrap;',
        'color:{c3};textbf:--rwrap;'
    ]

    return styles

def latexify_style(df, path, row_names=None, column_names=None, precision=2):
    if row_names is not None:
        df.replace(row_names, inplace=True)
    if column_names is not None:
        df.rename(columns={o: n for o, n in zip(df.columns, column_names)}, inplace=True)
    df.rename(index=BACKEND_NAMES, inplace=True)

    patt_rename = {}
    for patt in df.index.levels[0]:
        occ = df.loc[patt, 'Occurence (Train)'].iloc[0]
        patt_rename[patt] = f'{patt} ({int(occ):d})'
    df.rename(index=patt_rename, inplace=True)

    df.drop(columns=['Occurence (Train)'], inplace=True)
    subset_min = ['NSPDK', 'FCD']
    subset_max = ['Valid', 'Unique', 'Novel']

    s = df.style
    idx = pd.IndexSlice

    for patt in df.index.levels[0]:
        slice_min_ = idx[idx[patt, :], idx[subset_min]]
        s.apply(highlight_top3, type='min', subset=slice_min_)
        
        slice_max_ = idx[idx[patt, :], idx[subset_max]]
        s.apply(highlight_top3, type='max', subset=slice_max_)

    s.format(precision=precision, na_rep='-')  # na_rep = nan replacement
    s.format(precision=3, na_rep='-', subset=['NSPDK'])
    s.to_latex(path, hrules=True, multicol_align='c', multirow_align='c', clines='skip-last;data')

    line = []
    with open(path, 'r') as file_data:
        lines = file_data.readlines()

        for w in lines[2].split():
            match w:
                case 'Valid' | 'Unique' | 'Novel':
                    w += r'$\uparrow$'
                case 'NSPDK' | 'FCD':
                    w += r'$\downarrow$'
            line.append(w)
        # merge two lines
        s = lines[3].split()[:3]
        line = s + line[1:]
        lines[2] = ' '.join(line)
        lines[3] = ''

        print(''.join(lines), file=open(path, 'w'))

def latexify_table(r_name, w_name, clean_tex=True):
    with open(r_name) as f:
        table = f.read()

    doc = Document(documentclass='standalone', document_options=('preview'), geometry_options={'margin': '0cm'})
    doc.packages.append(Package('booktabs'))
    doc.packages.append(Package('multirow'))
    doc.packages.append(Package('xcolor', options='table'))
    doc.packages.append(NoEscape(r'\definecolor{c1}{RGB}{27,158,119}'))
    doc.packages.append(NoEscape(r'\definecolor{c2}{RGB}{117,112,179}'))
    doc.packages.append(NoEscape(r'\definecolor{c3}{RGB}{217,95,2}'))
    doc.append(NoEscape(table))
    doc.generate_pdf(f'{w_name}', clean_tex=clean_tex)

def load_eval(evaluation_dir, dataset, model):
    path = evaluation_dir + f'{dataset}/{model}/'
    df_list = []
    for patt in PATT_CONFIG[dataset]:
        df_list.extend([pd.read_csv(path + f) for f in os.listdir(path) if patt in f])
    b_frame = pd.concat(df_list)
    return b_frame

def conditional_table(b_frame, dataset, backends):
    index_arrays = [PATT_CONFIG[dataset], backends]
    index = pd.MultiIndex.from_product(index_arrays, names=['Pattern', 'Model'])

    d_frame = pd.DataFrame(0., index=index, columns=COLUMN_NAMES)
    g_frame = b_frame.groupby(['pattern', 'backend'])

    for (idx, res_frame) in g_frame:
        d_frame.loc[idx] =  [
                            res_frame['a_occ_trn'].mean(),
                            100*res_frame['valid'].mean(),
                            res_frame['nspdk_tst'].mean(skipna=False),
                            res_frame['fcd_trn'].mean(skipna=False),
                            100*res_frame['unique'].mean(),
                            100*res_frame['novel'].mean(),
                            res_frame['nat_inc'].mean(),
                            res_frame['nbo_inc'].mean()]

    return d_frame

if __name__ == "__main__":
    evaluation_dir = '/mnt/data/density_learning/pgc/cond/'

    df_qm9 = load_eval(evaluation_dir, 'qm9', 'marg_sort')
    ourmodels_qm9 = conditional_table(df_qm9, 'qm9', BACKEND_NAMES.keys())

    latexify_style(ourmodels_qm9, 'results/conditional_qm9.tab')
    latexify_table('results/conditional_qm9.tab', 'results/conditional_qm9')


    df_zinc250k = load_eval(evaluation_dir, 'zinc250k', 'marg_sort')
    ourmodels_zinc250k = conditional_table(df_zinc250k, 'zinc250k', BACKEND_NAMES.keys())

    latexify_style(ourmodels_zinc250k, 'results/conditional_zinc250k.tab')
    latexify_table('results/conditional_zinc250k.tab', 'results/conditional_zinc250k')
