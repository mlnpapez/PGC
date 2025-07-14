import os
import re
import numpy as np
import pandas as pd

from pylatex import Document, Package, NoEscape
from utils.datasets import BASE_DIR

EVALUATION_DIR = f'{BASE_DIR}gs0/eval/'

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

COLUMN_NAMES = [
        'Model',         'Valid',           'NSPDK',           'FCD',             'Unique',          'Novel']
def baseline_models_qm9():
    rows = [
        ['GraphAF',      r'74.43$\pm$2.55', r'0.021$\pm$0.003', r' 5.27$\pm$0.40', r'88.64$\pm$2.37', r'86.59$\pm$1.95'],
        ['GraphDF',      r'93.88$\pm$4.76', r'0.064$\pm$0.000', r'10.93$\pm$0.04', r'98.58$\pm$0.25', r'98.54$\pm$0.48'],
        ['MoFlow',       r'91.36$\pm$1.23', r'0.017$\pm$0.003', r' 4.47$\pm$0.60', r'98.65$\pm$0.57', r'94.72$\pm$0.77'],
        ['EDP-GNN',      r'47.52$\pm$3.60', r'0.005$\pm$0.001', r' 2.68$\pm$0.22', r'99.25$\pm$0.05', r'86.58$\pm$1.85'],
        ['GraphEBM',     r' 8.22$\pm$2.24', r'0.030$\pm$0.004', r' 6.14$\pm$0.41', r'97.90$\pm$0.14', r'97.01$\pm$0.17'],
        ['SPECTRE',      r'87.30$\pm$n/a',  r'0.163$\pm$n/a',   r'47.96$\pm$n/a',  r'35.70$\pm$n/a',  r'97.28$\pm$n/a' ],
        ['GDSS',         r'95.72$\pm$1.94', r'0.003$\pm$0.000', r' 2.90$\pm$0.28', r'98.46$\pm$0.61', r'86.27$\pm$2.29'],
        ['DiGress',      r'99.00$\pm$0.10', r'0.005$\pm$n/a',   r' 0.36$\pm$n/a',  r'96.20$\pm$n/a',  r'33.40$\pm$n/a' ],
        ['GRAPHARM',     r'90.25$\pm$n/a',  r'0.002$\pm$n/a',   r' 1.22$\pm$n/a',  r'95.62$\pm$n/a',  r'70.39$\pm$n/a' ]
    ]
    return pd.DataFrame(rows, columns=COLUMN_NAMES)
def baseline_models_zinc250k():
    rows = [
        ['GraphAF',      r'68.47$\pm$0.99', r'0.044$\pm$0.005', r'16.02$\pm$0.48', r'98.64$\pm$0.69', r'100.00$\pm$0.00'],
        ['GraphDF',      r'90.61$\pm$4.30', r'0.177$\pm$0.001', r'33.55$\pm$0.16', r'99.63$\pm$0.01', r' 99.99$\pm$0.01'],
        ['MoFlow',       r'63.11$\pm$5.17', r'0.046$\pm$0.002', r'20.93$\pm$0.18', r'99.99$\pm$0.01', r'100.00$\pm$0.00'],
        ['EDP-GNN',      r'82.97$\pm$2.73', r'0.049$\pm$0.006', r'16.74$\pm$1.30', r'99.79$\pm$0.08', r'100.00$\pm$0.00'],
        ['GraphEBM',     r' 5.29$\pm$3.83', r'0.212$\pm$0.005', r'35.47$\pm$5.33', r'98.79$\pm$0.15', r'100.00$\pm$0.00'],
        ['SPECTRE',      r'90.20$\pm$n/a',  r'0.109$\pm$n/a',   r'18.44$\pm$n/a',  r'67.05$\pm$n/a',  r'100.00$\pm$n/a' ],
        ['GDSS',         r'97.01$\pm$0.77', r'0.019$\pm$0.001', r'14.66$\pm$0.68', r'99.64$\pm$0.13', r'100.00$\pm$0.00'],
        ['DiGress',      r'91.02$\pm$n/a',  r'0.082$\pm$n/a',   r'23.06$\pm$n/a',  r'81.23$\pm$n/a',  r'100.00$\pm$n/a' ],
        ['GRAPHARM',     r'88.23$\pm$n/a',  r'0.055$\pm$n/a',   r'16.26$\pm$n/a',  r'99.46$\pm$n/a',  r'100.00$\pm$n/a' ]
    ]
    return pd.DataFrame(rows, columns=COLUMN_NAMES)

def highlight_top3(x, type='max'):
    x = [float(s.split('$\pm$')[0]) for s in x]
    styles = np.array(len(x)*[None])
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

def format_number(match):
    number = float(match.group(1))
    number_original = match.group(1)
    if (number > 0.3 and number < 10) or (number > 99.98 and number < 100):
        return r"\phantom{0}" + number_original
    else:
        return number_original

def latexify_style(df, path, row_names=None, column_names=None):
    if row_names is not None:
        df.replace(row_names, inplace=True)
    if column_names is not None:
        df.rename(columns={o: n for o, n in zip(df.columns, column_names)}, inplace=True)

    subset_min = [('QM9','NSPDK'), ('QM9','FCD'), ('Zinc250k','NSPDK'), ('Zinc250k','FCD')]
    subset_max = [('QM9','Valid'), ('QM9','Unique'), ('QM9','Novel'), ('Zinc250k','Valid'), ('Zinc250k','Unique'), ('Zinc250k','Novel')]

    s = df.style
    s = s.apply(highlight_top3, type='min', subset=subset_min)
    s = s.apply(highlight_top3, type='max', subset=subset_max)
    s.hide()
    s.to_latex(
        path,
        # column_format='lcccccccccc',
        hrules=True,
        multicol_align='c'
    )

    line = []
    with open(path, 'r') as file_data:
        file_data = re.sub(r'(\d*\.?\d+)(?=\s*\$\\pm\$)', format_number, file_data.read())

        lines = file_data.splitlines()
        for w in lines[3].split():
            match w:
                case 'Valid' | 'Unique' | 'Novel':
                    w += r'$\uparrow$'
                case 'NSPDK' | 'FCD':
                    w += r'$\downarrow$'
            line.append(w)
        lines[3] = ' '.join(line)

        lines.insert(3, '\\cmidrule(lr){2-6}')
        lines.insert(4, '\\cmidrule(lr){7-11}')
        lines.insert(16, '\\midrule')
        print('\n'.join(lines), file=open(path, 'w'))

def latexify_table(r_name, w_name, clean_tex=True):
    with open(r_name) as f:
        table = f.read()

    doc = Document(documentclass='standalone', document_options=('preview'), geometry_options={'margin': '0cm'})
    doc.packages.append(Package('booktabs'))
    doc.packages.append(Package('xcolor', options='table'))
    doc.packages.append(NoEscape(r'\definecolor{c1}{RGB}{27,158,119}'))
    doc.packages.append(NoEscape(r'\definecolor{c2}{RGB}{117,112,179}'))
    doc.packages.append(NoEscape(r'\definecolor{c3}{RGB}{217,95,2}'))
    doc.append(NoEscape(table))
    doc.generate_pdf(f'{w_name}', clean_tex=clean_tex)


def find_best(evaluation_dir, dataset, model, backends):
    d_frame = pd.DataFrame('', index=range(len(backends.keys())), columns=COLUMN_NAMES)
    path = evaluation_dir + f'metrics/{dataset}/{model}/'

    for i, backend in enumerate(backends.keys()):
        b_frame = pd.concat([pd.read_csv(path + f) for f in os.listdir(path) if backend in f])
        g_frame = b_frame.groupby(list(filter(lambda x: x not in IGNORE, b_frame.columns)))
        a_frame = g_frame.agg({'sam_fcd_val': 'mean'})
        f_frame = g_frame.get_group(a_frame['sam_fcd_val'].idxmin())

        d_frame.loc[i] = [
            BACKEND_NAMES[backend],
            f'{ 100*f_frame["sam_valid"].mean():.2f}$\\pm${ 100*f_frame["sam_valid"].std():.2f}',
            f'{ f_frame["sam_nspdk_tst"].mean():.3f}$\\pm${ f_frame["sam_nspdk_tst"].std():.3f}',
            f'{   f_frame["sam_fcd_trn"].mean():.2f}$\\pm${   f_frame["sam_fcd_trn"].std():.2f}',
            f'{100*f_frame["sam_unique"].mean():.2f}$\\pm${100*f_frame["sam_unique"].std():.2f}',
            f'{100*f_frame["sam_novel" ].mean():.2f}$\\pm${100*f_frame["sam_novel" ].std():.2f}'
        ]

    return d_frame


if __name__ == "__main__":
    baselines_qm9 = baseline_models_qm9()
    ourmodels_qm9 = find_best(EVALUATION_DIR, 'qm9', 'marg_sort', BACKEND_NAMES)
    allmodels_qm9 = pd.concat([baselines_qm9, ourmodels_qm9], ignore_index=True)

    baselines_zinc250k = baseline_models_zinc250k()
    ourmodels_zinc250k = find_best(EVALUATION_DIR, 'zinc250k', 'marg_sort', BACKEND_NAMES)
    allmodels_zinc250k = pd.concat([baselines_zinc250k, ourmodels_zinc250k], ignore_index=True)

    allmodels = allmodels_qm9.merge(allmodels_zinc250k, how='left', on='Model', suffixes=('-x', '-y'))
    allmodels.columns = COLUMN_NAMES + COLUMN_NAMES[1:]

    columns = [('','Model')] + [('QM9', name) for name in COLUMN_NAMES[1:]] + [('Zinc250k', name) for name in COLUMN_NAMES[1:]]
    allmodels.columns = pd.MultiIndex.from_tuples(columns)
    allmodels.head()

    latexify_style(allmodels, 'results/unconditional.tab')
    latexify_table('results/unconditional.tab', 'results/unconditional')
