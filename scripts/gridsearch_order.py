import os
import pandas as pd

from pylatex import Document, TikZ, NoEscape
from gridsearch_evaluate import IGNORE
from utils.datasets import BASE_DIR

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

def find_best(evaluation_dir, dataset, model):
    path = evaluation_dir + f'metrics/{dataset}/{model}/'
    b_frame = pd.concat([pd.read_csv(path + f) for f in os.listdir(path)])
    g_frame = b_frame.groupby(['backend', 'order'])

    f_frame = []
    for df in g_frame:
        df[1].dropna(axis=1, how='all', inplace=True)
        gf = df[1].groupby(list(filter(lambda x: x not in IGNORE, df[1].columns)))
        af = gf.agg({'sam_fcd_val': 'mean'})
        ff = gf.get_group(af['sam_fcd_val'].idxmin())
        f_frame.append(ff[['backend', 'order', 'sam_valid', 'sam_nspdk_tst', 'sam_fcd_trn', 'sam_unique', 'sam_novel']])

    f_frame = pd.concat(f_frame).groupby(['backend', 'order'])
    f_frame_m = f_frame.mean()
    f_frame_s = f_frame.std()

    return f_frame_m, f_frame_s

def nextgrouplot(pic, data_m, data_s, ylabel, args=None):
    ngp = f'\\nextgroupplot[xlabel={{Ordering (-)}}, ylabel={{{ylabel} (-)}}, bar width=4pt,'
    if args is not None:
        ngp += f', {args}]'
    else:
        ngp += r']'

    pic.append(NoEscape(ngp))
    for i, (m, s) in enumerate(zip(data_m.groupby(level=0), data_s.groupby(level=0))):
        pic.append(NoEscape(f'\\addplot+[fill=c{i}, draw=none, error bars/.cd, y dir=both, y explicit] coordinates {{' + ' '.join(f'({k}, {v}) +- ({-dev},{dev})' for (k, v), dev in zip(m[1].droplevel(0).to_dict().items(), s[1].droplevel(0).to_list())) + '};'))

if __name__ == "__main__":
    model = 'marg_sort'
    dataset = 'qm9'

    if dataset == 'qm9':
        ylim_nspdk = 0.02
        ylim_fcd = 5.0
        yshift = ''
    elif dataset == 'zinc250k':
        ylim_nspdk = 0.1
        ylim_fcd = 35.0
        yshift = r'y label style={at={(-0.23,0.5)}}'
    else:
        raise 'Unknown dataset'

    doc = Document(documentclass='standalone', document_options=('preview'), geometry_options={'margin': '1cm'})
    doc.packages.append(NoEscape(r'\usepackage{pgfplots}'))
    doc.packages.append(NoEscape(r'\pgfplotsset{compat=1.18}'))
    doc.packages.append(NoEscape(r'\usepgfplotslibrary{groupplots}'))

    doc.packages.append(NoEscape(r'\definecolor{c0}{RGB}{27,158,119}'))
    doc.packages.append(NoEscape(r'\definecolor{c1}{RGB}{117,112,179}'))
    doc.packages.append(NoEscape(r'\definecolor{c2}{RGB}{217,95,2}'))
    doc.packages.append(NoEscape(r'\definecolor{c3}{RGB}{231,41,138}'))
    doc.packages.append(NoEscape(r'\definecolor{c4}{RGB}{230,171,2}'))
    doc.packages.append(NoEscape(r'\definecolor{c5}{RGB}{166,118,29}'))

    frame_m, frame_s = find_best(EVALUATION_DIR, dataset, model)

    with doc.create(TikZ()) as pic:
        pic.append(NoEscape(r'\pgfplotsset{every tick label/.append style={font=\footnotesize}}'))
        pic.append(NoEscape(
            r'\begin{groupplot}[' +
                r'group style={group size=1 by 5, horizontal sep=55pt, vertical sep=35pt},' +
                r'xtick=data,' +
                # r'enlarge x limits=0.2,' +
                r'ybar=0pt,' +
                r'height=4.5cm,' +
                r'width=7cm,' +
                r'ymajorgrids,' +
                r'symbolic x coords={' + ', '.join(x for x in ORDER_NAMES.keys()) + r'},' +
                r'ymin=0,' +
                r'ymax=1.1,' +
                r'legend columns=-1,' +
                r'legend entries={' + ', '.join(f'\\strut {BACKEND_NAMES[x]}' for x in frame_m.index.levels[0]) + r'},' +
                r'legend to name=named,' +
                r'legend style={fill=none,draw=none,column sep=3pt},' +
                r'label style={font=\footnotesize},' +
                r'xticklabels={BFT, MCA, DFT, RCM, Random},' +
            r']'
        ))

        nextgrouplot(pic, frame_m['sam_valid'],     frame_s['sam_valid'],     r'Valid $\uparrow$')
        nextgrouplot(pic, frame_m['sam_unique'],    frame_s['sam_unique'],    r'Unique $\uparrow$')
        nextgrouplot(pic, frame_m['sam_novel'],     frame_s['sam_novel'],     r'Novel $\uparrow$')
        nextgrouplot(pic, frame_m['sam_fcd_trn'],   frame_s['sam_fcd_trn'],   r'FCD $\downarrow$',   f'ymax={ylim_fcd}')
        nextgrouplot(pic, frame_m['sam_nspdk_tst'], frame_s['sam_nspdk_tst'], r'NSPDK $\downarrow$', f'ymax={ylim_nspdk}, ' + yshift)

        pic.append(NoEscape(r'\end{groupplot}'))

        pic.append(NoEscape(r'\path (group c1r1.north east) -- node[above]{\ref{named}} (group c1r1.north west);'))

    doc.generate_pdf(f'results/gridsearch_order_{dataset}', clean_tex=False)
