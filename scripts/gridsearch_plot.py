import os
import pandas as pd

from pylatex import Document, TikZ, NoEscape
from utils.datasets import BASE_DIR

EVALUATION_DIR = f'{BASE_DIR}gs0/eval/'

BACKEND_NAMES = {
    'btree': 'BT',
    'vtree': 'LT',
    'rtree': 'RT',
    'ptree': 'RT-S',
    'ctree': 'HCLT'
}

# https://tikz.dev/pgfplots/reference-markers
MARKS = {
    'btree': '+',
    'vtree': '*',
    'rtree': 'o',
    'ptree': 'halfcircle',
    'ctree': 'diamond'
}

def nextgrouplot(pic, evaluation_dir, dataset, model, backends, ydata, ylabel, args=None):
    ngp = f'\\nextgroupplot[xlabel={{Number of parameters (-)}}, ylabel={{{ylabel} (-)}}'
    if args is not None:
        ngp += f', {args}]'
    else:
        ngp += r']'

    pic.append(NoEscape(ngp))
    path = evaluation_dir + f'metrics/{dataset}/{model}/'
    for i, backend in enumerate(backends):
        b_frame = pd.concat([pd.read_csv(path + f) for f in os.listdir(path) if backend in f])
        coordinates = list(b_frame[['num_params', ydata]].itertuples(index=False, name=None))
        pic.append(NoEscape(f'\\addplot [color=c{i}, mark={MARKS[backend]}, only marks] coordinates {{' + ' '.join(str(x) for x in coordinates) + '};' + f'\\addlegendentry{{{BACKEND_NAMES[backend]}}};'))


if __name__ == "__main__":
    model = 'marg_sort'
    dataset = 'qm9'
    ylim_nspdk = 0.1
    ylim_fcd = 10.0

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

    with doc.create(TikZ()) as pic:
        pic.append(NoEscape(r'\pgfplotsset{every tick label/.append style={font=\footnotesize}}'))
        pic.append(NoEscape(
            r'\begin{groupplot}[' +
                r'group style={group size=3 by 5, horizontal sep=55pt, vertical sep=35pt},' +
                r'height=5cm,' +
                r'width=6.4cm,' +
                r'xmode=log,' +
                r'ymin=0,' +
                r'ymax=1,' +
                r'legend style={font=\tiny,fill=none,draw=none,row sep=-3pt},' +
                r'legend pos=south west,' +
                r'legend cell align=left,' +
                r'label style={font=\footnotesize},' +
                r'y label style={at={(-0.12,0.5)}},' +
                r'x label style={at={(0.5,-0.09)}}' +
            r']'
        ))

        nextgrouplot(pic, EVALUATION_DIR, dataset, model, BACKEND_NAMES.keys(), 'sam_valid',     'Valid')
        nextgrouplot(pic, EVALUATION_DIR, dataset, model, BACKEND_NAMES.keys(), 'res_valid',     'Valid')
        nextgrouplot(pic, EVALUATION_DIR, dataset, model, BACKEND_NAMES.keys(), 'cor_valid',     'Valid')

        nextgrouplot(pic, EVALUATION_DIR, dataset, model, BACKEND_NAMES.keys(), 'sam_unique',    'Unique')
        nextgrouplot(pic, EVALUATION_DIR, dataset, model, BACKEND_NAMES.keys(), 'res_unique',    'Unique')
        nextgrouplot(pic, EVALUATION_DIR, dataset, model, BACKEND_NAMES.keys(), 'cor_unique',    'Unique')

        nextgrouplot(pic, EVALUATION_DIR, dataset, model, BACKEND_NAMES.keys(), 'sam_novel',     'Novel')
        nextgrouplot(pic, EVALUATION_DIR, dataset, model, BACKEND_NAMES.keys(), 'res_novel',     'Novel')
        nextgrouplot(pic, EVALUATION_DIR, dataset, model, BACKEND_NAMES.keys(), 'cor_novel',     'Novel')

        nextgrouplot(pic, EVALUATION_DIR, dataset, model, BACKEND_NAMES.keys(), 'sam_fcd_tst',   'FCD',   f'ymax={ylim_fcd}')
        nextgrouplot(pic, EVALUATION_DIR, dataset, model, BACKEND_NAMES.keys(), 'res_fcd_tst',   'FCD',   f'ymax={ylim_fcd}')
        nextgrouplot(pic, EVALUATION_DIR, dataset, model, BACKEND_NAMES.keys(), 'cor_fcd_tst',   'FCD',   f'ymax={ylim_fcd}')

        nextgrouplot(pic, EVALUATION_DIR, dataset, model, BACKEND_NAMES.keys(), 'sam_nspdk_tst', 'NSPDK', f'ymax={ylim_nspdk}, ' + r'y label style={at={(-0.23,0.5)}}')
        nextgrouplot(pic, EVALUATION_DIR, dataset, model, BACKEND_NAMES.keys(), 'res_nspdk_tst', 'NSPDK', f'ymax={ylim_nspdk}, ' + r'y label style={at={(-0.23,0.5)}}')
        nextgrouplot(pic, EVALUATION_DIR, dataset, model, BACKEND_NAMES.keys(), 'cor_nspdk_tst', 'NSPDK', f'ymax={ylim_nspdk}, ' + r'y label style={at={(-0.23,0.5)}}')

        pic.append(NoEscape(r'\end{groupplot}'))

        pic.append(NoEscape(r'\node (t1) at ($(group c1r1.center)!0.5!(group c1r1.center)+(0,2.1cm)$) {w/o resampling};'))
        pic.append(NoEscape(r'\node (t2) at ($(group c2r1.center)!0.5!(group c2r1.center)+(0,2.1cm)$) {w resampling};'))
        pic.append(NoEscape(r'\node (t3) at ($(group c3r1.center)!0.5!(group c3r1.center)+(0,2.1cm)$) {w correction};'))

    doc.generate_pdf('results/gridsearch_plot', clean_tex=False)
