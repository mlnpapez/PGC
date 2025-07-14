import torch

from utils.datasets import load_dataset, MOLECULAR_DATASETS
from utils.graphs import unflatt_tril
from pylatex import Document, TikZ, NoEscape
from math import isclose


def nextgrouplot(pic, matrix, title, colorbar=False):
    s = []
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            # s.append(f'({i},{j},{matrix[i, j]})')
            s.append(f'({i+1},{j+1}) [{matrix[i, j]}]')
        s.append('\n\n')
    s = ' '.join(s)

    ngp = f'\\nextgroupplot[xlabel={title}'
    if colorbar == True:
        ngp += r',colorbar]'
    else:
        ngp += r']'

    pic.append(NoEscape(ngp))
    pic.append(NoEscape(f'\\addplot[matrix plot, point meta=explicit] coordinates {{\n' + s + '\n};'))

def markzeros(pic, matrix):
    s = []
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if isclose(matrix[i, j], 0.0, abs_tol=1e-8):
                s.append(f'\\node[text=green] at (axis cs: {i+1},{j+1}) {{$*$}};')
        s.append('\n')
    s = ''.join(s)

    pic.append(NoEscape(s))

if __name__ == "__main__":
    dataset = 'qm9'
    max_atoms = MOLECULAR_DATASETS[dataset]['max_atoms']

    loader_uno = load_dataset(dataset, 100, [0.98, 0.01, 0.01], order='unordered')
    loader_can = load_dataset(dataset, 100, [0.98, 0.01, 0.01], order='canonical')
    loader_bft = load_dataset(dataset, 100, [0.98, 0.01, 0.01], order='bft')
    loader_dft = load_dataset(dataset, 100, [0.98, 0.01, 0.01], order='dft')
    loader_rcm = load_dataset(dataset, 100, [0.98, 0.01, 0.01], order='rcm')

    a_uno = torch.stack([b['a'] for b in loader_uno['loader_trn'].dataset])
    a_can = torch.stack([b['a'] for b in loader_can['loader_trn'].dataset])
    a_bft = torch.stack([b['a'] for b in loader_bft['loader_trn'].dataset])
    a_dft = torch.stack([b['a'] for b in loader_dft['loader_trn'].dataset])
    a_rcm = torch.stack([b['a'] for b in loader_rcm['loader_trn'].dataset])

    a_uno = unflatt_tril(a_uno, max_atoms)
    a_can = unflatt_tril(a_can, max_atoms)
    a_bft = unflatt_tril(a_bft, max_atoms)
    a_dft = unflatt_tril(a_dft, max_atoms)
    a_rcm = unflatt_tril(a_rcm, max_atoms)

    a_uno = (a_uno > 0).to(torch.float).mean(dim=0)
    a_can = (a_can > 0).to(torch.float).mean(dim=0)
    a_bft = (a_bft > 0).to(torch.float).mean(dim=0)
    a_dft = (a_dft > 0).to(torch.float).mean(dim=0)
    a_rcm = (a_rcm > 0).to(torch.float).mean(dim=0)

    # a_uno = (a_uno > 0.).to(torch.float)
    # a_can = (a_can > 0.).to(torch.float)
    # a_bft = (a_bft > 0.).to(torch.float)
    # a_dft = (a_dft > 0.).to(torch.float)
    # a_rcm = (a_rcm > 0.).to(torch.float)

    doc = Document(documentclass='standalone', document_options=('preview'), geometry_options={'margin': '1cm'})
    doc.packages.append(NoEscape(r'\usepackage{pgfplots}'))
    doc.packages.append(NoEscape(r'\pgfplotsset{compat=1.18}'))
    doc.packages.append(NoEscape(r'\usepgfplotslibrary{groupplots}'))

    with doc.create(TikZ(options=NoEscape(r'font=\footnotesize'))) as pic:
        pic.append(NoEscape(
            r'\begin{groupplot}[' +
                r'group style={group size=5 by 1},' +
                r'height=3.7cm,' +
                r'width=3.7cm,' +
                r'xticklabel pos=right,' +
                r'xtick={1,3,...,9},' +
                r'ytick={1,3,...,9},' +
                r'xmin=0.5,' +
                r'ymin=0.5,' +
                f'xmax={max_atoms}.5,' +
                f'ymax={max_atoms}.5,' +
                r'colormap name=hot,' +
                r'point meta min=0.0,' +
                r'point meta max=0.6,' +
                r'colorbar style={width=5pt}' +
            r']'
        ))

        nextgrouplot(pic, a_uno, 'Random')
        markzeros(pic, a_uno)
        nextgrouplot(pic, a_bft, 'BFT')
        markzeros(pic, a_bft)
        nextgrouplot(pic, a_dft, 'DFT')
        markzeros(pic, a_dft)
        nextgrouplot(pic, a_rcm, 'RCM')
        markzeros(pic, a_rcm)
        nextgrouplot(pic, a_can, 'MCA', True)
        markzeros(pic, a_can)

        pic.append(NoEscape(r'\end{groupplot}'))

    doc.generate_pdf('results/adjacency_plot', clean_tex=False)
