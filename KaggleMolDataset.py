import numpy as np
import pandas as pd
import pickle

from dgl.data.chem.utils import mol_to_bigraph, \
    CanonicalAtomFeaturizer
from dgl import DGLGraph
import torch
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures, rdmolops
from rdkit import RDConfig
import os.path as osp
from glob import glob
from collections import defaultdict
from functools import partial
from xyz2mol import read_xyz_file, xyz2mol
import constants as C

xyz_filepath_list = os.listdir(C.RAW_DATA_PATH + 'structures')
xyz_filepath_list.sort()


## Functions to create the RDKit mol objects
def mol_from_xyz(filepath, add_hs=True, compute_dist_centre=False):
    """Wrapper function for calling xyz2mol function."""
    charged_fragments = True  # alternatively radicals are made

    # quick is faster for large systems but requires networkx
    # if you don't want to install networkx set quick=False and
    # uncomment 'import networkx as nx' at the top of the file
    quick = True

    atomicNumList, charge, xyz_coordinates = read_xyz_file(filepath)
    mol, dMat = xyz2mol(atomicNumList, charge, xyz_coordinates,
                        charged_fragments, quick, check_chiral_stereo=False)

    return mol, np.array(xyz_coordinates), dMat

def my_mol_to_graph(mol, graph_constructor, atom_featurizer, bond_featurizer):
    """Convert an RDKit molecule object into a DGLGraph and featurize for it.
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    graph_constructor : callable
        Takes an RDKit molecule as input and returns a DGLGraph
    atom_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for atoms in a molecule, which can be used to update
        ndata for a DGLGraph.
    bond_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for bonds in a molecule, which can be used to update
        edata for a DGLGraph.
    Returns
    -------
    g : DGLGraph
        Converted DGLGraph for the molecule
    """
    #new_order = rdmolfiles.CanonicalRankAtoms(mol)
    #mol = rdmolops.RenumberAtoms(mol, new_order)
    g = graph_constructor(mol)

    if atom_featurizer is not None:
        g.ndata.update(atom_featurizer(mol))

    if bond_featurizer is not None:
        g.edata.update(bond_featurizer(mol))

    return g


def construct_bigraph_from_mol(mol, add_self_loop=False):
    """Construct a bi-directed DGLGraph with topology only for the molecule.
    The **i** th atom in the molecule, i.e. ``mol.GetAtomWithIdx(i)``, corresponds to the
    **i** th node in the returned DGLGraph.
    The **i** th bond in the molecule, i.e. ``mol.GetBondWithIdx(i)``, corresponds to the
    **(2i)**-th and **(2i+1)**-th edges in the returned DGLGraph. The **(2i)**-th and
    **(2i+1)**-th edges will be separately from **u** to **v** and **v** to **u**, where
    **u** is ``bond.GetBeginAtomIdx()`` and **v** is ``bond.GetEndAtomIdx()``.
    If self loops are added, the last **n** edges will separately be self loops for
    atoms ``0, 1, ..., n-1``.
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    add_self_loop : bool
        Whether to add self loops in DGLGraphs.
    Returns
    -------
    g : DGLGraph
        Empty bigraph topology of the molecule
    """
    g = DGLGraph()

    # Add nodes
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)

    # Add edges
    src_list = []
    dst_list = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src_list.extend([u, v])
        dst_list.extend([v, u])
    g.add_edges(src_list, dst_list)

    if add_self_loop:
        nodes = g.nodes()
        g.add_edges(nodes, nodes)

    return g

def my_mol_to_bigraph(mol, add_self_loop=False,
                   atom_featurizer=CanonicalAtomFeaturizer(),
                   bond_featurizer=None):
    """Convert an RDKit molecule object into a bi-directed DGLGraph and featurize for it.
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    add_self_loop : bool
        Whether to add self loops in DGLGraphs.
    atom_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to CanonicalAtomFeaturizer().
    bond_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for bonds in a molecule, which can be used to update
        edata for a DGLGraph.
    Returns
    -------
    g : DGLGraph
        Bi-directed DGLGraph for the molecule
    """
    return my_mol_to_graph(mol, partial(construct_bigraph_from_mol, add_self_loop=add_self_loop),
                        atom_featurizer, bond_featurizer)
  
def bond_featurizer(mol, self_loop=True):
    """Featurization for all bonds in a molecule.
    The bond indices will be preserved.
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object
    self_loop : bool
        Whether to add self loops. Default to be False.
    Returns
    -------
    bond_feats_dict : dict
        Dictionary for bond features
    """
    bond_feats_dict = defaultdict(list)

    mol_conformers = mol.GetConformers()
    assert len(mol_conformers) == 1
    geom = mol_conformers[0].GetPositions()

    num_atoms = mol.GetNumAtoms()
    for u in range(num_atoms):
        for v in range(num_atoms):
            if u == v and not self_loop:
                continue

            e_uv = mol.GetBondBetweenAtoms(u, v)
            if e_uv is None:
                continue
            else:
                bond_type = e_uv.GetBondType()
            bond_feats_dict['e_feat'].append([
                float(bond_type == x)
                for x in (Chem.rdchem.BondType.SINGLE,
                          Chem.rdchem.BondType.DOUBLE,
                          Chem.rdchem.BondType.TRIPLE,
                          Chem.rdchem.BondType.AROMATIC)
            ])
            bond_feats_dict['distance'].append(
                np.linalg.norm(geom[u] - geom[v]))

    bond_feats_dict['e_feat'] = torch.tensor(
        np.array(bond_feats_dict['e_feat']).astype(np.float32))
    bond_feats_dict['distance'] = torch.tensor(
        np.array(bond_feats_dict['distance']).astype(np.float32)).reshape(-1, 1)

    return bond_feats_dict


class KaggleMolDataset(object):
    def __init__(self,
                 file_list=xyz_filepath_list,
                 label_filepath=C.RAW_DATA_PATH,
                 store_path=C.PROC_DATA_PATH,
                 mode='train',
                 from_raw=False,
                 mol_to_graph=my_mol_to_bigraph,
                 atom_featurizer=CanonicalAtomFeaturizer,
                 bond_featurizer=bond_featurizer):

        assert mode in ['train', 'test'], \
            'Expect mode to be train or test, got {}.'.format(mode)

        self.mode = mode

        self.from_raw = from_raw
        """
        if not from_raw:
            file_name = "%s_processed" % (mode)
        else:
            file_name = "structures"
        self.file_dir = pathlib.Path(file_dir, file_name)
        """
        self.file_list = file_list
        self.store_path = store_path
        self.label_filepath = label_filepath
        self.mol_to_graph = mol_to_graph
        self._load(mol_to_graph, atom_featurizer, bond_featurizer)

    def _load(self, mol_to_graph, atom_featurizer, bond_featurizer):
        if not self.from_raw:
            with open(osp.join(self.store_path, "%s_graphs.pkl" % self.mode), "rb") as f:
                self.graphs = pickle.load(f)
            with open(osp.join(self.store_path, "%s_labels.pkl" % self.mode), "rb") as f:
                self.labels = pickle.load(f)
        else:
            print('Start preprocessing dataset...')
            print('Start loading target file...')
            labels = pd.read_csv(self.label_filepath + self.mode + '.csv')
            print('Target file loaded!')
            cnt = 0
            dataset_size = len(labels['molecule_name'].unique())
            mol_names = labels['molecule_name'].unique()
            self.graphs, self.labels = [], []
            
            for i in range(len(self.file_list)):
                mol_name = self.file_list[i].split('/')[-1][:-4]
                if mol_name in mol_names:
                    cnt += 1
                    if cnt %10 ==0:
                        print('Processing molecule {:d}/{:d}'.format(cnt, dataset_size))
                    mol, xyz, dist_matrix = mol_from_xyz(C.RAW_DATA_PATH + \
                                                         'structures/' +\
                                                         self.file_list[i])
                    Chem.SanitizeMol(mol)
                    graph = self.mol_to_graph(mol, bond_featurizer=bond_featurizer)
                    graph.gdata = {}
                    
                    smiles = Chem.MolToSmiles(mol)
                    graph.gdata['smiles'] = smiles
                    graph.gdata['mol_name'] = mol_name

                    graph.ndata['h'] = torch.cat([graph.ndata['h'], torch.tensor(xyz).float()],
                                                 dim=1)
                    self.graphs.append(graph)
                    label = labels[labels['molecule_name'] == mol_name].drop([
                        'molecule_name',
                        'type',
                        'id'
                    ],
                        axis=1
                    )
                    self.labels.append(label)
                    
            os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
            with open(osp.join(self.store_path, "%s_graphs.pkl" % self.mode), "wb") as f:
                pickle.dump(self.graphs, f)
            with open(osp.join(self.store_path, "%s_labels.pkl" % self.mode), "wb") as f:
                pickle.dump(self.labels, f)

        print(len(self.graphs), "loaded!")

    def __getitem__(self, item):
        """Get datapoint with index
        Parameters
        ----------
        item : int
            Datapoint index
        Returns
        -------
        str
            SMILES for the ith datapoint
        DGLGraph
            DGLGraph for the ith datapoint
        Tensor of dtype float32
            Labels of the datapoint for all tasks
        """
        g, l = self.graphs[item], self.labels[item]
        return g.smile, g, l

    def __len__(self):
        """Length of the dataset
        Returns
        -------
        int
            Length of Dataset
        """
        return len(self.graphs)
