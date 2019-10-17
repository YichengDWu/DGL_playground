# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 20:48:30 2019

@author: lenovo
"""

import os
import pickle
import zipfile
import pathlib
import dgl
import torch as th
import pandas as pd

class HARDataset(object):
    def __init__(self,
                 data_path = './data',
                 store_path = './proc',
                 Gamma = 50,
                 sell_loop = True,
                 from_raw = False
            ):
        
        self.data_path = data_path
        self.store_path = store_path
        self.from_raw = from_raw
        self.Gamma = 50
        self.file_name = '/dataset-har-PUC-Rio-ugulino'
        if not os.path.exists(self.data_path+self.file_name+ 'csv'):
            archive = zipfile.ZipFile(self.data_path+self.file_name+'.zip')
            archive.extractall(self.data_path)
            archive.close()
        self.njoints = 4
        self.spa_edges = list(set([(i,j) for i in range(self.njoints) \
                          for j in range(self.njoints)]))
        self.spa_edges.sort()
        self.tem_edges = [(i,i) for i in range(self.njoints)]
        self.pro = 0
        self._load()
    def frames_to_stgraph(self, frames):
        print(f"Processing stgraph {self.pro}...")
        frame_graphs = {}
        frame_graphs[('fm_0', 'connect', 'fm_0')] = self.spa_edges
        for frame in range(1, frames.shape[0]):
            frame_graphs[(f'fm_{frame-1}', 'samewith', f'fm_{frame}')] = self.tem_edges
            frame_graphs[(f'fm_{frame}', 'connect', f'fm_{frame}')] = self.spa_edges
            
            
        stgraph =  dgl.heterograph(frame_graphs)
        
        xyz = frames[['x_1', 'y1', 'z1', 
                      'x2', 'y2', 'z2',
                      'x3', 'y3', 'z3',
                      'x4', 'y4', 'z4']].values
        for frame in range(frames.shape[0]):
            #x,y,z data
            print(f"Processing frame {frame}...")
            stgraph.nodes[f'fm_{frame}'].data = xyz[frame,:].reshape(self.njoints,
                         -1)
        
        
        graph_feats = ['gender', 'age', 'how_tall_in_meters',
                       'weight', 'body_mass_index']  
        stgraph.gdata = {f:frames[f].values[0] for f in graph_feats}
        self.pro += 1
        print(stgraph)
        return stgraph, frames.iloc[0,-1]
    def _load(self):
        if not self.from_raw:
            with open(self.store_path+'/stgraphs.pkl',  "rb") as f:
                self.stgraphs = pickle.load(f)
            with open(self.store_path+'/labels.pkl',  "rb") as f:
                self.labels = pickle.load(f)
        else:
            print('Start preprocessing dataset...')
            #preprocess
            har = pd.read_csv(self.data_path+self.file_name+'.csv', sep = ';')
            har.loc[122076,'z4']=-144
            har['z4'] = har['z4'].astype('int')
            har['how_tall_in_meters'] = har['how_tall_in_meters'].map(lambda x: float(x.replace(',','.')))
            har['body_mass_index'] = har['body_mass_index'].map(lambda x: float(x.replace(',','.')))
            
            #
            self.stgraphs, self.labels = har.groupby(['user','class']).apply(self.frames_to_stgraph)
    
        with open(self.store_path+'/stgraphs.pkl', "wb") as f:
                pickle.dump(self.stgraphs, f)
        with open(self.store_path+'labels.pkl', "wb") as f:
                pickle.dump(self.labels, f)            
        
    def __getitem__(self, item):
        g, l = self.stgraphs[item], self.labels[item]
        return g, l
    
    def __len__(self):
        """Length of the dataset
        Returns
        -------
        int
            Length of Dataset
        """
        return len(self.stgraphs)