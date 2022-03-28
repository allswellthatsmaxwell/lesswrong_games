from codecs import ignore_errors
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from functools import cached_property
from typing import Tuple, Dict, List
import lightgbm as lgb

class Splitter:
    def __init__(self, response_name: str, trn=0.8, val=0.1, tst=0.1, ignore_cols=[]) -> None:
        self.trn = trn
        self.val = val
        self.tst = tst
        self.valtst = self.val + self.tst
        self.response_name = response_name
        self.ignore_cols = []

    def split_into_frames(self, dat) -> Dict[str, pd.DataFrame]:
        trn, valtst = train_test_split(dat, train_size=self.trn, test_size=self.valtst)
        val, tst = train_test_split(valtst, train_size=self.trn, test_size=self.valtst)
        return {'trn': trn, 'val': val, 'tst': tst}

    def split_into_Xy(self, dat) -> Dict[str, Dict[str, np.ndarray]]:        
        dsets = {}
        for name, subdat in self.split_into_frames(dat).items():
            X = subdat.drop(self.ignore_cols + [self.response_name], axis=1)
            y = subdat[self.response_name]
            dsets[name] = {'X': X, 'y': y}
        return dsets

class Datasets:
    def __init__(self, dat, splitter) -> None:
        self.dat = dat
        self.splitter = splitter

    @cached_property
    def sets(self):
        return self.splitter.split_into_Xy(self.dat)

    @cached_property
    def lgb_sets(self):
        dsets = {}
        for setname, matrices in self.sets.items():
            dsets[setname] = lgb.Dataset(matrices['X'], label=matrices['y'])
        return dsets

class PastGames:
    response_name = 'Deck_A_Win?'
    ignore_cols = ['Game ID', 'Deck_B_Win?']

    def __init__(self, datapath = "../data/mtg_output.csv", splitter=None) -> None:
        if splitter is None:
            splitter = Splitter(response_name=self.response_name, ignore_cols=self.ignore_cols)
        self.dat = pd.read_csv(datapath)
        self.dsets = Datasets(self.dat, splitter)

    @cached_property
    def cards(self) -> List[str]:
        return [c.replace("_Deck_A_Count", "") 
                for c in self.dat.columns 
                if 'Deck_A' in c and '?' not in c]