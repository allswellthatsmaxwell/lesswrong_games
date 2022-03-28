import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from functools import cached_property
from typing import Tuple, Dict, List

class Splitter:
    def __init__(self, trn=0.8, val=0.1, tst=0.1) -> None:
        self.trn = trn
        self.val = val
        self.tst = tst
        self.valtst = self.val + self.tst

    def split_into_frames(self, dat) -> Dict[str, pd.DataFrame]:
        trn, valtst = train_test_split(dat, train_size=self.trn, test_size=self.valtst)
        val, tst = train_test_split(valtst, train_size=self.trn, test_size=self.valtst)
        return {'trn': trn, 'val': val, 'tst': tst}

    def split_into_Xy(self, dat, response_name: str, ignore_cols=[]) -> Dict[str, Dict[str, np.ndarray]]:        
        dsets = {}
        for name, subdat in self.split_into_frames(dat).items():
            X = subdat.drop(ignore_cols + [response_name], axis=1)
            y = subdat[response_name]
            dsets[name] = {'X': X, 'y': y}
        return dsets


class PastGames:
    response_name = 'Deck_A_Win?'
    ignore_cols = ['Game ID', 'Deck_B_Win?']

    def __init__(self, datapath = "../data/mtg_output.csv", splitter=Splitter()) -> None:
        self.dat = pd.read_csv(datapath)
        self.dsets = splitter.split_into_Xy(self.dat, self.response_name, self.ignore_cols)

    @cached_property
    def cards(self) -> List[str]:
        return [c.replace("_Deck_A_Count", "") 
                for c in self.dat.columns 
                if 'Deck_A' in c and '?' not in c]
