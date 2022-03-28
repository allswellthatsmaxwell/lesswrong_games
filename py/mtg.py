from codecs import ignore_errors
import enum
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from functools import cached_property
import itertools
from typing import Tuple, Dict, List
import lightgbm as lgb

class Splitter:
    def __init__(self, response_name: str, trn=0.8, val=0.1, tst=0.1, ignore_cols=[]) -> None:
        self.trn = trn
        self.val = val
        self.tst = tst
        self.valtst = self.val + self.tst
        self.response_name = response_name
        self.ignore_cols = ignore_cols

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

    def __init__(self, datapath = "../data/mtg_output.csv", 
                 splitter=None) -> None:
        if splitter is None:
            splitter = Splitter(response_name=self.response_name, 
                                ignore_cols=self.ignore_cols)
        self.dat = pd.read_csv(datapath)
        self.dsets = Datasets(self.dat, splitter)

    @cached_property
    def cards(self) -> List[str]:
        return [c.replace("_Deck_A_Count", "") 
                for c in self.dat.columns 
                if 'Deck_A' in c and '?' not in c]
    
    @cached_property
    def idxs_to_cards(self):
        return dict([(i, card) for i, card in enumerate(self.cards)])

    @cached_property    
    def cards_to_idxs(self):
        return dict([(card, i) for i, card in enumerate(self.cards)])


class PossibleDecks:
    def __init__(self, past_games: PastGames) -> None:
        self.past_games = past_games
        self.ncards = len(self.past_games.cards)

    @cached_property
    def decks(self) -> pd.DataFrame:
        nchoices = self.ncards
        ncards = self.ncards
        self.deckset = DeckSet()
        self._decks(nchoices, ncards - 1, [0])
        return pd.DataFrame(self.deckset.decks, columns=self.past_games.cards)

    def _decks(self, budget: int, characters_remaining: int, curr_deck: List[int]):        
        if (characters_remaining == 0 & budget > 0) or budget < 0 or characters_remaining < 0:
            return 
        # If the deck is full, we get 0 of the remaining positions filled
        elif budget == 0:
            self.deckset.add(curr_deck + [0]*characters_remaining)
        else:
            # spend a budget point here; advance characters
            updated_deck1 = curr_deck + [1]
            self._decks(budget - 1, characters_remaining - 1, updated_deck1)
            # spend a budget point here; do not advance characters
            updated_deck2 = curr_deck.copy()
            updated_deck2[-1] += 1
            self._decks(budget - 1, characters_remaining, updated_deck2)
            # don't spend a budget point here; advance characters
            self._decks(budget, characters_remaining - 1, curr_deck + [0])
            

class DeckSet:
    def __init__(self) -> None:
        self.deckstrs = set()

    def add(self, deck: List[int]):
        self.deckstrs.add("_".join([str(s) for s in deck]))    

    @cached_property
    def decks(self):
        return [s.split('_') for s in self.deckstrs]



class RivalDeck:
    def __init__(self, past_games: PastGames) -> None:
        self.past_games = past_games

    @cached_property
    def deck(self):
        dat = pd.DataFrame([1] * len(self.past_games.idxs_to_cards)).transpose() 
        dat.columns = self.past_games.cards
        return dat
        

    