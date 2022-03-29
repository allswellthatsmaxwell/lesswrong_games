import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from functools import cached_property
import plotnine as pn
from typing import Tuple, Dict, List
import lightgbm as lgb
from scipy.special import comb

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
    cache_path = '../cache/possible_games.csv.gz'

    def __init__(self, past_games: PastGames) -> None:
        self.past_games = past_games
        self.ncards = len(self.past_games.cards)

    def get_decks(self):
        try:
            return pd.read_csv(self.cache_path)
        except FileNotFoundError:
            print(f"Didn't find {self.cache_path}; generating decks from scratch.")
            decks = self._get_decks()
            decks.to_csv(self.cache_path, compression='gzip', index=False)
            return decks

    
    def _get_decks(self) -> pd.DataFrame:
        nchoices = self.ncards
        ncards = self.ncards
        self.deckset = DeckSet()
        self._decks(nchoices, ncards - 1, [0])
        decks = pd.DataFrame(self.deckset.decks, columns=self.past_games.cards)
        assert decks.shape[0] == comb(nchoices, ncards, repetition=True, exact=True)
        return decks

    def _decks(self, budget: int, characters_remaining: int, curr_deck: List[int]):        
        if (characters_remaining == 0 & budget > 0) or budget < 0 or characters_remaining < 0:
            return 
        # If the deck is full, we get 0 of the remaining positions filled
        elif budget == 0:
            self.deckset.add(curr_deck + [0]*characters_remaining)
        else:
            # spend a budget point here; advance characters
            self._decks(budget - 1, characters_remaining - 1,  curr_deck + [1])
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
        

class DeckDefeater:
    def __init__(self, model, possible_decks: pd.DataFrame, rival_deck: pd.DataFrame) -> None:
        self.model = model
        self.possible_decks = possible_decks
        self.rival_deck = rival_deck
        assert rival_deck.shape[0] == 1
        assert len(model.classes_) == 2
        self.positive_class_idx = np.argmax(model.classes_)

    @property
    def possible_games(self) -> pd.DataFrame:
        self.possible_decks.columns = [c.replace('_Deck_A_Count', '') + '_Deck_A_Count' 
                                       for c in self.possible_decks.columns]
        self.rival_deck.columns = [c.replace('_Deck_B_Count', '') + '_Deck_B_Count' 
                                   for c in self.rival_deck.columns]
        possible_games = (
            pd.merge(self.possible_decks.assign(k='k'), 
                     self.rival_deck.assign(k='k'), 
                     on='k')
            .drop('k', axis=1)
            .astype(int))
        return possible_games

    @cached_property
    def possible_decks_and_predictions(self):
        possible_probas = self.model.predict_proba(self.possible_games)
        return (
            self.possible_games
            .assign(p=possible_probas[:, self.positive_class_idx])
            .sort_values('p', ascending=False)
            [list(self.possible_decks.columns) + ['p']]
            .reset_index()
            .drop('index', axis=1))

    @cached_property
    def mean_win_proba(self):
        return self.possible_decks_and_predictions['p'].mean()

    @cached_property
    def best_deck_str(self):
        best_matchup = self.possible_decks_and_predictions.iloc[0]
        best_deck = (
            best_matchup
            [self.possible_decks.columns]
            .astype(int)
            .reset_index())
        best_deck.columns = ['card', 'n']
        win_proba = best_matchup['p']
        return (
            '\n'.join(best_deck.apply(lambda r: f"{r['card']}: {r['n']}", axis=1)) +
            '\n\n' + f"win probability: {win_proba:.0%}")

    
        

class CalibrationAnalyzer:
    def __init__(self, y, pred, bucket_span=5) -> None:
        assert len(y) == len(pred)
        self.y = y
        self.pred = pred
        self.bucket_span = bucket_span
    
    @cached_property
    def dat(self):
        return (pd.DataFrame({'y': self.y, 'pred': self.pred})
                .assign(bucket=lambda d: (d['pred'] * 100 // self.bucket_span).astype(int)))

    @cached_property
    def buckets_dat(self):
        return (
            pd.merge(
                self.dat.groupby('bucket').agg({'y': np.mean, 'pred': np.mean}).reset_index(),
                self.dat.groupby('bucket').agg({'y': len}).reset_index(),
                on='bucket')
            .rename({'y_x': 'y_true', 'y_y': 'n'}, axis=1))

    @property
    def plot(self):
        breaks = np.arange(0, 1.001, self.bucket_span / 100)
        axis_kwargs = dict(breaks=breaks, limits=(0, 1),
                           labels=lambda ar: [int(x*100) for x in ar])
        return (
            pn.ggplot(self.buckets_dat.assign(group='group'),
                        pn.aes(y='y_true', x='pred', group='group')) +
            pn.geom_point(color='red') +
            pn.geom_line(color='blue') +
            pn.geom_abline(slope=1, intercept=0) +
            pn.scale_x_continuous(**axis_kwargs) +
            pn.scale_y_continuous(**axis_kwargs) +
            pn.theme_bw() +
            pn.theme(figure_size=(5, 5), panel_grid_minor=pn.element_blank(),
                     axis_text_y=pn.element_text(size=8), 
                     axis_text_x=pn.element_text(angle=0, size=6.5))                     
        )