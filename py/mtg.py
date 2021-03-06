import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from functools import cached_property
import plotnine as pn
from typing import Tuple, Dict, List
import lightgbm as lgb
from scipy.special import comb
import os
import pickle as pkl


LGB_PARAMS = {'objective': 'binary', 'num_leaves': 32,
              'l2_lambda': 0.3, 'max_depth': -1}


class Splitter:
    def __init__(self, response_name: str, trn=0.8, val=0.1, tst=0.1, ignore_cols=[]) -> None:
        self.trn = trn
        self.val = val
        self.tst = tst
        self.valtst = self.val + self.tst
        self.response_name = response_name
        self.ignore_cols = ignore_cols

    def split_into_frames(self, dat) -> Dict[str, pd.DataFrame]:
        trn, valtst = train_test_split(dat, train_size=self.trn, test_size=self.valtst,
                                       random_state=12)
        val, tst = train_test_split(valtst, train_size=self.trn, test_size=self.valtst, 
                                    random_state=12)
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

    @property
    def decksA(self):
        return self.dat[[col for col in self.dat.columns if 'Deck_A_Count' in col]]

    def deck_from_game_id(self, game_id: str):
        dat = pd.DataFrame([int(x) for x in game_id.split('_')]).transpose()
        dat.columns = self.cards
        return dat
        
    def read_or_train_model_all_data(
            self, all_data_model_fpath, split_data_model_fpath):        
        try:
            with open(all_data_model_fpath, 'rb') as f:
                model = pkl.load(f)    
        except FileNotFoundError:
            split_model = self.read_or_train_model_split_data(split_data_model_fpath)
            rounds = split_model.best_iteration_    
            model = lgb.LGBMClassifier(**LGB_PARAMS, num_boost_round=rounds, 
                                       random_state=12)
            sets = self.dsets.sets
            X = pd.concat([sets['trn']['X'], 
                           sets['val']['X'], 
                           sets['tst']['X']])
            y = np.concatenate([sets['trn']['y'],
                                sets['val']['y'], 
                                sets['tst']['y']])
            
            model.fit(X, y, eval_set=[(X, y)])
            with open(all_data_model_fpath, 'wb') as f:
                pkl.dump(model, f)
        return model

    def read_or_train_model_split_data(self, model_fpath):
        try:
            with open(model_fpath, 'rb') as f:
                model = pkl.load(f)    
        except FileNotFoundError:            
            model = lgb.LGBMClassifier(**LGB_PARAMS, num_boost_round=1000, 
                                       random_state=12)
            trn, val = self.dsets.sets['trn'], self.dsets.sets['val']
            model.fit(**self.dsets.sets['trn'], 
                    eval_set=[(trn['X'], trn['y']), (val['X'], val['y'])],
                    early_stopping_rounds=30)
            with open(model_fpath, 'wb') as f:
                pkl.dump(model, f)
        return model


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
    def best_deck(self):
        best_matchup = self.possible_decks_and_predictions.iloc[0]
        best_deck = (
            best_matchup
            [self.possible_decks.columns]
            .astype(int)
            .reset_index())
        best_deck.columns = ['card', 'n']
        cards, counts = best_deck['card'], best_deck['n']
        dat = pd.DataFrame(counts).transpose()
        dat.columns = cards
        return dat

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


def cross_join(a, b):
    return pd.merge(a.assign(k='k'),
                    b.assign(k='k'), 
                    on='k').drop('k', axis=1)


def _concat_row(r: pd.Series) -> str:
    return "_".join([str(x) for x in r])


def _assign_game_id(decks) -> None:
    decks['game_id'] = (
        decks
        .drop('game_id', axis=1, errors='ignore')
        .apply(_concat_row, axis=1))


class GameScorer:
    saved_scores_fpath = '../cache/DIFFERENTIATOR_saved_decks.csv'

    def __init__(self, possible_decks: pd.DataFrame, model, differentiator: str) -> None:
        self.saved_scores_fpath = self.saved_scores_fpath.replace(
            'DIFFERENTIATOR', differentiator)
        self.possible_decks = possible_decks
        self.model = model
        self.positive_class_idx = np.argmax(model.classes_)
        self.possible_decks_opponent = self.get_possible_opponent_decks()
        _assign_game_id(self.possible_decks)
        self._remove_already_checked_game_ids()

    def _remove_already_checked_game_ids(self) -> None:
        if os.path.exists(self.saved_scores_fpath):
            already_checked_ids = pd.read_csv(self.saved_scores_fpath)['game_id']
            self.possible_decks = self.possible_decks[
                ~self.possible_decks['game_id'].isin(already_checked_ids)]

    def get_possible_opponent_decks(self):
        possible_decks_opponent = self.possible_decks.drop('game_id', errors='ignore', axis=1)
        possible_decks_opponent.columns = [
            c.replace('A_Count', 'B_Count') 
            for c in possible_decks_opponent.columns]
        return possible_decks_opponent

    def score_chunks(self, chunk_nrow: int, max_chunks: int) -> pd.DataFrame:
        chunk_row_starts = range(0, self.possible_decks_opponent.shape[0], chunk_nrow)
        chunk_row_starts = list(chunk_row_starts[:max_chunks])
        chunks = []
        for chunk_row_start in chunk_row_starts:
            chunk_decks = self.possible_decks.iloc[
                chunk_row_start:(chunk_row_start + chunk_nrow), :]
            chunk = self._get_avg_score(chunk_decks)
            chunks.append(chunk)
            self.record_score_for_chunk(chunk)
        chunk_dat = pd.concat(chunks)        
        return chunk_dat

    def score_deck(self, deck: pd.DataFrame):
        deck = deck.copy()
        _assign_game_id(deck)
        return self._get_avg_score(deck)

    def record_score_for_chunk(self, scored_chunk: pd.DataFrame) -> None:
        if not os.path.exists(self.saved_scores_fpath):
            with open(self.saved_scores_fpath, 'w+') as f:
                f.write("game_id,p_avg\n")
        with open(self.saved_scores_fpath, 'a') as f:
            for r in scored_chunk.itertuples():
                f.write(f"{r.game_id},{r.p_avg:.5f}\n")

    @property
    def best_row(self):
        return (
            pd.read_csv(self.saved_scores_fpath)
            .sort_values('p_avg', ascending=False)
            .iloc[0])

    @property
    def best_deck(self):
        return self.best_row['game_id']

    def best_score(self):
        return self.best_row['p_avg']

    def _score_all_matches_chunk(self, possible_deckA_chunk: pd.DataFrame) -> pd.DataFrame:
        possible_matches_chunk = cross_join(possible_deckA_chunk,
                                            self.possible_decks_opponent)
        X = possible_matches_chunk.drop('game_id', axis=1)
        pred = self.model.predict_proba(X)[:, self.positive_class_idx]
        return possible_matches_chunk.assign(p=pred)
        
    def _get_avg_score(self, decks: pd.DataFrame) -> pd.DataFrame:
        return (
            self._score_all_matches_chunk(decks)
            .groupby('game_id')
            ['p'].mean()
            .reset_index(name='p_avg'))


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
                self.dat.groupby('bucket').agg({'y': np.mean, 
                                                'pred': np.mean}).reset_index(),
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


