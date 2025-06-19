from __future__ import annotations
import conllu_path as cp

import dataclasses
from typing import Dict
from collections import Counter

FeatureTuple = tuple[tuple[str, tuple],...]

class FeatureDict(Dict[str, set[str]]):
    def __init__(self, d):
        super().__init__(d)
        for k in self:
            v = self[k]
            super().__setitem__(k, {v} if isinstance(v, str) else set(v))

    def __setitem__(self, key, value):
        value = {value} if isinstance(value, str) else set(value)
        super().__setitem__(key, value)
    def to_tuple(self) -> FeatureTuple: # everything must be sorted
        items = [[k, list(v)] for k,v in self.items()]
        for k, v in items:
            v.sort()
        items.sort(key=lambda t : t[0]) # sort by key
        return tuple((k, tuple(v)) for k,v in items)

    def match_score(self, other : 'FeatureDict') -> int:
        keys = set(self).intersection(set(other))
        if all([self[k].intersection(other[k]) for k in keys]):
            return len(keys)
        else:
            return -1


@dataclasses.dataclass
class WordInflection:
    lemma : str
    feats2forms : dict[FeatureTuple, Counter[str]] = dataclasses.field(default_factory=dict)
    forms2feats : dict[str, Counter[FeatureTuple]]= dataclasses.field(default_factory=dict)

    def get_form(self, feats : dict) -> Counter|None:
        feats = FeatureDict(feats)
        if feats.to_tuple() in self.feats2forms: # are the feats in the form dict?
            return self.feats2forms[feats.to_tuple()]
        # search for closest match
        max_score, max_form = -1, None
        for k,v in self.feats2forms.items():
            score = FeatureDict(k).match_score(feats)
            if score > max_score:
                max_score, max_form = score, v
        return max_form

    def get_most_common_form(self, feats : dict) -> str|None:
        form = self.get_form(feats)
        if form is None:
            return None
        return form.most_common(1)[0][0]

    def get_modified_form(self, feats : dict, modif : dict) -> str|None:
        return self.get_most_common_form(feats | modif)


    def add_form(self, form : str, feats : dict):
        feats = FeatureDict(feats)
        key = feats.to_tuple()
        if key in self.feats2forms:
            self.feats2forms[key].update([form])
        else:
            self.feats2forms[key] = Counter([form])
        if form in self.forms2feats:
            self.forms2feats[form].update([key])
        else:
            self.forms2feats[form] = Counter([key])

def adj_get_case(adj : cp.Tree) -> set[str]:
    if adj.data('feats.Case'):
        return adj.data('feats.Case')
    if adj.sdata('deprel') == 'amod' and adj.parent:
        parent = adj.parent
        if parent.data('feats.Case'):
            return parent.data('feats.Case')
        if parent.data('misc.Case'):
            return parent.data('misc.Case')
        return {'Nom', 'Acc'}
    if cp.Search('/[deprel=cop]').match(adj): # nume predicativ
        return {'Nom', 'Acc'}
    return {'Nom', 'Acc'}

def capitalize(orig : str, changed : str) -> str:
    if orig.isupper():
        return changed.upper()
    if orig[0].isupper():
        return changed[0].upper() + changed[1:]
    return changed

def generate_changed_sentence(node : cp.Tree, changed_form : str) -> str:
    new_sentence = ''
    projection = node.sentence().projection()
    projection.sort(key=lambda n : n.id())
    for tok in projection:
        space_after = '' if tok.sdata('misc.SpaceAfter') == 'No' else ' '
        if tok != node:
            new_sentence += tok.sdata('form') + space_after
            continue
        new_sentence += capitalize(tok.sdata('form'), changed_form) + space_after
    return new_sentence

@dataclasses.dataclass
class BadSentenceDatum:
    bad_sentence : str
    good_sentence : str
    start_index : int
    bad_word : str
    good_word : str
    word_lemma : str
    word_pos : str
    error_type : str
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

def get_bad_gender_root(node : cp.Tree) -> cp.Tree:
    upos = node.sdata('upos')
    if upos == 'ADJ':
        if node.sdata('deprel') == 'amod':
            return node.parent
        nsubj : list[cp.Tree] = cp.Search('/[deprel~nsubj]').match(node)
        if nsubj and nsubj[0].data('feats.Gender'):
            return node
        return None
    if upos == 'DET':
        if node.sdata('lemma') == 'al':
            parent = node.parent
            if not parent or parent.sdata('deprel') not in ('nmod', 'nummod'):
                return None
            return parent.parent
    elif upos == 'NUM':
        if node.sdata('deprel') != 'nummod' or node.id() > node.parent.id(): # comes after
            return None

    return node.parent


def generate_bad_sentence_datum(node : cp.Tree, changed_form : str, error_type : str,
                                local_root = False) -> BadSentenceDatum:
    bad_sentence = ''
    good_sentence = ''
    if local_root:
        projection = get_bad_gender_root(node)
        if not projection:
            return None
        projection = projection.projection()
    else:
        projection = node.sentence().projection()
    projection.sort(key=lambda n : n.id())
    good_word = node.sdata('form')
    bad_word = capitalize(good_word, changed_form)
    start_index = -1

    for tok in projection:
        space_after = '' if tok.sdata('misc.SpaceAfter') == 'No' else ' '
        good_sentence += tok.sdata('form') + space_after
        if tok != node:
            bad_sentence += tok.sdata('form') + space_after
            continue
        start_index = len(bad_sentence)
        bad_sentence += bad_word + space_after

    return BadSentenceDatum(bad_sentence=bad_sentence, good_sentence=good_sentence,
                            start_index=start_index, bad_word=bad_word, good_word=good_word,
                            word_lemma=node.sdata('lemma'), word_pos=node.sdata('upos'),
                            error_type=error_type)


if __name__ == "__main__":

    doc = cp.Doc.from_conllu('./conllu/rrt-all.3.1.annot-uid.conllu')
    words = list(doc.search('.//[upos=ADJ,DET | (upos=NUM feats.NumForm=Word) ]'))
    inflection_tables = {'ADJ':{}, 'DET':{}, 'NUM':{}}
    for w in words:
        lemma = w.sdata('lemma') #.lower()
        upos = w.sdata('upos')
        inflection_table = inflection_tables[upos]
        if lemma not in inflection_table:
            inflection_table[lemma] = WordInflection(lemma)
        inflection_table[lemma].add_form(w.sdata('form').lower(), w.data('feats').to_dict())

    found, not_found = 0, 0

    bad_sentences = []

    for w in words:
        feats = w.data('feats').to_dict()
        upos = w.sdata('upos')
        if 'Gender' not in feats:
            continue
        other_gender = {'Masc'} if 'Fem' in feats['Gender'] else {'Fem'}
        if upos == 'ADJ' and 'Case' not in feats:
            feats['Case'] = adj_get_case(w)
        inflection_table = inflection_tables[upos]
        other_form = inflection_table[w.sdata('lemma')].get_modified_form(feats, {'Gender':other_gender})
        if other_form:
            found += 1
            datum = generate_bad_sentence_datum(w, other_form, 'Gender', True)
            if datum:
                print(datum.good_sentence, datum.bad_sentence)
                bad_sentences.append(datum)
        else:
            not_found += 1
