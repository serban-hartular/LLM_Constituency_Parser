# This is a sample Python script.
from __future__ import annotations

import dataclasses
import enum
import itertools
import re
import string

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import conllu_path as cp



def split_by_continuity(tok_list : list[cp.Tree]) -> list[list[cp.Tree]]:
    llist = [[tok_list[0]]]
    for tok in tok_list[1:]:
        if int(tok.id()) - int(llist[-1][-1].id()) == 1: # are successive
            llist[-1].append(tok)
        else: # new sublist
            llist.append([tok])
    return llist

def is_all_punct(tok_list : list[cp.Tree]) -> bool:
    for tok in tok_list:
        if tok.sdata('deprel') != 'punct':
            return False
    return True


PUNCTUATION = ''.join(set(string.punctuation).difference('()[]{}'))

class BasicRole(enum.Enum):
    HEAD = 'head'
    COMPLEMENT = 'complement'
    ADJUNCT = 'adjunct'
@dataclasses.dataclass
class TokenSpan:
    tokens : list[cp.Tree]
    is_constituent : bool
    head : TokenSpan = None
    subunits : list[TokenSpan] = None
    basic_role : BasicRole = None
    phrase_type : str = None
    syntactic_fn : str = None
    is_continuous : bool = None
    parent : TokenSpan = None

    def __post_init__(self):
        if self.subunits:
            for subspan in self.subunits:
                subspan.parent = self
        continuity_split = split_by_continuity(self.tokens)
        continuity_split = [l for l in continuity_split if not is_all_punct(l)]
        if len(continuity_split) == 1: # is good
            self.tokens = continuity_split[0]
            self.is_continuous = True
        else:
            self.is_continuous = False

    def contains_discontinuity(self) -> list[TokenSpan]:
        if not self.subunits:
            return [] if self.is_continuous else [self]
        return list(itertools.chain.from_iterable([t.contains_discontinuity() for t in self.subunits]))


    def to_text(self, stripped : bool = False) -> str:
        s = ''.join([t.sdata('form') + ('' if t.sdata('misc.SpaceAfter') == 'No' else ' ')
                        for t in self.tokens])
        if stripped:
            s = s.strip(PUNCTUATION + string.whitespace)
        return s

    def to_flat(self):
        s = self.phrase_type + " " + self.to_text()
        return '(' + s + ")"

    def to_children(self) -> str:
        s =  '(' + self.phrase_type + ' '
        if self.subunits:
            s += ' '.join([('h' if c == self.head else '') + c.to_flat() for c in self.subunits])
        else:
            s += self.to_text()
        s += ')'
        return s

    def to_tree(self) -> str:
        if not self.subunits:
            return self.to_flat()
        s = '(' + self.phrase_type + ' '
        s += ' '.join([('h' if c == self.head else '') + c.to_tree() for c in self.subunits])
        s += ')'
        return s
    def __str__(self):
        return self.to_children()
    def __repr__(self):
        return repr(str(self))

upos2phrase = {'NOUN':'NP', 'VERB':'VP', 'PRON':'NP', 'ADJ':'AdjP', 'ADV':'AdvP', 'PROPN':'NP'}

def get_node_info(tree : cp.Tree) -> dict:
    upos, deprel = tree.sdata('upos'), tree.sdata('deprel')
    is_constituent = upos in ('NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'PROPN')
    if tree.sdata('deprel') == 'advmod' and tree.sdata('lemma') in \
            ('mai', 'și', 'chiar', 'numai', 'măcar', 'cam', 'tot', 'nu'):
        is_constituent = False
    phrase_type = upos2phrase[upos] if is_constituent else upos
    basic_role = None
    if is_constituent:
        if deprel == 'root':
            basic_role = BasicRole.HEAD
        elif re.search(r'(comp)|(dobj)|(subj)|(iobj)', upos):
            basic_role = BasicRole.COMPLEMENT
        else:
            basic_role = BasicRole.ADJUNCT
    return {'is_constituent':is_constituent, 'phrase_type':phrase_type, 'basic_role':basic_role,
            'syntactic_fn':deprel}


def constituency_split(tree : cp.Tree, tokens : list[cp.Tree] = None) -> TokenSpan:
    if tokens is None:
        tokens = tree.projection()
    basic_info = get_node_info(tree)
    basic_info['tokens'] = tokens
    children = [c for c in tree.children() if c in tokens and c.sdata('deprel') not in ('punct', 'fixed', 'flat')]
    if not children:
        return TokenSpan(**basic_info)
    # CP, PP, RP
    first = children[0]
    cop_search: list[cp.Tree] = [r for r in children if r.sdata('deprel') == 'cop']

    if (first.sdata('deprel') in ('case', 'mark') or 'Rel' in first.sdata('feats.PronType')) and\
            first.sdata('feats.PartType') != 'Inf':
        phrase_type = 'PP' if first.sdata('deprel') == 'case' else 'CP' if first.sdata('deprel') == 'mark' else 'RP'
        first_proj = first.projection()
        # head_info = get_node_info(first) | {'tokens':first_proj}
        # head = TokenSpan(**head_info)
        head = constituency_split(first)
        basic_info.update({'phrase_type':phrase_type})
        subunits = [head, constituency_split(tree, [t for t in tokens if t not in first_proj])]
        basic_info.update({'head':head, 'subunits':subunits})
        component = TokenSpan(**basic_info)
        # return component
    elif cop_search:
        if [r for r in children if r.sdata('deprel') in ('conj', 'parataxis')]: # no conjuncts
            return TokenSpan(tokens, True, None, None, basic_info['basic_role'], 'VP', basic_info['syntactic_fn'])
        cop = cop_search[0]
        cop_projection = cop.projection()
        verb_dep_roots : list[cp.Tree] = cp.Search('/[deprel~subj,obj,obl,comp,advmod,advcl]').match(tree)
        verb_dep_roots = [r for r in verb_dep_roots if r in children and r.sdata('feats.Strength') != 'Weak']
        verb_dep_proj = list(itertools.chain.from_iterable([r.projection() for r in verb_dep_roots]))

        npred_proj = [t for t in tokens if t not in verb_dep_proj and t not in cop_projection]

        constituent_roots = [tree, cop] + verb_dep_roots
        constituent_roots.sort(key=lambda t: t.id())
        constituents = []
        head_constituent = None
        for r in constituent_roots:
            if r == tree: # the noun predicative
                constituent = constituency_split(r, npred_proj)
                constituent.syntactic_fn = 'npred'
                constituents.append(constituent)
            elif r == cop: # the copulative verb
                head_constituent = TokenSpan(cop_projection, True, None, None, BasicRole.HEAD,
                                             'VP',
                                             tree.sdata('deprel'))
                constituents.append(head_constituent)
            else:
                constituents.append(constituency_split(r))
        assert head_constituent is not None
        basic_info.update({'head': head_constituent, 'subunits': constituents, 'phrase_type':'VP'})
        component = TokenSpan(**basic_info)
    else:    # basic case
        if [r for r in children if r.sdata('deprel') in ('conj', 'parataxis')]:  # no conjuncts
            return TokenSpan(tokens, True, None, None, basic_info['basic_role'], basic_info['phrase_type'], basic_info['syntactic_fn'])

        dependent_roots : list[cp.Tree] = cp.Search('/[deprel~subj,obj,obl,comp,mod,acl,advcl,parataxis]').match(tree)
        dependent_roots = [r for r in dependent_roots if r in children and r.sdata('feats.Strength') != 'Weak']
        constituent_roots =  dependent_roots + [tree]
        constituent_roots.sort(key=lambda t: t.id())
        dependents_proj = list(itertools.chain.from_iterable([d.projection() for d in dependent_roots]))
        head_proj = [t for t in tokens if t not in dependents_proj]
        head_constituent = TokenSpan(head_proj, True, None, None, BasicRole.HEAD,
                                      basic_info['phrase_type'],
                                      tree.sdata('deprel'))
        constituents = [constituency_split(r) if r != tree else head_constituent
                            for r in constituent_roots]
        if len(constituents) == 1:
            component = head_constituent
        else:
            basic_info.update({'head':head_constituent, 'subunits':constituents})
            component = TokenSpan(**basic_info)
    if component.contains_discontinuity():
        component.head = None
        component.subunits = None
        component.is_continuous = True
    return component

# def generate_training(c : TokenSpan):

@dataclasses.dataclass
class TrainingDatum:
    constituent : str
    sentence : str
    label : str
    subunits : list[str]
    head : str
    dependents : list[str]

    @staticmethod
    def from_token_span(t : TokenSpan, sentence_text : str) -> TrainingDatum:
        return TrainingDatum(constituent=t.to_text(True),
                             sentence=sentence_text,
                             label=t.phrase_type,
                             subunits=[r.to_text(True) for r in t.subunits] if t.subunits else None,
                             head=t.head.to_text(True) if t.head else None,
                             dependents=[r.to_text(True) for r in t.subunits if r != t.head] if t.subunits else None,
                        )

    @staticmethod
    def from_token_span_recursive(t : TokenSpan, sentence_text : str = None) -> list[TrainingDatum]:
        if sentence_text is None:
            sentence_text = t.to_text()
        l = [TrainingDatum.from_token_span(t, sentence_text)]
        if t.subunits:
            l += list(itertools.chain.from_iterable(
                [TrainingDatum.from_token_span_recursive(r, sentence_text) for r in t.subunits]))
        return l
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    def parse_sentence(sent: str | cp.Sentence) -> TokenSpan:
        if isinstance(sent, str):
            sent = doc.get_sentence(sent)
        return constituency_split(sent.root)

    doc = cp.Doc.from_conllu('./conllu/ro_rrt-ud-dev.conllu')

    # to exclude:
    # NumForm=Digit
    # Abbr=Yes
    # ([a-z]+)
    skip_count = 0
    sentence_parses = []
    for sentence in doc:
        # if sentence.search('.//[feats.NumForm=Digit | feats.Abbr=Yes]') or \
        if        {':'}.intersection(set(sentence.text)):
            skip_count += 1
            continue
        c = constituency_split(sentence.root)
        discontinuous = [r.parent for r in c.contains_discontinuity()]
        if discontinuous:
            print('Discontinuity at ' + sentence.sent_id + ": ", discontinuous)
            skip_count += 1
            continue
        sentence_parses.append(c)
    print('Skipped', skip_count, 'of', len(doc))
