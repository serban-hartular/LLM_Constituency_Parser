from typing import Generator

import conllu_path as cp


def yield_sentences(file_list : list[str]) -> Generator[cp.Sentence, None, None]:
    for filename in file_list:
        for s in cp.iter_sentences_from_conllu(filename):
            yield s

if __name__ == "__main__":
    conllu_files = ['./conllu/en_gum-ud-dev.conllu', './conllu/en_gum-ud-train.conllu',
                    './conllu/en_gum-ud-test.conllu']
    bare_sg_nouns = []
    for s in yield_sentences(conllu_files):
        bare_sg_nouns.extend(s.search('.//[upos=NOUN !deprel=compound,conj !<[deprel=det,nmod:poss] ]'))
    lemmas = {n.sdata('lemma') for n in bare_sg_nouns}
