
import json
from pathlib import Path
from collections import Counter
from typing import List


PAD = '<pad>'
GO = '<go>'
EOS = '<eos>'
UNK = '<unk>'


class Vocabulary(object):

    specials = [PAD, GO, EOS, UNK]

    def __init__(self, toks: List[str]):
        self._toks = toks
        self._id2w = self.specials + toks
        self._w2id = {w: ix for ix, w in enumerate(self._id2w)}
        self._size = len(self._id2w)

    @property
    def size(self):
        return self._size

    def word2id(self, word):
        return self._w2id.get(word, self._w2id[UNK])

    def id2word(self, wid):
        if wid >= self._size:
            raise ValueError(f"Word ID our of vocab range: {wid}")
        return self._id2w[wid]

    def to_file(self, path: Path):
        path.write_text("\n".join(self._toks), encoding='utf-8')

    @classmethod
    def from_file(cls, path: Path):
        return cls(path.read_text(encoding='utf-8').split("\n"))


def procFileToBow(file: Path) -> List[str]:
    return [w for ln in file.read_text("utf-8").split("\n") if ln
            for w in json.loads(ln)['review'].split()]


def buildVocab(path: Path, min_occur=1) -> Vocabulary:
    words = [w for file in path.glob("*.txt") for w in procFileToBow(file)]
    bow = [w for w, ct in Counter(words).items() if ct >= min_occur]
    return Vocabulary(bow)


def buildMixVocab(source, target, min_occur=2):
    srcWords = [w for file in source.glob("*.txt")
                for w in procFileToBow(file)]
    tgtWords = [w for file in target.glob("*.txt")
                for w in procFileToBow(file)]

    bow = [w for w, ct in Counter(srcWords + tgtWords).items()
           if ct >= min_occur]
    return Vocabulary(bow)
