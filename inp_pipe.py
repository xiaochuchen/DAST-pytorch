
import numpy as np
import json
import random
import string
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from collections import namedtuple
from itertools import islice, cycle
from nltk.tokenize import word_tokenize
from vocab import Vocabulary, PAD, GO, EOS
from typing import List, Iterable, Tuple


def cyclingZip(a: Iterable, b: Iterable, mxlen: int) -> Iterable[Tuple]:
    acycle = islice(cycle(a), mxlen)
    bcycle = islice(cycle(b), mxlen)
    return zip(acycle, bcycle)


class Example(namedtuple("Example",
    ["enc", "encLen", "dec", "decLen", "tar", "label", "tgtDom",
     "orig", "styref"])):
    pass


class BinaryDataset(Dataset):

    def __init__(self, data: List[Example], isOnline=False):
        super().__init__()
        pos = [e for e in data if e.label == 1]
        neg = [e for e in data if e.label == 0]
        assert len(pos) > 0 and len(neg) > 0, "Not a 2-class data set"
        self._data = pos + neg
        self._pix = range(len(pos))
        self._nix = range(len(pos), len(pos) + len(neg))
        self._isOnline = isOnline

    @property
    def pix(self):
        return self._pix

    @property
    def nix(self):
        return self._nix

    @property
    def isOnline(self):
        return self._isOnline

    def __getitem__(self, ix):
        return self._data[ix]

    def __len__(self):
        return len(self._data)


class BalancedBatchSampler(Sampler):

    def __init__(self, data: BinaryDataset, bz: int, drop_last=False):
        super().__init__(data)
        self.pix = data.pix
        self.nix = data.nix
        self.mxlen = max(len(self.pix), len(self.nix))
        assert bz > 0 and bz % 2 == 0, "Batch size should be an even number"
        self.bz = bz
        self.drop_last = drop_last

    def __iter__(self):
        pos = random.sample(self.pix, len(self.pix))
        neg = random.sample(self.nix, len(self.nix))

        batch = []
        for pi, ni in cyclingZip(pos, neg, self.mxlen):
            batch.extend([pi, ni])
            if len(batch) == self.bz:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return self.mxlen*2 // self.bz
        else:
            return (self.mxlen*2 + self.bz - 1) // self.bz


def procExample(orig: str, styref: str, score: int, vocab: Vocabulary,
                mlen: int, tgtDom: bool, device=None) -> Example:
    pad = vocab.word2id(PAD)
    go = vocab.word2id(GO)
    eos = vocab.word2id(EOS)

    label = torch.tensor(1 if score > 0 else 0, dtype=torch.float, device=device)
    ids = [vocab.word2id(w) for w in orig.split()[:mlen - 1]]

    enc = np.ones(mlen - 1, dtype=np.int)*pad
    enc[:len(ids)] = ids
    encT = torch.tensor(enc, dtype=torch.long, device=device)

    dec = np.ones(mlen, dtype=np.int)*pad
    dec[:len(ids)+1] = [go] + ids
    decT = torch.tensor(dec, dtype=torch.long, device=device)

    tar = np.ones(mlen, dtype=np.int)*pad
    tar[:len(ids)+1] = ids + [eos]
    tarT = torch.tensor(tar, dtype=torch.long, device=device)

    encLen = torch.tensor(len(ids), dtype=torch.long, device=device)
    decLen = torch.tensor(len(ids) + 1, dtype=torch.long, device=device)

    tgtDomT = torch.tensor(tgtDom, dtype=torch.float, device=device)
    return Example(encT, encLen, decT, decLen, tarT, label, tgtDomT, orig, styref)


def prepJsonData(line: str):
    data = json.loads(line)
    return data['review'], data['review'], data['score']


def prepOnlineData(line: str):
    orig, styref = line.split("\t")
    orig = orig.strip()
    styref = styref.strip()

    # fix punctuation and tokenization problems in the annotated data
    if orig[-1] != styref[-1] and orig[-1] in string.punctuation:
        styref += orig[-1]
    orig = " ".join(word_tokenize(orig))
    styref = " ".join(word_tokenize(styref))
    return orig, styref


def buildExamples(filelist, vocab, mlen, tgtDom, device, frac):
    examples = []
    for f in filelist:
        lines = [ln for ln in f.read_text().split("\n") if ln]
        k = max(int(frac * len(lines)), 1)
        samples = random.sample(lines, k)
        examples.extend([
            procExample(*prepJsonData(ln), vocab, mlen=mlen, tgtDom=tgtDom, device=device)
            for ln in samples
        ])
    return examples


def getBalancedLoader(path, mode, vocab, bz, mlen, tgtDom, device=None, frac=1.0):
    dpath = path / mode
    filelist = dpath.glob("*.txt")

    examples = buildExamples(filelist, vocab, mlen, tgtDom, device, frac)
    bds = BinaryDataset(examples)
    bdl = DataLoader(bds,
        batch_sampler=BalancedBatchSampler(bds, bz, drop_last=True))
    return bdl


def getTargetLoader(path, mode, vocab, bz, mlen, device=None, frac=1.0):
    dpath = path / mode
    filelist = dpath.glob("*.txt")

    examples = buildExamples(filelist, vocab, mlen, True, device, frac)
    bds = BinaryDataset(examples)
    bdl = DataLoader(bds, batch_size=bz, shuffle=True)
    return bdl


def getMixLoader(spath, tpath, mode, vocab, bz, mlen, device=None, frac=1.0):
    dpath = spath / mode
    sfiles = dpath.glob("*.txt")
    srcExamples = buildExamples(sfiles, vocab, mlen, False, device, frac)

    dpath = tpath / mode
    tfiles = dpath.glob("*.txt")
    tgtExmaples = buildExamples(tfiles, vocab, mlen, True, device, frac)

    bds = BinaryDataset(srcExamples + tgtExmaples)
    bdl = DataLoader(bds, batch_size=bz, shuffle=True)
    return bdl


def getOnlineDataLoader(path, mode, vocab, bz, mlen, device=None):
    dpath = path / mode
    files = dpath.glob("reference.*")

    examples = []
    for f in files:
        score = int(f.name[-1])
        examples.extend([
            procExample(*prepOnlineData(ln), score=score, vocab=vocab,
                mlen=mlen, tgtDom=True, device=device)
            for ln in f.read_text().split("\n") if ln
        ])

    bds = BinaryDataset(examples, isOnline=True)
    bdl = DataLoader(bds, batch_size=bz)
    return bdl
