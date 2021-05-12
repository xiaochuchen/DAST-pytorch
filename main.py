
import re
from pathlib import Path
import json
import click
import torch

from models import DAST, Evaluator
from vocab import buildVocab, buildMixVocab, Vocabulary
from inp_pipe import (getBalancedLoader, getOnlineDataLoader,
                      getMixLoader, getTargetLoader)
from trainers import DASTTrainer, ClassifierTrainer
from constants import (IMDB_DIR, YELP_DIR, MIX_VOCAB, TAR_VOCAB, IY_CLF_DIR,
                       YELP_CLF_DIR, OUT_DIR, LOG_DIR)


def loadWeights(model, path, device):
    savelist = list(Path(path).glob("*.pt"))
    ep = -1
    gstep = 0
    if savelist:
        toload = max(savelist, key=lambda p: p.stat().st_ctime)
        print(f"loading {toload}")
        ep, gstep = re.compile("_(\d+)_(\d+)\.pt").search(str(toload)).groups()
        model.load_state_dict(torch.load(str(toload), map_location=device))
    model.to(device)
    return model, int(ep) + 1, int(gstep)


@click.group()
def main():
    pass


@main.command()
@click.argument('config_json')
def train_DAST(config_json: str):
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Detected device: {DEV}")

    params = json.loads(Path(config_json).read_text())
    frac = params['fraction']
    bz = params['batch_size']
    mlen = params['max_seq_len']
    embz = params['emb_size']
    hz = params['enc_dim']
    domz = params['domain_dim']
    stylez = params['style_label_dim']
    nfilters = params['n_filters']
    alpha = params['alpha']
    rho = params['rho']

    maxEpoch = params['max_epoch']
    prepEpoch = params['prep_epoch']
    lr = params['learning_rate']

    if not MIX_VOCAB.is_file():
        mixVocab = buildMixVocab(IMDB_DIR / "train", YELP_DIR / "train")
        mixVocab.to_file(MIX_VOCAB)
    mixVocab = Vocabulary.from_file(MIX_VOCAB)
    print(f"Mix vocabulary size: {mixVocab.size}")

    if not TAR_VOCAB.is_file():
        tarVocab = buildVocab(YELP_DIR / "train")
        tarVocab.to_file(TAR_VOCAB)
    tarVocab = Vocabulary.from_file(TAR_VOCAB)
    print(f"Target vocabulary size: {tarVocab.size}")

    srcLoader = getBalancedLoader(IMDB_DIR, "train", mixVocab, bz, mlen,
        tgtDom=False, device=DEV, frac=1.0)
    tgtLoader = getBalancedLoader(YELP_DIR, "train", mixVocab, bz, mlen,
        tgtDom=True, device=DEV, frac=frac)
    valLoader = getBalancedLoader(YELP_DIR, "valid", mixVocab, bz, mlen,
        tgtDom=True, device=DEV, frac=1.0)
    onlineLoader = getOnlineDataLoader(YELP_DIR, "online-test", mixVocab,
        bz, mlen, device=DEV)

    modDAST = DAST(mixVocab.size, embz, hz, domz, stylez, nfilters, alpha, rho, DEV)
    modDAST, ep_init, gstep = loadWeights(modDAST, OUT_DIR, DEV)

    targetClf = Evaluator(tarVocab.size, embz, nfilters)
    targetClf, *_ = loadWeights(targetClf, YELP_CLF_DIR, DEV)

    domainClf = Evaluator(mixVocab.size, embz, nfilters)
    domainClf, *_ = loadWeights(domainClf, IY_CLF_DIR, DEV)

    trainer = DASTTrainer(
        modDAST, srcLoader, tgtLoader, valLoader, onlineLoader,
        targetClf, domainClf, mixVocab, tarVocab,
        maxEpoch, prepEpoch, lr, mlen,
        OUT_DIR, LOG_DIR, DEV)
    trainer.train(ep_init, gstep)

    bleuRef, bleuOri, tranAcc, domAcc, sampleStr, recBleu = trainer.evaluate(onlineLoader)
    print(f"bleuRefOL: {bleuRef}, bleuOri: {bleuOri}, recBleu: {recBleu}")
    print(f"S-Acc: {tranAcc}, D-Acc: {domAcc}")
    Path("smallsample.txt").write_text(sampleStr)


@main.command()
@click.argument('config_json')
def train_classifier(config_json: str):
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Detected device: {DEV}")

    params = json.loads(Path(config_json).read_text())
    frac = params['fraction']
    bz = params['batch_size']
    mlen = params['max_seq_len']
    embz = params['emb_size']
    nfilters = params['n_filters']
    maxEpoch = params['max_epoch']
    lr = params['learning_rate']
    isDomain = params['is_domain']

    outpath = IY_CLF_DIR if isDomain else YELP_CLF_DIR
    vocabFile = MIX_VOCAB if isDomain else TAR_VOCAB
    if not vocabFile.is_file():
        vocab = buildMixVocab(IMDB_DIR / "train", YELP_DIR / "train") \
            if isDomain else buildVocab(YELP_DIR / "train")
        vocab.to_file(vocabFile)
    vocab = Vocabulary.from_file(vocabFile)
    print(f"Vocabulary size: {vocab.size}")

    if isDomain:
        trainLoader = getMixLoader(IMDB_DIR, YELP_DIR, "train", vocab, bz, mlen,
            device=DEV, frac=frac)
        validLoader = getMixLoader(IMDB_DIR, YELP_DIR, "valid", vocab, bz, mlen,
            device=DEV, frac=frac)
        testLoader = getMixLoader(IMDB_DIR, YELP_DIR, "test", vocab, bz, mlen,
            device=DEV, frac=frac)
    else:
        trainLoader = getTargetLoader(YELP_DIR, "train", vocab, bz, mlen,
            device=DEV, frac=frac)
        validLoader = getTargetLoader(YELP_DIR, "valid", vocab, bz, mlen,
            device=DEV, frac=frac)
        testLoader = getTargetLoader(YELP_DIR, "test", vocab, bz, mlen,
            device=DEV, frac=frac)

    model = Evaluator(vocab.size, embz, nfilters)
    model, ep_init, gstep = loadWeights(model, outpath, DEV)

    trainer = ClassifierTrainer(
        model, trainLoader, validLoader, vocab,
        maxEpoch, lr, mlen, isDomain,
        outpath, LOG_DIR)
    trainer.train(ep_init, gstep)

    acc = trainer.evaluate(testLoader)
    print(f" TEST Accu: {acc:.5f}")


if __name__ == '__main__':
    main()
