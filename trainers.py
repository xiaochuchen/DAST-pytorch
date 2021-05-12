
import numpy as np
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from inp_pipe import cyclingZip


class DASTTrainer(object):

    def __init__(self, model, srcLoader, tgtLoader, valLoader, onlineLoader,
                 targetClf, domainClf, mixVocab, tarVocab,
                 maxEpoch, prepEpoch, lr, mlen,
                 outpath, logdir, device):
        self.model = model
        self.srcLoader = srcLoader
        self.tgtLoader = tgtLoader
        self.valLoader = valLoader
        self.onlineLoader = onlineLoader
        self.targetClf = targetClf
        self.domainClf = domainClf
        self.mixVocab = mixVocab
        self.tarVocab = tarVocab
        self.maxEpoch = maxEpoch
        self.prepEpoch = prepEpoch
        self.mlen = mlen
        self.outpath = outpath
        self.device = device
        self.batchnum = max(len(srcLoader), len(tgtLoader))
        self.checkpoint = 1 * self.batchnum
        self.cutAcc = 0.90
        self.optimAE = torch.optim.AdamW(list(model.gen.parameters()), lr=lr,
                                        betas=(0.5, 0.999))
        self.optimDisc = torch.optim.AdamW(list(model.discTgt.parameters()) +
                                    list(model.discSrc.parameters()), lr=lr,
                                        betas=(0.5, 0.999))
        self.optimTot = torch.optim.AdamW(list(model.parameters()), lr=lr,
                                         betas=(0.5, 0.999))
        self.writer = SummaryWriter(str(logdir))
        self.smoothie = SmoothingFunction().method4
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimAE, step_size=2, gamma=0.5, verbose=True)

    def train(self, ep_init, gstep):
        bestbleu = 0.0
        for ep in range(ep_init, self.maxEpoch):
            pbar = tqdm(total=self.batchnum, desc=f"Epoch {ep}")
            for src, tgt in cyclingZip(self.srcLoader, self.tgtLoader, self.batchnum):
                self.model.train()
                aeLoss, discLoss, genLoss = self._trainPrepStep(src, tgt) \
                    if ep < self.prepEpoch else self._trainStyleStep(src, tgt)

                # print out
                pbar.update(1)
                pbar.set_postfix_str(f"aeLoss: {aeLoss:.5f} |"
                                     f" discLoss: {discLoss:.5f} |"
                                     f" genLoss: {genLoss:.5f}")
                gstep += 1
                # logging
                self.writer.add_scalars(
                    'loss', {'aeloss': aeLoss,
                             'discloss': discLoss,
                             'genLoss': genLoss
                             }, gstep)

                outSample = self.outpath / f"ol_results_{ep}_{gstep}.txt"
                outModel = self.outpath / f"iytransmod_{ep}_{gstep}.pt"
                if gstep % self.checkpoint == 0:
                    bleuRef, bleuOri, tranAcc, domAcc, _, recBleu = \
                        self.evaluate(self.valLoader)
                    print(f"bleu_ref: {bleuRef:.5f}, rec_bleu: {recBleu:.5f}")
                    print(f"domain_acc: {domAcc:.5f}, tran_acc: {tranAcc:.5f}")
                    self.writer.add_scalars('accu',
                        {'val_transfer': tranAcc,
                         'val_domain': domAcc}, gstep)
                    self.writer.add_scalars('bleu',
                        {'val_ref': bleuRef, 'val_rec': recBleu}, gstep)

                    if bleuRef > bestbleu and tranAcc > self.cutAcc:
                        bestbleu = bleuRef
                        bleuRefOL, bleuOriOL, tranAccOL, domAccOL, sampleStr, recBleuOL = \
                            self.evaluate(self.onlineLoader)
                        print(f"bleu_ori: {bleuOriOL:.5f}, bleu_ref: {bleuRefOL:.5f}, rec_bleu: {recBleuOL:.5f}")
                        print(f"domain_acc: {domAccOL:.5f}, tran_acc: {tranAccOL:.5f}")
                        self.writer.add_scalars('accu',
                            {'ol_transfer': tranAccOL,
                             'ol_domain': domAccOL}, gstep)
                        self.writer.add_scalars('bleu',
                            {'ol_ref': bleuRefOL,
                             'ol_ori': bleuOriOL, 'ol_rec': recBleuOL}, gstep)
                        if ep >= self.prepEpoch:
                            outSample.write_text(sampleStr)

                    # save model
                    torch.save(self.model.state_dict(), outModel)
            if ep >= self.prepEpoch:
                self.scheduler.step()
            pbar.close()

    def _trainPrepStep(self, src, tgt):
        # train discriminator
        discLoss = self.model.classify(src, tgt)
        discLoss.backward()
        self.optimDisc.step()
        self.optimDisc.zero_grad()

        # train generator
        aeLoss, genLoss = self.model(src, tgt, stylize=False)
        aeLoss.backward()
        self.optimAE.step()
        self.optimAE.zero_grad()
        return aeLoss.item(), discLoss.item(), genLoss

    def _trainStyleStep(self, src, tgt):
        # freeze discriminator training
        discLoss = 0.0

        totLoss, genLoss = self.model(src, tgt, stylize=True)
        totLoss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.gen.parameters(), 30.0)
        self.optimAE.step()
        self.optimTot.zero_grad()
        return totLoss.item() - self.model.rho * genLoss.item(), discLoss, genLoss.item()

    def evaluate(self, dataLoader):
        self.model.eval()
        self.domainClf.eval()
        self.targetClf.eval()
        tranAcc = 0
        domAcc = 0

        recBows = []
        hypoBows = []
        styleRefs = []
        originals = []
        for batch in tqdm(dataLoader, desc="Validating"):
            aeLoss, discLoss, genLoss, recIDs, tranIDs = self.model.evaluate(batch)
            recBows.extend(self.idToBow(recIDs, self.mixVocab))
            tranBows = self.idToBow(tranIDs, self.mixVocab)
            hypoBows.extend(tranBows)
            styleRefs.extend(batch.styref)
            originals.extend(batch.orig)

            tranInps = self.bowToID(tranBows, self.tarVocab)
            _, tranPreds = self.targetClf(*tranInps)
            labels = 1 - batch.label
            tranAcc += (tranPreds == labels).sum(0).item()

            domInps = self.bowToID(tranBows, self.mixVocab)
            _, domPreds = self.domainClf(*domInps)
            domAcc += (domPreds == 1).sum(0).item()

        recBleu = corpus_bleu([[ori.split()] for ori in originals],
            recBows, smoothing_function=self.smoothie)

        bleuRef = corpus_bleu([[ref.split()] for ref in styleRefs],
            hypoBows, smoothing_function=self.smoothie)
        bleuOri = corpus_bleu([[ori.split()] for ori in originals],
            hypoBows, smoothing_function=self.smoothie) \
            if dataLoader.dataset.isOnline else bleuRef

        bz = getattr(dataLoader.batch_sampler, 'bz', dataLoader.batch_size)
        tranAcc /= len(dataLoader) * bz
        domAcc /= len(dataLoader) * bz
        sampleStr = self.outputString(recBows, hypoBows, styleRefs, originals)
        return bleuRef, bleuOri, tranAcc, domAcc, sampleStr, recBleu

    @staticmethod
    def idToBow(ids, vocab):
        eos = vocab.word2id('<eos>')
        bows = []
        for sent in ids:
            one = []
            for i in sent:
                if i == eos:
                    break
                else:
                    one.append(vocab.id2word(i))
            bows.append(one)
        return bows

    def bowToID(self, bows, vocab):
        pad = vocab.word2id('<pad>')
        encTs = []
        encLens = []
        for bow in bows:
            ids = [vocab.word2id(w) for w in bow][:self.mlen-1]
            enc = np.ones(self.mlen - 1, dtype=np.int) * pad
            enc[:len(ids)] = ids
            encTs.append(torch.tensor(enc, dtype=torch.long,
                device=self.device))
            encLens.append(torch.tensor(len(ids), dtype=torch.long,
                device=self.device))
        return torch.stack(encTs), torch.stack(encLens)

    @staticmethod
    def outputString(recBows, hypoBows, styleRefs, originals):
        datalist = []
        for rec, hypo, ref, ori in zip(recBows, hypoBows, styleRefs, originals):
            data = json.dumps({
                "original": ori,
                "reconstruction": rec,
                "transfered": hypo,
                "style_ref": ref
            }, indent=2)
            datalist.append(data)
        return "\n".join(datalist)


class ClassifierTrainer(object):

    def __init__(self, model, trainLoader, validLoader, vocab,
                 maxEpoch, lr, mlen, isDomain,
                 outpath, logdir):
        self.model = model
        self.trainLoader = trainLoader
        self.validLoader = validLoader
        self.vocab = vocab
        self.maxEpoch = maxEpoch
        self.mlen = mlen
        self.isDomain = isDomain
        self.prefix = 'domain' if isDomain else 'style'
        self.outpath = outpath
        self.logdir = logdir
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.writer = SummaryWriter(str(logdir))
        self.checkpoint = 2500

    def train(self, ep_init, gstep):
        best_dev = -np.inf
        tr_loss = 0.0
        for ep in range(ep_init, self.maxEpoch):
            pbar = tqdm(total=len(self.trainLoader), desc=f"Epoch {ep}")
            for inp in self.trainLoader:
                self.model.train()
                loss = self._trainStep(inp)
                tr_loss += loss
                gstep += 1

                pbar.update(1)
                mean_loss = tr_loss / gstep
                pbar.set_postfix_str(f"{self.prefix}Loss: {mean_loss:.5f}")

                self.writer.add_scalar(f'{self.prefix}Loss', loss, gstep)

                outModel = self.outpath / f"{self.prefix}Classif_{ep}_{gstep}.pt"
                if gstep % self.checkpoint == 0:
                    acc = self.evaluate(self.validLoader)
                    print(f" Validation Accu: {acc:.5f}")
                    if acc > best_dev:
                        best_dev = acc
                        torch.save(self.model.state_dict(), str(outModel))
            pbar.close()

    def _trainStep(self, inp):
        labels = inp.tgtDom if self.isDomain else inp.label
        loss, _ = self.model(inp.enc, inp.encLen, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def evaluate(self, dataLoader):
        self.model.eval()
        total = 0
        correct = 0
        for inp in dataLoader:
            labels = inp.tgtDom if self.isDomain else inp.label
            _, preds = self.model(inp.enc, inp.encLen)
            total += len(preds)
            correct += (preds == labels).sum(0).item()

        acc = correct / total
        return acc
