
import torch
from torch import LongTensor, Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


def paddedMask(seqs: LongTensor, seqLens: LongTensor) -> Tensor:
    bz, max_len, *_ = seqs.shape
    mask = torch.arange(max_len, device=seqs.device).expand(bz, max_len) < seqLens.unsqueeze(1)
    return mask


class ClassifierCNN(nn.Module):

    def __init__(self, filsize, nfilters, dz):
        super().__init__()
        self.nfilters = nfilters
        self.cnn = nn.Conv2d(in_channels=1, out_channels=nfilters,
            kernel_size=(filsize, dz), stride=(1, 1))
        self.leakyReLU = nn.LeakyReLU()

    def forward(self, inp):
        inp = torch.unsqueeze(inp, dim=1)
        conv = self.cnn(inp)
        h = self.leakyReLU(conv)
        pooled = h.max(dim=2).values.reshape((-1, self.nfilters))
        return pooled


class BaseClassifier(nn.Module):
    def __init__(self, vz, embz, nfilters):
        super().__init__()
        self.filterL = [1, 2, 3, 4, 5]
        self.emb = nn.Embedding(vz, embedding_dim=embz)
        self.cnnL = nn.ModuleList([ClassifierCNN(filsize, nfilters, embz)
                                   for filsize in self.filterL])
        self.dropout = nn.Dropout(0.5)  # default 0.5

    def forward(self, inp, inpLen, labels=None):
        embed = self.emb(inp) if inp.dim() < 3 else \
            torch.tensordot(inp, self.emb.weight, dims=[[2], [0]])[:, :-1, :]
        mask = paddedMask(embed, inpLen).unsqueeze(-1)  # (bz, ts, 1)
        embedMsk = embed * mask
        outputs = torch.cat([mod(embedMsk) for mod in self.cnnL], dim=1)
        dout = self.dropout(outputs)
        return dout


class Discriminator(BaseClassifier):
    def __init__(self, vz, embz, nfilters):
        super().__init__(vz, embz, nfilters)
        self.linear = nn.Linear(nfilters * len(self.filterL), 1)
        self.bceloss = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, inp, inpLen, labels=None):
        dout = super().forward(inp, inpLen, labels)
        logits = self.linear(dout).reshape(-1)
        bceloss = self.bceloss(logits, labels) if labels is not None else None
        return bceloss, logits


class Evaluator(BaseClassifier):
    def __init__(self, vz, embz, nfilters):
        super().__init__(vz, embz, nfilters)
        self.linear = nn.Linear(nfilters * len(self.filterL), 2)
        self.celoss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, inp, inpLen, labels=None):
        dout = super().forward(inp, inpLen, labels)
        logits = self.linear(dout)
        loss = self.celoss(logits, labels.long()) if labels is not None else None
        preds = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
        return loss, preds


class Generator(nn.Module):

    def __init__(self, vz, embz, hz, domz, stylez, device):
        super().__init__()
        self.hz_dec = hz + domz + stylez  # hidden dim of decoder
        self.domLinear = nn.Linear(1, out_features=domz)  # domain vector size = 50
        self.emb = nn.Embedding(vz, embedding_dim=embz)
        self.dropout = nn.Dropout(0.5)
        self.gruEnc = nn.GRU(embz, hz, batch_first=True)  # hidden dim of encoder = 500
        self.gruDecCell = nn.GRUCell(embz, self.hz_dec)
        self.proj = nn.Linear(self.hz_dec, out_features=vz)
        self.styLinear = nn.Linear(1, out_features=stylez)  # style label vector
        self.celoss = nn.CrossEntropyLoss(reduction='none')
        self.device = device

    def _encode(self, inp):
        encs = inp.enc[:, :max(inp.encLen)]  # trim extra padding
        encEmb = self.emb(encs)
        encInp = self.dropout(encEmb)
        encPacked = pack_padded_sequence(encInp, inp.encLen.cpu(),
            batch_first=True, enforce_sorted=False)
        _, ht = self.gruEnc(encPacked)

        labels = inp.label
        domVec = self.domLinear(torch.ones(labels.shape[0], 1, device=self.device) *
                                inp.tgtDom.reshape(-1, 1))
        styOrig = self.styLinear(labels.reshape([-1, 1]))
        styTran = self.styLinear(1 - labels.reshape([-1, 1]))
        origInfo = torch.cat([styOrig, domVec, ht.squeeze()], dim=1)
        tranInfo = torch.cat([styTran, domVec, ht.squeeze()], dim=1)
        return origInfo, tranInfo

    def autogen(self, inp):
        origInfo, tranInfo = self._encode(inp)
        out = self._decodeWithTeacher(inp.dec, origInfo)
        dmask = paddedMask(inp.dec, inp.decLen).reshape(-1)
        decOut = self.dropout(out).reshape([-1, self.hz_dec])
        logits = self.proj(decOut)
        loss = self.celoss(logits, inp.tar.reshape(-1)) * dmask
        aeLoss = loss.sum() / inp.enc.shape[0]
        return aeLoss, origInfo, tranInfo

    def forward(self, inp, isEval=False):
        aeLoss, origInfo, tranInfo = self.autogen(inp)
        tranSoft = self._decodeGumbel(inp.dec, tranInfo)

        recIDs, tranIDs = None, None
        if isEval:
            recIDs = self._decodeGreedy(inp.dec, origInfo)
            tranIDs = self._decodeGreedy(inp.dec, tranInfo)
        return aeLoss, tranSoft, recIDs, tranIDs

    def _decodeWithTeacher(self, decs, h0):
        h = h0
        hid = []
        for dt in torch.split(decs, 1, dim=1):
            decEmb = self.emb(dt.squeeze())
            inp = self.dropout(decEmb)
            h = self.gruDecCell(inp, h)
            hid.append(h)
        out = torch.stack(hid, dim=1)
        return out  # (bz, ts, hz)

    def _decodeGumbel(self, decs, h0):
        h = h0
        probaL = []
        pemb = None
        for dt in torch.split(decs, 1, dim=1):
            decEmb = self.emb(dt.squeeze()) if pemb is None else pemb
            inp = self.dropout(decEmb)
            h = self.gruDecCell(inp, h)
            hdrop = self.dropout(h)
            proba = F.gumbel_softmax(self.proj(hdrop), tau=0.1)  # eps = 1e-20 deprecated, tau is gamma in the original code
            probaL.append(proba)
            pemb = torch.matmul(proba, self.emb.weight)  # (bz, embz)
        return torch.stack(probaL, dim=1)  # (bz, ts, vz)

    def _decodeGreedy(self, decs, h0):
        h = h0
        idL = []
        pemb = None
        for dt in torch.split(decs, 1, dim=1):
            decEmb = self.emb(dt.squeeze()) if pemb is None else pemb
            inp = self.dropout(decEmb)
            h = self.gruDecCell(inp, h)
            hdrop = self.dropout(h)
            ids = torch.argmax(self.proj(hdrop), 1)
            idL.append(ids)
            pemb = self.emb(ids)  # (bz, embz)
        return torch.stack(idL, dim=1)  # (bz, ts)

    def domainLoss(self):
        posv = self.domLinear(torch.ones(1, 1, device=self.device))
        negv = self.domLinear(torch.zeros(1, 1, device=self.device))
        domLoss = F.mse_loss(posv, negv, reduction='sum') / 2
        return domLoss


class DAST(nn.Module):

    def __init__(self, vz, embz, hz, domz, stylez, nfilters,
                 alpha, rho, device):
        super().__init__()
        self.gen = Generator(vz, embz, hz, domz, stylez, device)
        self.discTgt = Discriminator(vz, embz, nfilters)
        self.discSrc = Discriminator(vz, embz, nfilters)
        self.alpha = alpha
        self.rho = rho

    def forward(self, src, tgt, stylize=False):
        domLoss = self.gen.domainLoss()

        # style-transfer
        if stylize:
            tgtAELoss, tgtTranSoft, *_ = self.gen(tgt)
            srcAELoss, srcTranSoft, *_ = self.gen(src)

            tgtGLoss, *_ = self.discTgt(tgtTranSoft, tgt.encLen, 1-tgt.label)
            srcGLoss, *_ = self.discSrc(srcTranSoft, src.encLen, 1-src.label)

            genLoss = tgtGLoss + srcGLoss
        else:
            tgtAELoss, *_ = self.gen.autogen(tgt)
            srcAELoss, *_ = self.gen.autogen(src)
            genLoss = 0.0

        aeLoss = tgtAELoss + srcAELoss + self.alpha * domLoss
        totLoss = aeLoss + self.rho * genLoss
        return totLoss, genLoss

    def classify(self, src, tgt):
        tgtDLoss, *_ = self.discTgt(tgt.enc, tgt.encLen, tgt.label)
        srcDLoss, *_ = self.discSrc(src.enc, src.encLen, src.label)
        return tgtDLoss + srcDLoss

    def evaluate(self, tgt):
        aeLoss, tranSoft, recIDs, tranIDs = self.gen(tgt, isEval=True)
        discLoss, *_ = self.discTgt(tgt.enc, tgt.encLen, tgt.label)
        genLoss, *_ = self.discTgt(tranSoft, tgt.encLen, 1-tgt.label)
        return aeLoss, discLoss, genLoss, recIDs, tranIDs
