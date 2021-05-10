
import torch
from torch import LongTensor, Tensor
import torch.nn as nn
import torch.nn.functional as F


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


class Discriminator(nn.Module):

    def __init__(self, vz, embz, nfilters):
        super().__init__()
        filterL = [1, 2, 3, 4, 5]
        self.emb = nn.Embedding(vz, embedding_dim=embz)
        self.cnnL = nn.ModuleList([ClassifierCNN(filsize, nfilters, embz)
                                   for filsize in filterL])
        self.dropout = nn.Dropout(0.5)  # default 0.5
        self.linear = nn.Linear(nfilters * len(filterL), 2)
        self.celoss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, inp, inpLen, labels=None):
        mask = paddedMask(inp, inpLen).unsqueeze(-1)  # (bz, ts, 1)
        embed = self.emb(inp) if inp.dim() < 3 else \
            torch.tensordot(inp, self.emb.weight, dims=[[2], [0]])
        embedMsk = embed * mask
        outputs = torch.cat([mod(embedMsk) for mod in self.cnnL], dim=1)
        dout = self.dropout(outputs)
        logits = self.linear(dout)
        preds = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
        loss = self.celoss(logits, labels) if labels is not None else None
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

    def autogen(self, inp):
        origInfo, tranInfo = self._encode(inp)
        out, _, _ = self._decode(inp.dec, origInfo)
        dmask = paddedMask(inp.dec, inp.decLen).reshape(-1)
        decOut = self.dropout(out).reshape([-1, self.hz_dec])
        logits = self.proj(decOut)
        loss = self.celoss(logits, inp.tar.reshape(-1)) * dmask
        aeLoss = loss.mean() / inp.tar.shape[0]
        return aeLoss, origInfo, tranInfo

    def forward(self, inp, isEval=False):
        aeLoss, origInfo, tranInfo = self.autogen(inp)
        _, _, tranSoft = self._decode(inp.dec, tranInfo, self.gumbel)

        recIDs, tranIDs = None, None
        if isEval:
            _, _, recIDs = self._decode(inp.dec, origInfo, self.greedy)
            _, _, tranIDs = self._decode(inp.dec, tranInfo, self.greedy)
        return aeLoss, tranSoft, recIDs, tranIDs

    def _encode(self, inp):
        encs = inp.enc[:, :max(inp.encLen)]  # trim extra padding
        encEmb = self.emb(encs)
        encInp = self.dropout(encEmb)
        _, ht = self.gruEnc(encInp)

        labels = inp.label.float()
        domVec = self.domLinear(torch.ones(labels.shape[0], 1, device=self.device) *
                                inp.tgtDom.reshape(-1, 1))
        styOrig = self.styLinear(labels.reshape([-1, 1]))
        styTran = self.styLinear(1 - labels.reshape([-1, 1]))
        origInfo = torch.cat([styOrig, domVec, ht.squeeze()], dim=1)
        tranInfo = torch.cat([styTran, domVec, ht.squeeze()], dim=1)
        return origInfo, tranInfo

    def _decode(self, decs, h0, func=None):
        hid = []
        pemb = None
        idL = []
        h = h0
        for dt in torch.split(decs, 1, dim=1):
            decEmb = self.emb(dt.squeeze()) if pemb is None else pemb
            inp = self.dropout(decEmb)
            h = self.gruDecCell(inp, h)
            hid.append(h)
            pemb, pid = func(h) if func else (None, dt)
            idL.append(pid)

        out = torch.stack(hid, dim=1)
        predIDs = torch.stack(idL, dim=1)
        return out, h, predIDs

    def gumbel(self, h):
        proba = F.gumbel_softmax(self.proj(h), tau=0.1)  # eps = 1e-20 deprecated, tau is gamma in the original code
        inp = torch.matmul(proba, self.emb.weight)
        return inp, proba

    def greedy(self, h):
        ids = torch.argmax(self.proj(h), 1)
        inp = self.emb(ids)
        return inp, ids

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

            self.discTgt.eval()
            self.discSrc.eval()
            tgtGLoss, *_ = self.discTgt(tgtTranSoft[:, :-1, :],
                tgt.encLen, 1-tgt.label)
            srcGLoss, *_ = self.discSrc(srcTranSoft[:, :-1, :],
                src.encLen, 1-src.label)

            genLoss = tgtGLoss + srcGLoss
        else:
            tgtAELoss, *_ = self.gen.autogen(tgt)
            srcAELoss, *_ = self.gen.autogen(src)
            genLoss = 0.0

        aeLoss = tgtAELoss + srcAELoss + self.alpha * domLoss
        totLoss = aeLoss + self.rho * genLoss
        return totLoss, genLoss

    def classify(self, src, tgt):
        tgtDLoss, *_ = self.discTgt(tgt.dec[:, 1:], tgt.encLen, tgt.label)
        srcDLoss, *_ = self.discSrc(src.dec[:, 1:], src.encLen, src.label)
        return tgtDLoss + srcDLoss

    def evaluate(self, tgt):
        aeLoss, tranSoft, recIDs, tranIDs = self.gen(tgt, isEval=True)
        discLoss, *_ = self.discTgt(tgt.dec[:, 1:], tgt.encLen, tgt.label)
        genLoss, *_ = self.discTgt(tranSoft[:, :-1, :], tgt.encLen, 1-tgt.label)
        return aeLoss, discLoss, genLoss, recIDs, tranIDs
