import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, sys, os

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def clazz_2_model(clazz):
    from .__init__ import __all__ as all_models

    for clazz1 in all_models:
        if clazz1.__name__ == clazz:
            return clazz1
    raise "Did not find class " + clazz

class BaseModel(nn.Module):
    def __init__(self, meta):
        super().__init__()
        meta.setdefault('ii', 0)
        assert 'title' in meta
        self.meta = meta
        for k,v in meta.items(): setattr(self, k, v)

    def save(self, iter=0, dir='saves'):
        sd = self.state_dict()
        path = os.path.join(dir,self.title + '.{}.pt'.format(iter))
        torch.save({
            'meta': self.meta,
            'sd': sd,
            'clazz': self.__class__.__name__,
            'iter': iter},
            path)
        return path

    @staticmethod
    def load(fname, newMeta, retainedKeys):
        d = torch.load(fname)

        # meta = AttributeDict(d['meta'])
        # Copy any new keys that should be overwritten
        meta = dict(d['meta'])
        for k,v in newMeta.items():
            if k not in retainedKeys: meta[k] = newMeta[k]
        d['meta'] = meta


        model = clazz_2_model(d['clazz'])(d['meta'])
        model.load_state_dict(d['sd'])
        model.ii = d['iter']
        return model, d

    def forward(self, x, *a, **k):
        x = (x.float() / 255) - .5
        return self._forward(x, *a, **k)


def test_save_load():
    from .extractor.extractor1 import Extractor_1
    meta = dict(title='hello')
    model = Extractor_1(meta)

    model.save(0)
    model2 = BaseModel.load('saves/hello.0.pt')
    print(' - loaded model\n', model2)

if __name__ == '__main__':
    test_save_load()

