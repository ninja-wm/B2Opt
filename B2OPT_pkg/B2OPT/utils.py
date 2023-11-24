import torch
import pickle
from B2OPT.imports import DEVICE

def sampleBatchPop(batchSize,popSize,dim,xlb,xub):
    batchpop=torch.rand((batchSize,popSize,dim)).to(DEVICE)
    batchpop=batchpop*(xub-xlb)+xlb
    return batchpop

def lossFunc(father,off,fun=None):
    return  (torch.mean(fun['fun'](off,fun['bias']))-torch.mean(
        fun['fun'](father,fun['bias'])))/torch.abs(torch.mean(fun['fun'](father,fun['bias'])))
    # return 0.5*torch.mean(fun['fun'](off,fun['bias']))**2




def reOffSet(fun,zeroOffset=True):
    ub=fun['bub']
    lb=fun['blb']
    dim=fun['dim']
    # zeroOffset=True
    if not zeroOffset:
        bias=torch.rand(dim)*(ub-lb)+lb
    else:
        bias=torch.zeros(dim)
    fun['bias']=bias.to(DEVICE)
    return fun


def dump(file,path):
    with open(path,'wb') as f:
        pickle.dump(file,f)




def load(path):
    with open(path,'rb') as f:
        file=pickle.load(f)
    return file
