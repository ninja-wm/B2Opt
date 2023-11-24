# B2Opt-Learning-to-Optimize-Black-box-Optimization-with-Little-Budget
This repository is the official implementation of the source code of the paper "B2Opt: Learning to Optimize Black-box Optimization with Little Budget".

# Installation
This project requires running under the Ubuntu 20.04 system, and you need to install the cuda version of pytorch >= 1.12 first.
```bash
pip install -r requirements.txt
git clone git@github.com:ninja-wm/B2Opt-Learning-to-Optimize-Black-box-Optimization-with-Little-Budget.git
cd B2Opt_pkg
./install.sh
```
# Quick Start
We provide demos for training and testing B2Opt below.
## train

```python
import os
import pickle
import numpy as np
import torch
from B2OPT.problem import Problem
from B2OPT.model import B2opt
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
def genOffset(dim, fun):
    if fun['fid'] in ['cecf1']:
        fun['bias'] = (torch.rand(dim, device=DEVICE)-0.5) * \
            (fun['xub']-fun['xlb'])
        fun['w'] = torch.randn((dim, 1), device=DEVICE)
    elif fun['fid'] in ['cecf2', 'cecf3']:
        fun['bias'] = (torch.rand(dim, device=DEVICE)-0.5) * \
            (fun['xub']-fun['xlb'])

FUNCTIONS=dict()
def testfun1(x,b=None,w=None):
    #(xi-bi)^2
    # b,n,dim
    #w (dim,1)
    batch,n,dim=x.shape
    z=x if b is None else x-b.view(-1)
    sc=torch.sin(z)
    sc=sc@w  #b,n,d @  d ,1 = b,n,1
    sc=torch.pow(sc,2).view(batch,n)
    return sc


def testfun2(x,b=None):
    if not b is None:
        b=b.view(-1)
        z=x-b
    else:
        z=x
    sc=torch.sum(torch.abs(z),dim=2)
    return sc
    
    
def testfun3(x,b=None):
    if not b is None:
        z=x-b
    else:
        z=x
    z1=z[:,:,:-1]
    z2=z[:,:,1:]
    sc=torch.sum(torch.abs(z1+z2),dim=2)+torch.sum(torch.abs(z),dim=2)
    return sc

FUNCTIONS['testf1']={
'fid':'testf1',
'fun':testfun1,
'bias':None,
'w':None,
'xub':10,
'xlb':-10,
'bub':10,
'blb':-10,
}

FUNCTIONS['testf2']={
'fid':'testf2',
'fun':testfun2,
'bias':None,
    'xub':10,
    'xlb':-10,
    'bub':10,
    'blb':-10,
}

FUNCTIONS['testf3']={
'fid':'testf3',
'fun':testfun3,
'bias':None,
    'xub':10,
    'xlb':-10,
    'bub':10,
    'blb':-10,
}

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
class myProblem(Problem):
    def __init__(self,fun=None,repaire=True,dim=None):
        super().__init__()
        self.fun=fun
        self.useRepaire=repaire
        self.dim=dim
      
    def repaire(self,x):
        xlbmask=torch.zeros_like(x,device=DEVICE)
        xlbmask[x<self.fun['xlb']]=1
        normalmask=1-xlbmask
        xlbmask=xlbmask*self.fun['xlb']
        x=normalmask*x+xlbmask
        xubmask=torch.zeros_like(x,device=DEVICE)
        xubmask[x>self.fun['xub']]=1
        normalmask=1-xubmask
        xubmask=xubmask*self.fun['xub']
        x=normalmask*x+xubmask
        return x
    
    def calfitness(self,x):
        if self.useRepaire:
            x1=self.repaire(x)
        else:
            x1=x
        r=getFitness(x1,self.fun)
        return r
    
    def genRandomPop(self,batchShape):
        lb=self.fun['xlb'] 
        ub=self.fun['xub']
        return torch.rand(batchShape,device=DEVICE)*(ub-lb)+lb

    def reoffset(self):
        genOffset(self.dim,self.fun)
        
    def setOffset(self,offset):
        for key in offset.keys():
            self.fun[key]=offset[key]
    
    def lossFunc(self,father,off):
        meanoff=torch.mean(self.calfitness(off))
        meanfather=torch.mean(self.calfitness(father))
        r=(meanoff-meanfather)/torch.abs(meanfather)
        
        if r is torch.nan:
            print('find a nan')
            os.system('pause')
        return r
    
    def getfunname(self):
        return self.fun['fid']
    
    def setfun(self,fun):
        self.fun=fun

def train(expname=None,dim=None,hiddendim=None,ems=None,ws=None,popsize=100,problem=None,maxepoch=None,lr=None,batchsize=None,saveStep=None,T=10,funset=None,needsave=True):
    b2opt=B2opt(dim=dim,hidden_dim=hiddendim,popSize=100,ems=ems,ws=ws).to(DEVICE)
    opt=torch.optim.Adam(b2opt.parameters(),lr=lr)
    batchShape=(batchsize,popsize,dim)
    bar=tqdm(range(maxepoch),ncols=120)
    losslist=[]
    plt.figure(figsize=(12,9))
    minloss=None
    for epoch in bar:
        if (epoch+1)%100==0:
            for param_group in opt.param_groups:
                param_group['lr'] *=0.9
        if epoch==0 or (epoch+1)%T==0:
            for fun in funset:
                genOffset(dim,fun)
            plt.plot(losslist)
            plt.savefig("./imgs/trainloss/exp(%s)_dim(%d).png"%(expname,dim))
             
        for fun in funset:
            problem.setfun(fun)
            pop=problem.genRandomPop(batchShape)
            offpop,trail,evalnums=b2opt(pop,problem)
            loss=problem.lossFunc(pop,offpop)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(b2opt.parameters(), 10, norm_type=2)
            opt.step()
        
        totalloss=None
        for fun in funset:
            problem.setfun(fun)
            pop=problem.genRandomPop(batchShape)
            offpop,trail,evalnums=b2opt(pop,problem)
            loss=problem.lossFunc(pop,offpop)
            if totalloss is None:
                totalloss=loss
            else:
                totalloss+=loss
        totalloss/=len(funset) 
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(b2opt.parameters(), 10, norm_type=2)
        opt.step()
        losslist.append(totalloss.item())
        if (minloss is None or totalloss.item()<minloss) and needsave:
            minloss=totalloss.item()
            flist=os.listdir('./ckpt/')
            for file in flist:
                if file.startswith('exp(%s)_dim(%d)'%(expname,dim)):
                    os.remove('./ckpt/'+file)
            torch.save(b2opt.state_dict(),"./ckpt/exp(%s)_dim(%d)_epoch(%d).pth"%(expname,dim,epoch+1))
            
        bar.set_description("exp(%s)_dim(%d)_loss:%.6f|minloss:%.6f"%(expname,dim,totalloss.item(),minloss))
         
          
if __name__=="__main__":
    print('start')
    funset=[FUNCTIONS['testf1'],FUNCTIONS['testf2'],FUNCTIONS['testf3']]
    dim=10
    problem=myProblem(fun=F['testf1'],dim=dim,repaire=True)
    expname='train_as_30OBs_with_ws'
    hiddendim=200
    ems=30
    ws=True
    popsize=100
    problem=problem
    maxepoch=1000
    lr=0.001
    batchsize=64
    saveStep=100
    T=20
    train(expname=expname,dim=dim,hiddendim=hiddendim,ems=ems,ws=ws,popsize=popsize,problem=problem,
    maxepoch=maxepoch,lr=lr,batchsize=batchsize,saveStep=saveStep,T=T,funset=funset)
    
    
## test




# Citiation
