import os
import pickle
import numpy as np
import torch
from B2OPT.problem import Problem
from B2OPT.model import B2opt
from BBOB.cecfunctions import FUNCTIONS as F
from BBOB.bbobfunctions import FUNCTIONS as BBOBF
from BBOB.utils import getFitness,genOffset,setOffset,getOffset
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import argparse

os.environ['CUDA_LAUNCH_BLOCKING']='1'

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



class bbobProblem(Problem):
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
            
        b,n,d=x.shape
        x1=x1.view((-1,d))
        r=getFitness(x1,self.fun)   #b,n,1
        r=torch.unsqueeze(r,-1)
        r=r.view((b,n))
        
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




def train(expname=None,dim=None,hiddendim=None,ems=None,ws=None,popsize=100,problem=None,
          maxepoch=None,lr=None,batchsize=None,T=10,funset=None,needsave=True):
    
    
    b2opt=B2opt(dim=dim,hidden_dim=hiddendim,popSize=popsize,ems=ems,ws=ws).to(DEVICE)
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
            plt.savefig('./imgs/trainloss/%s_ems_%d_ws_%s_d%d.png'%(expname,ems,str(ws),dim))
       
        
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
            torch.save(b2opt.state_dict(),'./ckpt/%s_ems_%d_ws_%s_d%d.pth'%(expname,ems,str(ws),dim))
            print('minloss is %.2f'%minloss,
                  'checpoint is saved to ./ckpt/%s_ems_%d_ws_%s_d%d.pth'%(expname,ems,str(ws),dim))
            
        bar.set_description("exp(%s)_dim(%d)_loss:%.6f|minloss:%.6f"%(expname,dim,totalloss.item(),minloss))
        if torch.isnan(totalloss):
            print('find a nan')
        if totalloss.item()==torch.nan:
            print('find a nan')
            os.system('pause')
            
        



def test(expname=None,dim=None,hiddendim=None,ems=None,ws=None,popsize=100,problem=None,
       batchsize=None,runs=5):
    b2opt=B2opt(dim=dim,hidden_dim=hiddendim,popSize=popsize,ems=ems,ws=ws).to(DEVICE)
    # try:
    b2opt.load_state_dict(torch.load('./ckpt/%s_ems_%d_ws_%s_d%d.pth'%(expname,ems,str(ws),dim)))
    print('checkpoint %s loaded!'%('./ckpt/%s_ems_%d_ws_%s_d%d.pth'%(expname,ems,str(ws),dim)))
    # except:
    #     print('we can not find checkpoint %s!'%('./ckpt/%s_ems_%d_ws_%s_d%d.pth'%(expname,ems,str(ws),dim)))
    #     return
    b2opt.eval()
    batchShape=(batchsize,popsize,dim)
    bar=tqdm(range(runs))
    trails=[]
    
    
    if problem.getfunname()=='cecf1':
        offset={
            'bias':torch.zeros(dim,device=DEVICE),
            'w':torch.eye(dim,device=DEVICE),
        }
    else:
        offset={
            'bias':torch.zeros(dim,device=DEVICE),
        }
    
    problem.setOffset(offset)
    with torch.no_grad():
        for run in bar:
            pop=problem.genRandomPop(batchShape)
            offpop,trail,evalnums=b2opt(pop,problem)
            besttrailindex=torch.argmin(trail[:,-1])
            besttrail=trail[besttrailindex]
            trails.append(torch.unsqueeze(besttrail,dim=0))
    trails=torch.cat(trails,dim=0)
    print("B2OPT-%s | mean:%.2E(%.2E)"%(problem.getfunname(),torch.mean(trails[:,-1]),torch.std(trails[:,-1])))
    totaltrail={
        'trails':trails.detach().cpu().numpy(),
        'evalnums':evalnums,
    }
    
    totaltrail['mean']=np.mean(totaltrail['trails'][:,-1])
    totaltrail['std']=np.std(totaltrail['trails'][:,-1])
    
    
    
    with open('./trails/exp(%s)_f(%s)_dim(%d).pkl'%(expname,problem.getfunname(),dim),'wb') as f:
        pickle.dump(totaltrail,f)
    
    print('dim%d'%dim,problem.getfunname(),'B2OPT','%.2E(%.2E)'%(totaltrail['mean'],totaltrail['std']))
            
            


def expForFunction(dim=5,expname='basemodel',ems=30,ws=True,
                   popsize=100,maxepoch=1000,lr=0.001):
    '''
    这是用来测试B2Opt在测试函数上的表现的函数
    '''
    print('start')
    funset=[F['cecf1'],F['cecf2'],F['cecf3']]
    # torch.autograd.set_detect_anomaly(True)
    problem=myProblem(fun=F['cecf1'],dim=dim,repaire=True)
    hiddendim=200
    lr=0.001
    batchsize=64
    T=20
    train(expname=expname,dim=dim,hiddendim=hiddendim,ems=ems,ws=ws,popsize=popsize,problem=problem,
    maxepoch=maxepoch,lr=lr,batchsize=batchsize,T=T,funset=funset)
    
    
    

def testSysFuns(dim=5,expname='basemodel',ems=30,ws=True,
                   popsize=100):
    problem=myProblem(fun=F['cecf1'],dim=dim,repaire=True)
    hiddendim=200
    batchsize=64
    testfunset=[F['cecf4'],F['cecf5'],F['cecf6'],F['cecf7'],F['cecf8'],F['cecf9']]
    for fun in testfunset:
        problem.setfun(fun)
        test(expname=expname,dim=dim,hiddendim=hiddendim,ems=ems,ws=ws,popsize=popsize,problem=problem,batchsize=batchsize,runs=5)


def testBBOBFuns(expname=None,dim=None,hiddendim=200,ems=None,ws=None,popsize=100,problem=None,
       batchsize=64,runs=5):
    b2opt=B2opt(dim=dim,hidden_dim=hiddendim,popSize=100,ems=ems,ws=ws).to(DEVICE)
    b2opt.load_state_dict(torch.load('./ckpt/%s_ems_%d_ws_%s_d%d.pth'%(expname,ems,str(ws),dim)))
    b2opt.eval()
    batchShape=(batchsize,popsize,dim)
    bar=tqdm(range(runs))
    trails=[]
    with torch.no_grad():
        for run in bar:
            pop=problem.genRandomPop(batchShape)
            offpop,trail,evalnums=b2opt(pop,problem)
            besttrailindex=torch.argmin(trail[:,-1])
            besttrail=trail[besttrailindex]
            trails.append(torch.unsqueeze(besttrail,dim=0))
    trails=torch.cat(trails,dim=0)
    print("B2OPT-bbob-%s | mean:%.2E(%.2E)"%(problem.getfunname(),torch.mean(trails[:,-1]),torch.std(trails[:,-1])))
    totaltrail={
        'trails':trails.detach().cpu().numpy(),
        'evalnums':evalnums,
    }
    totaltrail['mean']=np.mean(totaltrail['trails'][:,-1])
    totaltrail['std']=np.std(totaltrail['trails'][:,-1])
    
    
    
    with open('./trails/exp(%s)_f(%s)_dim(%d).pkl'%(expname,problem.getfunname(),dim),'wb') as f:
        pickle.dump(totaltrail,f)
    
    return (problem.getfunname(),'B2Opt','%.2E(%.2E)'%(totaltrail['mean'],totaltrail['std']))    



def parseargs():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dim','-d',required=True,type=int,help='a integer stands for the dimension')
    parser.add_argument('--expname','-expname',required=True,type=str,default='testB2opt',
                        help='a name to mark the model and experiment')
    parser.add_argument('--ems','-ems',required=True,type=int,default=305,help='the number of ems')
    parser.add_argument('--ws','-ws',required=True,type=bool,default=True,choices=[True,False],
                        help='a flag indicates whether parameters of ems are shared')
    parser.add_argument('--popsize','-popsize',required=True,type=int,default=100)
    parser.add_argument('--maxepoch','-maxepoch',required=False,type=int,default=1000,help='number of training iterations')
    parser.add_argument('--lr','-lr',required=False,type=float,default=0.001)
    parser.add_argument('--mode','-mode',required=True,type=str,default='test',choices=['train','test'])
    parser.add_argument('--target','-target',required=False,type=str,default='bbob',choices=['sys','bbob'],
                        help='test target')
    args=parser.parse_args()
    return args


def genBBOBoffset(dim=10):
    offsets=dict()
    bar=tqdm(range(1,25))
    offsets=dict()
    for fid in bar :
        f=BBOBF[fid]
        genOffset(dim,f) 
        if not fid in [5,24]:
            f['xopt']=torch.zeros((dim,)).cuda()
        f['fopt']=0
        offsets[fid]=getOffset(f)
    with open('bbobOffsets_dim%d.pkl'%dim,'wb') as f:
        pickle.dump(offsets,f)


if __name__=="__main__":
    args=parseargs()
    dim=args.dim
    expname=args.expname
    ems=args.ems
    ws=args.ws
    popsize=args.popsize
    maxepoch=args.maxepoch
    lr=args.lr
    mode=args.mode
    target=args.target
    if mode=='train':
        print('rdy to train B2Opt(ems:%d ws:%s dim:%d),expname:%s'%(ems,str(ws),dim,expname))
        expForFunction(dim=dim,expname=expname,ems=ems,ws=ws,popsize=popsize,maxepoch=maxepoch,lr=lr)
    else:
        if target=='sys':
            print('rdy to test on synthetic functions')
            testSysFuns(dim=dim,expname=expname,ems=ems,ws=ws,popsize=popsize)
        
        if target=='bbob':
            print('rdy to test on bbob functions')
            testfunset=[i for i in range(1,25)] 
            problem=bbobProblem(fun=BBOBF[1],dim=dim,repaire=True)
            if not os.path.exists('bbobOffsets_dim%d.pkl'%dim):
                genBBOBoffset(dim)
            with open('bbobOffsets_dim%d.pkl'%dim,'rb') as f:
                offsets=pickle.load(f)
                for fid in testfunset:
                    fun=BBOBF[fid]
                    fun['xlb']=-5
                    fun['xub']=5
                    offset=offsets[fun['fid']]
                    setOffset(fun,offset)
                    problem.setfun(fun)
                    try:
                        testBBOBFuns(dim=dim,expname=expname,ems=ems,ws=ws,popsize=popsize,problem=problem)
                    except:
                        pass



 















