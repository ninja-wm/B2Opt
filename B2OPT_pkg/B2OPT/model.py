import torch.nn as nn
import torch
from B2OPT.imports import *
from B2OPT.problem import *

class AttnWithFit (nn.Module):
    def __init__(self,popSize=100,hiddenDim=100):
        super().__init__()
        self.popSize=popSize
        self.attn=nn.Parameter(torch.randn((1,self.popSize,self.popSize)),requires_grad=True)
        self.q =nn.Sequential(
                              nn.Linear(1, hiddenDim),
                              )
        self.k =nn.Sequential(
                              nn.Linear(1, hiddenDim),
                              )
        self.num_heads = 1
        self.F=nn.Parameter(torch.randn((2,)),requires_grad=True)
        
        
    def forward(self,x,fitx):
        B, N, C = fitx.shape
        q = self.q(fitx).view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)      #B，H，N，SEQ
        k = self.k(fitx).view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        fitattn = q @ k.transpose(2, 3) * (x.shape[-1] ** -0.5)                             #b,h,n,n   (a11,a12,aij,...,ann)
        fitattn = torch.squeeze(fitattn.softmax(dim=-1) ,dim=1)
        y1=self.attn.softmax(dim=-1)@x
        y2=fitattn@x
        y=y1*self.F.softmax(-1)[0]+y2*self.F.softmax(-1)[1]
        return y

    def getStrategy(self,fitx,dim):
        B, N, C =fitx.shape
        q = self.q(fitx).view(B, N, self.num_heads, -1).permute(0, 2, 1, 3) #B，H，N，SEQ
        k = self.k(fitx).view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        fitattn = q @ k.transpose(2, 3) * (dim** -0.5) 
        fitattn = torch.squeeze(fitattn.softmax(dim=-1) ,dim=1)
        return self.F.softmax(-1)[0]*self.attn.softmax(dim=-1)+self.F.softmax(-1)[1]*fitattn




class SM(nn.Module):
    def __init__(self):
        super().__init__()
        
    
    def forward(self,batchpop1,batchpop2,fatherfit,childfit,minimize=True):
        '''
        实现选择操作,默认是最小化函数，若minimize=False,则为最大化目标值问题
        '''
        fit1=fatherfit
        fit2=childfit
        batchMask=fit1-fit2   #b,n,1
        if minimize:
            batchMask[batchMask>=0]=0
            batchMask[batchMask<0]=1
        else:
            batchMask[batchMask<=0]=0
            batchMask[batchMask>0]=1
        batchMask=torch.unsqueeze(batchMask,2)
        batchMask1=torch.ones_like(batchMask).to(DEVICE)-batchMask
        nextPop=batchpop1*batchMask+batchpop2*batchMask1
        return nextPop
            




class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def sortpop(self,x,fitness):
        '''
        说明：
        输入：x(b,n,dim),f(x)(b,n,1)
        输出：排序后的x和f(x)
        '''
        fitness,fitindex=torch.sort(fitness,dim=-1)  #b,n fit：the order of individuals
        y=torch.zeros_like(x)
        for index,pop in enumerate(x):
            pop=x[index]
            y[index]=torch.index_select(pop,0,fitindex[index])
        return y,fitness






class  OB(BaseModel):
    def __init__(self,dim=64,hidden_dim=100,popSize=10,temid=0):
        super().__init__()
        self.dim=dim
        self.trm=AttnWithFit(popSize=popSize,hiddenDim=hidden_dim)
        self.mut=nn.Sequential(
            nn.Linear(dim,dim),
            nn.ReLU(),
            nn.Linear(dim,dim)
        )
        self.id=temid
        self.vis=False
        self.f1=nn.Parameter(torch.randn((1,popSize,1)),requires_grad=True)
        self.f2=nn.Parameter(torch.randn((1,popSize,1)),requires_grad=True)
        self.f3=nn.Parameter(torch.randn((1,popSize,1)),requires_grad=True)
        self.sm=SM()
        

    
    def forward(self,x,problem,xfit=None):
        b,n,d=x.shape
        if not xfit is None:
            fatherfit=xfit
        else:
            fatherfit=problem.calfitness(x)
        fitx=fatherfit.softmax(dim=-1)
        fitx=fitx.view(b,n,1)
        crosspop=self.trm(x,fitx)   ##A & AF
        offpop=self.mut(crosspop)  ##NN   MUT
        off=self.f1*x+self.f2*crosspop+self.f3*offpop
        childfit=problem.calfitness(off)
        nextpop=self.sm(x,off,fatherfit,childfit)
        return nextpop
    




class B2opt(BaseModel):
    def __init__(self,dim=64,hidden_dim=100,popSize=10,ems=10,ws=False):
        super().__init__()
        self.ems=ems
        self.ws=ws
        if self.ws:
            self.ob=OB(dim,hidden_dim,popSize)
        else:
            self.ob=torch.nn.ModuleList([OB(dim,hidden_dim,popSize,i) for i in range(ems)])

        
    def forward(self,x,problem):
        self.trail=None
        self.evalnum=[]
        
        if self.ws is True:            
            for i in range(self.ems):
                fatherfit=problem.calfitness(x) #b,n
                x,fatherfit=self.sortpop(x,fatherfit)
                trail=torch.min(fatherfit,dim=-1)[0].view(-1,1)
                if self.trail is None:
                    self.trail=trail
                    self.evalnum.append(x.shape[1])
                else:
                    self.trail=torch.cat((self.trail,trail),dim=-1)
                x=self.ob(x,problem,fatherfit)
                self.evalnum.append(self.evalnum[-1]+x.shape[1])
        else:
            for ob in (self.ob):
                fatherfit=problem.calfitness(x) #b,n
                x,fatherfit=self.sortpop(x,fatherfit)
                trail=torch.min(fatherfit,dim=-1)[0].view(-1,1)
                if self.trail is None:
                    self.trail=trail
                    self.evalnum.append(x.shape[1])
                else:
                    self.trail=torch.cat((self.trail,trail),dim=-1)
                x=ob(x,problem,fatherfit)
                self.evalnum.append(self.evalnum[-1]+x.shape[1])
        
        return x,self.trail,self.evalnum
            



# if __name__=='__main__':
#     task=myProblem()
#     x=torch.randn((10,10,10),device=DEVICE)
#     model=B2opt(dim=10,hidden_dim=100,popSize=10,ems=10,ws=False).to(DEVICE)
#     x,trail,evalnum=model(x,task)
#     print(trail)
#     print(evalnum)