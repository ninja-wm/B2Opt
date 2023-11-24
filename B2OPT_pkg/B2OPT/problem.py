class Problem:
    def __init__(self):
        self.useRepaire=False
        
    
    def repaire(self,x):
        raise NotImplementedError
    
    def calfitness(self,x):
        raise NotImplementedError