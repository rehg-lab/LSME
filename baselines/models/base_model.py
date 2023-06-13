from abc import ABC, abstractmethod

class BaseModel:

    @abstractmethod
    def build_net(self):
        pass
    
    @abstractmethod
    def define_optimizer(self):
        pass

    @abstractmethod
    def train_epoch(self):
        pass
    
    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def get_datasets(self):
        pass

    @abstractmethod
    def get_loaders(self):
        pass

    @abstractmethod
    def train_full(self):
        pass

'''    
    @abstractmethod
    def eval_lowshot(self):
        pass
   
    @abstractmethod
    def eval_viz(self):
        pass
'''
