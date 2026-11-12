import numpy as np

rng=np.random.default_rng(42)

def zero_init(fan_in,fan_out):
    return np.zeros((fan_out,fan_in))

def xavier_uniform(fan_in,fan_out):
    limit=np.sqrt(6/(fan_in+fan_out))
    
    return rng.uniform(-limit,limit,size=(fan_out,fan_in))

def xavier_normal(fan_in,fan_out):
    std=np.sqrt(2/(fan_in+fan_out))
    
    return rng.normal(0,std,size=(fan_out,fan_in))


def HE_uniform(fan_in,fan_out):
    limit=np.sqrt(6/(fan_in))
    
    return rng.uniform(-limit,limit,size=(fan_out,fan_in))

def HE_normal(fan_in,fan_out):
    std=np.sqrt(2/(fan_in))
    
    return rng.normal(0,std,size=(fan_out,fan_in))