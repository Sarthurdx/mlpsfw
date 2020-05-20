import numpy as np


#enough for the start, more sophisticated generator will be implemented later

#return simple random data in the same shape as the target data
def dataGeneratorSimple(targetData):
    return np.random.uniform(-1,1,targetData.shape)
 

