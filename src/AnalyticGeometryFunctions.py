import numpy as np

def transiteCartesianToPolar(vector):
    return np.arctan2(vector[1], vector[0])

def transitePolarToCartesian(angle):
    return np.array([np.cos(angle), np.sin(angle)])

def computeAngleBetweenVectors(vector1, vector2):
    vectoriseInnerProduct = np.dot(vector1, vector2.T)
    if np.ndim(vectoriseInnerProduct) > 0:
        innerProduct = vectoriseInnerProduct.diagonal()
        normProduct = computeVectorNorm(vector1) * computeVectorNorm(vector2)
    else:
        innerProduct = vectoriseInnerProduct
        normProduct = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    angle = np.arccos(innerProduct / (normProduct + 1e-100))
    return angle

def computeVectorNorm(vector):
    return np.power(np.power(vector, 2).sum(axis = 1), 0.5)

