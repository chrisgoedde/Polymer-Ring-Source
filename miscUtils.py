import os
import pickle
from os.path import expanduser

import numpy as np

def quoted(theString):
    """quoted puts double quotes around any string"""

    return '"' + theString + '"'

def say(theString):

    os.system('say -v Samantha "' + theString + '"')

def pickleTuple(picklePath, pickleFile, tuple):

    if not os.path.exists(picklePath):

        os.makedirs(picklePath)

    pickleHandle = open(picklePath + '/' + pickleFile + '.pickle', 'wb')
    pickle.dump(tuple, pickleHandle, protocol = -1)
    pickleHandle.close()

    print('Saved pickle file ' + picklePath + '/' + pickleFile + '.pickle')

def unpickleTuple(picklePath, pickleFile):

    print('Reading pickle file ' + picklePath + '/' + pickleFile + '.pickle')

    pickleHandle = open(picklePath + '/' + pickleFile + '.pickle', 'rb')
    tuple = pickle.load(pickleHandle)
    pickleHandle.close()

    return tuple

def shortListString(l):

    if sorted(l) == list(range(min(l), max(l)+1)):

        theString = str(min(l)) + '-' + str(max(l))

    else:

        theString = str(l)

    return theString

def autocorrelationFunction(tS):

	acf = np.zeros(tS.shape)
	acf[0] = 1
	(T,) = tS.shape
	tSMean = np.mean(tS)
	tSDev = np.sum((tS-tSMean)**2)
	
	for t in range(1,T):
	
		acf[t] = np.sum((tS[t:]-tSMean)*(tS[:-t]-tSMean)) / tSDev
		
	firstZero = np.where(acf <= 0)[0]
		
	return acf, firstZero
	