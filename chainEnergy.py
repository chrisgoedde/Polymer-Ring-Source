import numpy as np
import prody as pd

def chainEnergy(pathName, psfFile, coords):

	# First, parse the psf and parameter files. We get most of the information
	# from the psf file, and only take the atom locations from the pdb (or dcd)
	# file. We feed the filenames to the functions without the extensions.

	# atomList is a list with numAtoms elements. Each element is a dictionary.
	# The entries in each dictionary are 'serial', 'segname', 'resid', 'resname',
	# 'name', 'type', 'charge', and 'mass'. Note that atomList itself is not
	# necessarily ordered in any way; the atoms are entered into the list in the 
	# order they appear in the psf file.
	
	# bondList, angleList, and dihedralList are all lists of bonded atoms. Each
	# element is in bondList is a list of two atom serial numbers, each element
	# in angle list is a list of three atom serial numbers, and each element of
	# dihedralList is a list of four atom serial numbers.
	
	atomList, bondList, angleList, dihedralList = readPSF(pathName + '/' + psfFile)
	
	# nbArray is a (numAtoms, numAtoms) numpy array with elements 0, 1 or 2. It
	# is upper-triangular, so the diagonal is all zeros. In the upper triangle,
	# a zero element means the atoms are not bonded, a one means an ordinary bond,
	# and a 2 means the two atoms are the 1-4 pair for the '1-4scaled' parameter
	# for the LJ potential.
	
	nbArray = findNonbonded(atomList, bondList, angleList, dihedralList)
	
	# eta, rMin, and charge start as (numAtoms, numAtoms) numpy arrays that hold
	# the parameters for the LJ and Coulomb potentials for the non-bonded energies.
	# But what is actually returned is a (numNonbonded,) numpy array with only the
	# necessary parameters for the atoms that are actually interacting via the 
	# nonbonded forces.

	eta, rMin, charge = findNonbondedParameters(atomList, nbArray)
	
	# bondParams is a two-element tuple that holds the kBond and b0 parameters
	# for the bonded energy. kBond and b0 are each (numBonds,) numpy arrays.

	# angleParams is a four-element tuple that holds the kAngle, theta0,
	# kUB, and s0 parameters for the flexing angle energy. kAngle, theta0, kUB,
	# and s0, are all (numAngles,) numpy arrays.

	# dihedralParams is a three-element tuple that holds the kDihedral, mode, and
	# delta parameters dihedral energy. kDihedral, mode, and delta are all lists
	# with numDihedral elements. The elements of these lists are numpy arrays with
	# between 1 and 4 elements, depending on the atoms associated with each dihedral
	# bond.

	bondParams, angleParams, dihedralParams = findBondedParameters(atomList, bondList, angleList, dihedralList)
	
	# r is a numpy array of atom positions. Its shape is (3, numTimes, numAtoms).
	# The exact value of numTimes will depend on the number of frames in the coord
	# argument.
	
	r = coords
	numTimes = coords.shape[1]
	
	# distance is a numpy array with shape (numNonbonded, numTimes), where numNonbonded
	# is the number of nonbonded interactions between atoms as derived from the psf
	# file.
	
	distance = findNonbondedDistance(atomList, nbArray, r, numTimes)
	
	# bDist, aDist, and dDist are the distances to be used in calculating the energies
	# for the bonded interactions. They are numpy arrays with shapes (numBonds, numTimes),
	# (numAngles, numTimes), and (numDihedrals, numTimes) respectively.
	
	bDist, aDist, dDist = findBondedDistance(bondList, angleList, dihedralList, r, numTimes)
	
	eLJ = LJ(eta, rMin, distance, numTimes)
	print("Calculated LJ energy")
	
	eCoulomb = Coulomb(charge, distance, numTimes)
	print("Calculated Coulomb energy")
	
	eBond = bondEnergy(bondParams, bDist, numTimes)
	print("Calculated bond energy")
	
	eAngle = angleEnergy(angleParams, aDist, numTimes)
	print("Calculated angle energy")
	
	eDihedral = dihedralEnergy(dihedralParams, dDist, numTimes)
	print("Calculated dihedral energy")
	
	return eLJ, eCoulomb, eBond, eAngle, eDihedral
	
def LJ(eta, rMin, r, numTimes):

	n = len(eta)
 
	e = np.tile(eta, numTimes).reshape((numTimes, n)).T
	rM = np.tile(rMin, numTimes).reshape((numTimes, n)).T
	
	energy = np.sum(e * ( (rM/r)**12 - 2*(rM/r)**6 ), axis=0)
	
	return fixDim(energy)
	
def Coulomb(charge, r, numTimes):

	n = len(charge)
	
	c = np.tile(charge, numTimes).reshape((numTimes, n)).T
	
	energy = np.sum(331.6 * c / r, axis=0)
	
	return fixDim(energy)
	
def bondEnergy(bP, b, numTimes):

	(kBond, b0) = bP
	
	n = len(kBond)
	
	kB = np.tile(kBond, numTimes).reshape((numTimes, n)).T
	bb = np.tile(b0, numTimes).reshape((numTimes, n)).T
	
	energy = np.sum(kB * (b-bb)**2, axis=0)
	
	return fixDim(energy)
	
def angleEnergy(aP, aD, numTimes):

	(kAngle, theta0, kUB, s0) = aP
	(theta, s) = aD
	
	n = len(kAngle)
	
	kA = np.tile(kAngle, numTimes).reshape((numTimes, n)).T
	t0 = np.tile(theta0, numTimes).reshape((numTimes, n)).T
	kU = np.tile(kUB, numTimes).reshape((numTimes, n)).T
	ss = np.tile(s0, numTimes).reshape((numTimes, n)).T
	
	energy = np.sum(kA * (theta-t0)**2 + kU * (s-ss)**2, axis=0)
	
	return fixDim(energy)
	
def dihedralEnergy(dP, d, numTimes):

	(kDihedral, mode, delta) = dP
		
	n = len(kDihedral)
	
	energy = 0
	
	for i in range(n):
	
		m = len(kDihedral[i])
		
		kD = np.tile(kDihedral[i], numTimes).reshape((numTimes, m)).T
		mo = np.tile(mode[i], numTimes).reshape((numTimes, m)).T
		de = np.tile(delta[i], numTimes).reshape((numTimes, m)).T
		
		energy = energy + np.sum(kD * (1 + np.cos(mo*d[i]-de)), axis=0)
		
	return fixDim(energy)
	
def fixDim(e):
	
	if e.shape[0] == 1:
	
		return e[0]
		
	else:
	
		return e
		
def findBondedDistance(bondList, angleList, dihedralList, r, numTimes):

	numBonds = len(bondList)
	numAngles = len(angleList)
	numDihedrals = len(dihedralList)
	
	bonds = np.zeros((numBonds, numTimes))
	
	for i in range(numBonds):
		
		deltaR = r[:,:,bondList[i][0]] - r[:,:,bondList[i][1]]
		bonds[i, :] = np.linalg.norm(deltaR, axis = 0)
			
	theta = np.zeros((numAngles, numTimes))
	s = np.zeros((numAngles, numTimes))
	
	for i in range(numAngles):
		
		r1 = r[:,:,angleList[i][0]] - r[:,:,angleList[i][1]]
		r2 = r[:,:,angleList[i][2]] - r[:,:,angleList[i][1]]
		
		D = np.sum(r1.conj()*r2, axis=0)
		M = np.linalg.norm(r1, axis=0)*np.linalg.norm(r2, axis=0)

		theta[i, :] = abs(np.arccos(D/M))
		
		deltaR = r[:,:,angleList[i][0]] - r[:,:,angleList[i][2]]
		s[i, :] = np.linalg.norm(deltaR, axis = 0)		  

	phi = np.zeros((numDihedrals, numTimes))

	for i in range(numDihedrals):
		
		r1 = r[:,:,dihedralList[i][1]] - r[:,:,dihedralList[i][0]]
		r2 = r[:,:,dihedralList[i][2]] - r[:,:,dihedralList[i][1]]
		r3 = r[:,:,dihedralList[i][3]] - r[:,:,dihedralList[i][2]]
		
		v1 = np.cross(r1.T, r2.T)
		v2 = np.cross(r2.T, r3.T)
		
		v1 = v1.T
		v2 = v2.T
		
		ratio = np.sum(v1.conj()*v2, axis=0)/(np.linalg.norm(v1, axis=0)*np.linalg.norm(v2, axis=0))
		
		ratio = np.minimum(ratio, np.ones(ratio.shape))
		ratio = np.maximum(ratio, -np.ones(ratio.shape))
			
		phi[i,:] = np.arccos(ratio)

	return bonds, (theta, s), phi

def findNonbondedDistance(atomList, nbArray, r, numTimes):

	# First, find the number of atoms and then make a distance array.
	
	numAtoms = nbArray.shape[0]
	
	dArray = np.zeros((numAtoms, numAtoms, numTimes))

	# we go through the upper half of nbArray, calculating the distance for
	# any nonbonded pairs. Remember that nbArray is indexed by the serial number
	# of the atoms, which is not necessarily the same as the row, column index.
	
	for i in range(numAtoms):
	
		for j in range(i+1, numAtoms):
		
			# The i index is always less than the j index. Now find the row
			# and column index for the two given atoms.
		
			row, col = findIndices(atomList, i, j)
			
			# If the pair is non-bonded, calculate the parameters.
			
			if nbArray[row, col] > 0:
			
				for n in range(numTimes):
				
					dArray[row, col, n] = np.linalg.norm(r[:, n, row]-r[:, n, col], axis = 0)

	# Use a little python magic to turn the arrays into column vectors that only
	# contain the distances for the nonbonded interactions.
	
	values = [ 1, 2 ]	 
	mask = np.isin(nbArray, values)
	
	tempArray = dArray[:, :, 0]
	dTemp = tempArray[mask]
		
	distance = np.zeros((len(dTemp), numTimes))
	distance[:, 0] = dTemp
		
	for n in range(1, numTimes):
		
		tempArray = dArray[:, :, n]
		distance[:, n] = tempArray[mask]
	
	return distance
	
def findBondedParameters(atomList, bondList, angleList, dihedralList):

	# first, get the bond parameters
	# we use the bond array to create parallel arrays for kBond and b0
	
	numBonds = len(bondList)
	kBond = np.zeros(numBonds)
	b0 = np.zeros(numBonds)

	# there are four atom types, CC32A, CC33A, HCA2, and HCA3
	# we can convert these strings to numbers using the ord() function,
	# then add the values for the pairs to get a unique encoding for
	# each pair.

	for i in range(numBonds):
	
		pair = sum([ ord(x) for x in atomList[bondList[i][0]]['type'] ]) \
			+ sum([ ord(x) for x in atomList[bondList[i][1]]['type'] ])
			
		if pair == 554: # [ HCA2 CC32A ]
			
			kBond[i] = 309
			b0[i] = 1.111
			
		elif pair == 556: # [ HCA3 CC33A ]
			
			kBond[i] = 322
			b0[i] = 1.111
			
		elif pair == 600: # [ CC32A CC32A ]
			
			kBond[i] = 222.5
			b0[i] = 1.530
			
		elif pair == 601: # [ CC33A CC32A ]
			
			kBond[i] = 222.5
			b0[i] = 1.528
			
		else:
			
			print('Found an unprocessed pair: [ ' + atomList[bondList[i][0]]['type'] \
				+ ' ' + atomList[bondList[i][1]]['type'] + ' ], ' + str(pair))
				
	# we find the number of angle bonds
	# we create parallel arrays for the four angle energy parameters
	
	numAngles = len(angleList)
	kAngle = np.zeros(numAngles)
	theta0 = np.zeros(numAngles)
	kUB = np.zeros(numAngles)
	s0 = np.zeros(numAngles)
	
	# there are four atom types, CC32A, CC33A, HCA2, and HCA3
	# we can convert these strings to numbers using the ord() function,
	# then add the values for the triplets to get a unique encoding for
	# each triplet.
	
	for i in range(numAngles):
		
		triplet = sum([ ord(x) for x in atomList[angleList[i][0]]['type'] ]) \
			+ sum([ ord(x) for x in atomList[angleList[i][1]]['type'] ]) \
			+ sum([ ord(x) for x in atomList[angleList[i][2]]['type'] ])
		
		if (triplet == 855 # [ CC33A CC32A HCA2 ] 
			or triplet == 856): # [ HCA3 CC33A CC32A ]
			
			kAngle[i] = 34.6
			theta0[i] = np.radians(110.1)
			kUB[i] = 22.53
			s0[i] = 2.179
			
		elif triplet == 808: # [ HCA2 CC32A HCA2 ]
			
			kAngle[i] = 35.5
			theta0[i] = np.radians(109)
			kUB[i] = 5.40
			s0[i] = 1.802
			
		elif triplet == 900: # [ CC32A CC32A CC32A ]
			
			kAngle[i] = 58.35
			theta0[i] = np.radians(113.6)
			kUB[i] = 11.16
			s0[i] = 2.561
			
		elif triplet == 854: # [ HCA2 CC32A CC32A ]
			
			kAngle[i] = 26.5
			theta0[i] = np.radians(110.1)
			kUB[i] = 22.53
			s0[i] = 2.179
			
		elif triplet == 811: # [ HCA3 CC33A HCA3 ]
			
			kAngle[i] = 35.5
			theta0[i] = np.radians(108.4)
			kUB[i] = 5.40
			s0[i] = 1.802
			
		elif triplet == 901: # [ CC33A CC32A CC32A ]
			
			kAngle[i] = 58
			theta0[i] = np.radians(115)
			kUB[i] = 8.00
			s0[i] = 2.561
			
		else:
			
			print('Found an unprocessed triplet: [ ' \
				+ atomList[angleList[i][0]]['type'] + ' ' \
				+ atomList[angleList[i][1]]['type'] + ' ' \
				+ atomList[angleList[i][2]]['type'] + ' ], ' + str(triplet))
				
	numDihedrals = len(dihedralList)
	kDihedral = []
	mode = []
	delta = []
	
	for i in range(numDihedrals):
		
		quad = sum([ ord(x) for x in atomList[dihedralList[i][0]]['type'] ]) \
			+ sum([ ord(x) for x in atomList[dihedralList[i][1]]['type'] ]) \
			+ sum([ ord(x) for x in atomList[dihedralList[i][2]]['type'] ]) \
			+ sum([ ord(x) for x in atomList[dihedralList[i][3]]['type'] ])
		
		if (quad == 1110 # [ HCA3 CC33A CC32A HCA2 ]
				or quad == 1156): # [ HCA3 CC33A CC32A CC32A ]
			
			kDihedral.append(np.array([0.16]))
			mode.append(np.array([3]))
			delta.append(np.array([0]))
			
		elif (quad == 1155 # [ HCA2 CC32A CC32A CC33A ]
				or quad == 1108 # [ HCA2 CC32A CC32A HCA2 ]
				or quad == 1154): # [ HCA2 CC32A CC32A CC32A ]
			
			kDihedral.append(np.array([0.19]))
			mode.append(np.array([3]))
			delta.append(np.array([0]))
			
		elif quad == 1201: # [ CC32A CC32A CC32A CC33A ]
			
			kDihedral.append(np.array([0.20391, 0.10824, 0.08133, 0.15051]))
			mode.append(np.array([5, 4, 3, 2]))
			delta.append(np.radians(np.array([0, 0, 180, 0])))
			
		elif quad == 1200: # [ CC32A CC32A CC32A CC32A ]
			
			kDihedral.append(np.array([0.11251, 0.09458, 0.14975, 0.06450]))
			mode.append(np.array([5, 4, 3, 2]))
			delta.append(np.radians(np.array([0, 0, 180, 0])))
			
		else:
			
			print('Found an unprocessed quad: [ ' \
				+ atomList[angleList[i][0]]['type'] + ' ' \
				+ atomList[angleList[i][1]]['type'] + ' ' \
				+ atomList[angleList[i][2]]['type'] + ' ' \
				+ atomList[angleList[i][3]]['type'] + ' ], ' + str(quad))

	return (kBond, b0), (kAngle, theta0, kUB, s0), (kDihedral, mode, delta)

def findNonbondedParameters(atomList, nbArray):

	# We start with two lists for the basic parameters, based on atom type.
	# Each list will have two entries, one for standard parameters, and one
	# for the modified 1-4 parameters. We will use the value stored in nbArray
	# (either 1 or 2) to determine which to use.
	
	# Each entry of the list is a dictionary that contains the parts that are
	# used to calculate the parameters themselves.
	
	eta = [{}, {}]
	rMin = [{}, {}]
	
	eta[0]['CC32A'] = 0.056
	rMin[0]['CC32A'] = 2.01
	eta[1]['CC32A'] = 0.01
	rMin[1]['CC32A'] = 1.9
	
	eta[0]['CC33A'] = 0.078
	rMin[0]['CC33A'] = 2.04
	eta[1]['CC33A'] = 0.01
	rMin[1]['CC33A'] = 1.9
	
	eta[0]['HCA2'] = 0.035
	rMin[0]['HCA2'] = 1.34
	eta[1]['HCA2'] = 0.035
	rMin[1]['HCA2'] = 1.34
	
	eta[0]['HCA3'] = 0.024
	rMin[0]['HCA3'] = 1.34
	eta[1]['HCA3'] = 0.024
	rMin[1]['HCA3'] = 1.34
	
	# Get the number of atoms.
	
	numAtoms = nbArray.shape[0]
	
	# These arrays hold the actual parameters.
	
	etaArray = np.zeros(nbArray.shape)
	rMinArray = np.zeros(nbArray.shape)
	chargeArray = np.zeros(nbArray.shape)
	
	# March through the upper-triangular part of the nbArray.
	# Note that i and j are indices to the atomList list, which is
	# used to pull the serial numbers of each pair of atoms. These
	# serial numbers are then used as row and column indices for the
	# nbArray.
	
	# In most pdb/psf files, we will probably have these indices match,
	# but this is not guaranteed. I think I did this wrong in my Matlab
	# code, but it worked.
	
	for i in range(numAtoms):
	
		for j in range(i+1, numAtoms):
		
			# The i index is always less than the j index. Now find the row
			# and column index for the two given atoms.
		
			row, col = findIndices(atomList, i, j)
			
			# If the pair is non-bonded, calculate the parameters.
			
			if nbArray[row, col] > 0:
			
				# print((row, col))
		
				# Charge is easy; we just multiply the charge of each atom in the pair.
			
				chargeArray[row, col] = atomList[i]['charge'] * atomList[j]['charge']
				
				s = int(nbArray[row, col] - 1)
				
				etaArray[row, col] = np.sqrt(eta[s][atomList[i]['type']] * eta[s][atomList[j]['type']])
				rMinArray[row, col] = rMin[s][atomList[i]['type']] + rMin[s][atomList[j]['type']]
	
	# Use a little python magic to turn the arrays into column vectors that only
	# contain the parameters for the nonbonded interactions.
	
	values = [ 1, 2 ]	 
	mask = np.isin(nbArray, values)
	
	etaArray = etaArray[mask]
	rMinArray = rMinArray[mask]
	chargeArray = chargeArray[mask]	   
	
	return etaArray, rMinArray, chargeArray
	
def findIndices(atomList, i, j):

	row = int(np.min([atomList[i]['serial'], atomList[j]['serial']]))
	col = int(np.max([atomList[i]['serial'], atomList[j]['serial']]))
		
	return row, col
		
def findNonbonded(atomList, bondList, angleList, dihedralList):

	numAtoms = len(atomList)
	numBonds = len(bondList)
	numAngles = len(angleList)
	numDihedrals = len(dihedralList)
	
	# Everything starts non-bonded. We'll set bonded pairs/triplets/quads to 0,
	# and set 1-4 dihedral pairs to 2.
	
	nbArray = np.ones((numAtoms, numAtoms))
	
	# Set the bonded pairs to zero. We do this symmetrically because we don't
	# know the order of the atoms in the pair.
	
	for i in range(numBonds):
	
		setElementsToZero(nbArray, bondList[i], 2)
		
	# Now do the same for the angles, which come in triplets.
	
	for i in range(numAngles):
	
		setElementsToZero(nbArray, angleList[i], 3)
		
	# Lastly, we do the dihedrals, which come in sets of four.
	# However, we give the 1-4 pairs a special marker.
	
	for i in range(numDihedrals):
	
		setElementsToZero(nbArray, dihedralList[i], 3)
		
		nbArray[dihedralList[i][0], dihedralList[i][3]] = 2
		nbArray[dihedralList[i][3], dihedralList[i][0]] = 2
		
	# Set the lower-triangular part to zero, including the diagonal.
	
	nbArray = np.triu(nbArray, 1)
	
	return nbArray 

def setElementsToZero(theArray, theTuple, num):

	for i in range(1,num):
	
		theArray[theTuple[0], theTuple[i]] = 0
		theArray[theTuple[i], theTuple[0]] = 0
		
def readPosition(fileName, numAtoms, dcd = False, stride = 1):
	
	# Make prody be quiet
	
	pd.confProDy(verbosity = 'warning')

	# Read in the pdb file using the ProDy utilities
	
	atoms = pd.parsePDB(fileName + '.pdb')

	if dcd:
	
		coordEnsemble = pd.parseDCD(fileName + '.dcd')
		numTimes = int(coordEnsemble.numCoordsets()/stride)+1
	
	else:
	
		numTimes = 1
		
	r = np.zeros((3, numTimes, numAtoms))	 

	# Extract the coordinates, and return them along with the number of atoms
	
	for i in range(numAtoms):
	
		nextAtom = atoms.select('serial ' + str(i+1))
		nextCoord = nextAtom.getCoords()
		
		if dcd:
		
			coordEnsemble.setAtoms(nextAtom)
			nextCoordset = coordEnsemble.getCoordsets()
			
			r[0, 0, i] = nextCoord[0][0]
			r[1, 0, i] = nextCoord[0][1]
			r[2, 0, i] = nextCoord[0][2]
			
			r[0, 1:, i] = nextCoordset[(stride-1)::stride, 0, 0]
			r[1, 1:, i] = nextCoordset[(stride-1)::stride, 0, 1]
			r[2, 1:, i] = nextCoordset[(stride-1)::stride, 0, 2]
			
		else:
		
			r[0, 0, i] = nextCoord[0][0]
			r[1, 0, i] = nextCoord[0][1]
			r[2, 0, i] = nextCoord[0][2]

	return r, numTimes
	
def readPSF(fileName):

	# Open the psf file for reading
	
	fH = open(fileName + '.psf', "r")
	
	# lineList is a list with each line of the psf file as an element.
	# Blank lines will show up as a '\n' element.
	
	lineList = fH.readlines()
	
	# Find the line with '!NATOM' and extract the number of atoms in the file.
	# atomLine is the line number (counting from 0) of the atoms line in the psf file.
	# Then repeat this for the number of bonds, angles and dihedrals.
	
	atomLine = next(i for i,x in enumerate(lineList) if x.rfind('!NATOM') > 0)
	numAtoms = parseHeader(lineList[atomLine])
	
	bondLine = next(i for i,x in enumerate(lineList) if x.rfind('!NBOND') > 0)
	numBonds = parseHeader(lineList[bondLine])
	
	angleLine = next(i for i,x in enumerate(lineList) if x.rfind('!NTHETA') > 0)
	numAngles = parseHeader(lineList[angleLine])
	
	dihedralLine = next(i for i,x in enumerate(lineList) if x.rfind('!NPHI') > 0)
	numDihedrals = parseHeader(lineList[dihedralLine])
	
	# atomList is a list of dictionaries, one for each atom in the psf file.
	# The entries in each dictionary are 'serial', 'segname', 'resid', 'resname',
	# 'name', 'type', 'charge', and 'mass'
	
	atomList = parseAtoms(lineList, atomLine+1, numAtoms)
	
	# bondList is a list of bonds, each stored in a tuple with 2 elements.
	# angleList is a list of angles, each stored in a tuple with 3 elements.
	# dihedralList is a list of dihedrals, each stored in a tuple with 4 elements.
	
	bondList = parseBonds(lineList, bondLine+1, numBonds, 2)
	angleList = parseBonds(lineList, angleLine+1, numAngles, 3)
	dihedralList = parseBonds(lineList, dihedralLine+1, numDihedrals, 4)
	
	fH.close()
	
	return atomList, bondList, angleList, dihedralList
	
def parseHeader(line):

	return(int(line.lstrip(' ').split(' ')[0]))
	
def parseAtoms(lineList, atomLine, num):

	atomList = []

	for i in range(num):
	
		theDict = {}

		nextAtom = lineList[atomLine+i].split()
		
		# Note that we subtract 1 from the serial because python uses zero-based counting.
		
		theDict['serial'] = int(nextAtom[0]) - 1
		theDict['segname'] = nextAtom[1]
		theDict['resid'] = int(nextAtom[2])
		theDict['resname'] = nextAtom[3]
		theDict['name'] = nextAtom[4]
		theDict['type'] = nextAtom[5]
		theDict['charge'] = float(nextAtom[6])
		theDict['mass'] = float(nextAtom[7])
		
		atomList.append(theDict)
	
	return atomList
	
def parseBonds(lineList, bondLine, num, numIn):

	if numIn == 2:
		numLines = int(np.ceil(num/4))
	elif numIn == 3:
		numLines = int(np.ceil(num/3))
	elif numIn == 4:
		numLines = int(np.ceil(num/2))

	bondList = []
	bondGrab = range(numIn)
	
	for i in range(numLines):
	
		nextBond = lineList[bondLine+i].split()
		
		# We have to offset the atom serial numbers by 1 to account for 
		# zero-based counting in python.
		
		nextBond = [ int(i)-1 for i in nextBond ]
		
		for j in range(0, len(nextBond), numIn):
			
			bondList.append(tuple(nextBond[j:j+numIn]))

	return bondList