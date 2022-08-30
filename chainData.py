import numpy as np
import prodyUtils as prU

# Use ProDy (http://prody.csb.pitt.edu) to get the coordinates for the carbon 
# atoms in the chain from the namd simulation files.

def getChainCoords(dataPathName, fileName, initFileName = '', \
					getInitialValue = True, getVelocity = True, \
					atomType = 'carbon'):

	# The coordinates for the initial condition are in a separate file from the
	# coordinates in the rest of the trajectory. By default, this file will be
	# named "fileName-restraint".
	
	if initFileName == '':

		initFileName = fileName + '-restraint'

	# Get the initial coordinates for the carbon atoms.
	
	if getInitialValue:

		c0 = prU.getTimeSeries(dataPathName, initFileName, atomType)

		if getVelocity:

			v0 = prU.getTimeSeries(dataPathName, initFileName, atomType, type = 'Velocity')

	# Get the coordinates for the carbon atoms for the rest of the trajectory.
	
	c1 = prU.getTimeSeries(dataPathName, fileName, atomType)

	# Optionally get the velocity. If not, return an array of zeros.
	
	if getVelocity:

		v1 = prU.getTimeSeries(dataPathName, fileName, atomType, type = 'Velocity')
		
	else:
	
		v1 = np.zeros(c1.shape)

	# We return the coordinates and velocities.
	
	if getInitialValue:

		# Add a frame for the initial condition, if necessary.
		# The size of c1 is the 3 x (number of frames) x (number of carbon atoms)
		
		(numX, numF, numA) = c1.shape
		coords = np.zeros((numX, numF+1, numA))
		vel = np.zeros((numX, numF+1, numA))

		coords[:,0,:] = c0[:,-1,:]
		coords[:,1:,:] = c1

		if getVelocity:

			vel[:,0,:] = v0[:,-1,:]
			vel[:,1:,:] = v1

		return (coords, vel)
		
	else:
	
		return (c1, v1)

# Pare down the coordinates by using a particular stride or range of the data.
# We also use the stride and range information to construct a path for saving
# figures and for constructing strings for comments and titles. Finally, we
# calculate the duration of the data in ps.

def pareChainCoords(dataPathName, fileName, origCoords, origVel, theStride, theRange):

	# Get the time step and the dcd frequency from the NAMD log file.

	(tS, dcdF, numSteps) = prU.readNAMDLog(dataPathName, fileName)

	# If the range is empty, use all the data. Add the range and stride info
	# to the relevant path and comment strings.
	
	if theRange == ():

		duration = int((tS * numSteps) / 1000)
		strideComment = 'Range = (0, ' + str(duration) + ')'
		picturePath = 'Range = (0, ' + str(duration) + ')'

	else:

		duration = theRange[1] - theRange[0]
		strideComment = 'Range = (' + str(theRange[0]) + ', ' + str(theRange[1]) + ')'
		picturePath = 'Range = (' + str(theRange[0]) + ', ' + str(theRange[1]) + ')'

	strideComment = strideComment + ', Stride = ' + str(theStride)
	picturePath = picturePath + '/' + 'Stride = ' + str(theStride)

	# Pare the coordinate data to account for the range.
	# The range is given in ps, so we need to calculate frame numbers.

	if theRange != ():

		startFrame = int((theRange[0] * 1000) / (tS * dcdF))
		endFrame = int((theRange[1] * 1000) / (tS * dcdF)) + 1
		newCoords = origCoords[:, startFrame:endFrame, :]
		newVel = origVel[:, startFrame:endFrame, :]

	else:

		newCoords = origCoords
		newVel = origVel

	# Account for the stride. Should be able to do this in prody, but it doesn't
	# work right now and I don't feel like trying to update it to see if it's fixed.
	# Note! This comment is very old!

	if theStride > 1:

		coords = newCoords[:,0::theStride,:]
		vel = newVel[:,0::theStride,:]

	else:

		coords = newCoords
		vel = newVel

	# Find the number of atoms and the number of frames.

	numFrames = coords.shape[1]
	numAtoms = coords.shape[2]

	# We have the time step and output frequency from the NAMD log file. We have
	# already divided the time step by 1000 to convert from fs to ps.
	# We now construct a time array to use for figures, etc.

	if theRange == ():

		time = np.linspace(0, duration, numFrames)

	else:

		time = np.linspace(theRange[0], theRange[0] + duration, numFrames)
		
	# Return all the stuff we've calculated or modified.

	return (time, coords, vel, numAtoms, numFrames, duration, strideComment, picturePath)

def getDihedralChain(coords, radian=False, ring=False):
	"""Returns the dihedral angle in degrees for an alkane chain."""

	n = coords.shape[0]

	if ring:

		d = np.zeros((n+3, 3))

		d[0:n,:]=coords
		d[n:,:]=coords[0:3,:]

		n = n + 3

	else:

		d = coords

	c1 = d[0:(n-3)]
	c2 = d[1:(n-2)]
	c3 = d[2:(n-1)]
	c4 = d[3:n]

	a1 = c2 - c1
	a2 = c3 - c2
	a3 = c4 - c3

	v1 = np.cross(a1, a2)
	v1 = v1 / np.linalg.norm(v1, axis = -1, keepdims = True)
	v2 = np.cross(a2, a3)
	v2 = v2 / np.linalg.norm(v2, axis = -1, keepdims = True)

	porm = np.sign((v1 * a3).sum(-1))

	temp = (v1*v2).sum(-1) / ((v1**2).sum(-1) * (v2**2).sum(-1))**0.5

	temp[temp < -1] = -1
	temp[temp > 1] = 1

	rad = np.arccos(temp)
	rad[porm == -1] = -rad[porm == -1]
	rad[rad < 0] = rad[rad < 0] + 2*np.pi

	if radian:
		return rad
	else:
		return rad * 180 / np.pi

def getAngleChain(coords, radian=False):
	"""Returns bond angle in degrees."""

	n = coords.shape[0]

	c1 = coords[0:(n-2)]
	c2 = coords[1:(n-1)]
	c3 = coords[2:n]

	r1 = c1 - c2
	r2 = c3 - c2

	D = np.sum(r1.conj()*r2, axis=-1)
	M = np.linalg.norm(r1, axis=-1)*np.linalg.norm(r2, axis=-1)

	theta = abs(np.arccos(D/M))

	if radian:
		return theta
	else:
		return theta * 180 / np.pi

def getGyrationRadius(coords):
	"""Returns the radius of gyration for a single species of atom."""

	n = coords.shape[1]

	cMean = np.mean(coords, axis = 1)

	newCoords = coords - np.reshape(np.tile(cMean, n), (n, 3)).T

	R2 = np.sum(newCoords**2)

	return np.sqrt(R2/n)

def getPrincipalMoments(coords, vectors = False):
	"""Returns the principal moments of inertia for a single species of atom."""

	n = coords.shape[1]

	cMean = np.mean(coords, axis = 1)

	newCoords = coords - np.reshape(np.tile(cMean, n), (n, 3)).T

	M = np.matmul(newCoords, newCoords.T)

	Tr = np.trace(M)

	I = np.zeros((3,3))
	np.fill_diagonal(I, Tr)

	I = I - M

	Ip, v = np.linalg.eig(I)

	if vectors:

		return (Ip, v)

	else:

		return Ip

def getShapeParameters(coords):
	"""Returns the asphericity and anisotropy parameters for a single species of atom."""

	n = coords.shape[1]

	cMean = np.mean(coords, axis = 1)

	newCoords = coords - np.reshape(np.tile(cMean, n), (n, 3)).T

	Q = np.matmul(newCoords, newCoords.T)/n

	TrQ = np.trace(Q)
	I = np.zeros((3,3))
	np.fill_diagonal(I, TrQ)
	QHat = Q - I/3

	TrQHat2 = np.trace(np.matmul(QHat, QHat))

	Delta = 1.5 * TrQHat2 / TrQ**2

	Sigma = 4 * (np.linalg.det(QHat)) / (np.sqrt(2*TrQHat2/3))**3

	return (Delta, Sigma)

def getDeltaDerivative(coords, vel):
	"""Returns the asphericity and its time derivative."""

	n = coords.shape[1]

	# Velocity is in m/s, so we'll convert to A/ps.

	vel = vel * 1e10 / 1e12

	cMean = np.mean(coords, axis = 1)
	vMean = np.mean(vel, axis = 1)

	newCoords = coords - np.reshape(np.tile(cMean, n), (n, 3)).T
	newVel = vel - np.reshape(np.tile(vMean, n), (n, 3)).T

	Q = np.matmul(newCoords, newCoords.T)/n
	V = np.matmul(newVel, newVel.T)/n

	TrQ = np.trace(Q)
	TrV = np.trace(V)

	IQ = np.zeros((3,3))
	np.fill_diagonal(IQ, TrQ)
	IV = np.zeros((3,3))
	np.fill_diagonal(IV, TrV)

	QHat = Q - IQ/3
	VHat = V - IV/3

	TrQHat2 = np.trace(np.matmul(QHat, QHat))
	TrQHatVHat = np.trace(np.matmul(QHat, VHat))

	Delta = 1.5 * TrQHat2 / TrQ**2
	DeltaDot = 3 * (TrQHatVHat / TrQ**2 - TrQHat2 * TrV / TrQ**3)

	return (Delta, DeltaDot)

def getDistanceChain(coords, ring = True, spacing = 3, useAvg = True):
	"""Returns the distance between carbons for an alkane chain."""

	n = coords.shape[0]

	# Note that this won't work unless ring is True ...

	if ring:

		if useAvg:

			d = (coords + np.roll(coords, -1, 0))/2

		else:

			d = coords

		d = np.roll(d, -1, 0)

		numShift = int((spacing-1)/2)

		cl = np.zeros(d.shape)
		cr = np.zeros(d.shape)

		cl[0:numShift,:] = d[(0-numShift):,:]
		cl[numShift:n,:] = d[0:(n-numShift),:]

		cr[0:(n-numShift),:] = d[(0+numShift):n,:]
		cr[(n-numShift):n,:] = d[0:numShift,:]

	else:

		if useAvg:

			d = coords[0:(n-1),:] + np.diff(coords, axis=0)/2

		else:

			d = coords

		cl = d[0:(n-spacing+1)]
		cr = d[(spacing-1):n]

	dist = np.linalg.norm(cr-cl, axis = -1)

	return dist

def getCurvatureChain(coords, ring = True, spacing = 3, useAvg = True):
	"""Returns the curvature between carbons for an alkane chain."""

	n = coords.shape[0]

	# Note that this won't work unless ring is True ...

	if ring:

		if useAvg:

			d = (coords + np.roll(coords, -1, 0))/2

		else:

			d = coords

		d = np.roll(d, -1, 0)

		numShift = int((spacing-1)/2)

		cl = np.zeros(d.shape)
		cr = np.zeros(d.shape)
		cm = d

		cl[0:numShift,:] = d[(0-numShift):,:]
		cl[numShift:n,:] = d[0:(n-numShift),:]

		cr[0:(n-numShift),:] = d[(0+numShift):n,:]
		cr[(n-numShift):n,:] = d[0:numShift,:]

	else:

		if useAvg:

			d = coords[0:(n-1),:] + np.diff(coords, axis=0)/2

		else:

			d = coords

		cl = d[0:(n-spacing+1)]
		cr = d[(spacing-1):n]

	A = np.linalg.norm(cr-cl, axis = -1)
	B = np.linalg.norm(cm-cl, axis = -1)
	C = np.linalg.norm(cr-cm, axis = -1)

	p = (A + B + C) /2

	area = np.sqrt(p*(p-A)*(p-B)*(p-C))

	radius = (A * B * C) / (4*area)

	return 1/radius

def getTemperature(vel, mass):
	"""Returns the temperature as calculated from the velocity of the atoms."""

	kB = 1.3806488e-23

	KE = 0.5 * mass * 3 * np.mean(vel*vel)

	return (2 * KE / (3 * kB))
