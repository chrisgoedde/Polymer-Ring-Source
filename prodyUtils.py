import prody as pd
import numpy as np

# Convenience method for reading coordinates from a pdb file

def getCoords(pathName, fileName, sel):

	# pathName is the path to the pdb folder
	# fileName is the name of the pdb file, without the extension
	# sel is a tuple containing the types (chain, atom, etc) and
	# names for the selection, in a single ordered pair, or it is a single
	# string to be passed along to prody

	# Read in the pdb file

	atoms = pd.parsePDB(pathName + '/' + fileName + '.pdb')

	# Make the desired selections

	atoms = makeSelection(atoms, sel)

	if atoms is None:

		return 0, ( np.array([]), np.array([]), np.array([]) )

	coords = atoms.getCoords()

	# Extract the coordinates, and return them along with the number of atoms

	nAtoms = coords.shape[0]
	r = np.array([coords[:, 0], coords[:, 1], coords[:, 2]])

	return r, nAtoms

# Convenience method for reading coordinates and velocities from dcd files

def getTimeSeries(pathName, fileName, sel, stride = 1, firstFrame = None, \
					lastFrame = None, type = 'Coord'):

	# velFactor converts namd internal units to A/ps
	# forceFactor converts namd internal units to N

	velFactor = 20.45482706
	forceFactor = 6.9477e-11

	pdbFile = pathName + '/' + fileName + '.pdb'

	if type == 'Velocity':

		# We will return the velocity in m/s

		scaleFactor = 100 * velFactor
		dcdFile = pathName + '/' + fileName + '.veldcd'

	elif type == 'Force':

		# We will return the force in pN

		scaleFactor = 1e12 * forceFactor
		dcdFile = pathName + '/' + fileName + '.forcedcd'

	else:

		# We will return the coordinates in A

		scaleFactor = 1.0
		dcdFile = pathName + '/' + fileName + '.dcd'

	# Now, open up the DCD file and get the ensembles for the selection

	atoms = pd.parsePDB(pdbFile, chain = 'PEM')
	coordEnsemble = pd.parseDCD(dcdFile, step = stride, start = firstFrame, stop = lastFrame)

	atoms = makeSelection(atoms, sel)

	if atoms is None:

		return np.array([]), np.array([])

	coordEnsemble.setAtoms(atoms)
	atomC = coordEnsemble.getCoordsets()

	coordX = scaleFactor * atomC[:, :, 0]
	coordY = scaleFactor * atomC[:, :, 1]
	coordZ = scaleFactor * atomC[:, :, 2]

	return np.array([ coordX, coordY, coordZ ])
	
def setRangeAndStride(dataPathName, fileName, theStride, theRange, \
						getInitialValue = True):

	# Get the time step and the dcd frequency from the NAMD log file.

	(tS, dcdF, numSteps) = readNAMDLog(dataPathName, fileName)

	# If the range is empty, use all the data. Add the range and stride info
	# to the relevant path and comment strings.
	
	if theRange == ():

		duration = int((tS * numSteps) / 1000)
		strideComment = 'Range = (0, ' + str(duration) + ')'
		picturePath = 'Range = (0, ' + str(duration) + ')'
		
		print('What?')
		if getInitialValue:
			startFrame = theStride-1
			print('???')
		else:
			startFrame = None
			
		endFrame = None

	else:

		duration = theRange[1] - theRange[0]
		strideComment = 'Range = (' + str(theRange[0]) + ', ' + str(theRange[1]) + ')'
		picturePath = 'Range = (' + str(theRange[0]) + ', ' + str(theRange[1]) + ')'

		startFrame = int((theRange[0] * 1000) / (tS * dcdF))
		endFrame = int((theRange[1] * 1000) / (tS * dcdF))

	strideComment = strideComment + ', Stride = ' + str(theStride)
	picturePath = picturePath + '/' + 'Stride = ' + str(theStride)

	return (startFrame, endFrame, duration, strideComment, picturePath)

def makeSelection(atoms, sel):

	# If sel is a tuple, paste the two parts together. Otherwise, assume
	# it's a string and pass it along.

	if type(sel) is tuple:

		atoms = atoms.select(sel[0] + " " + sel[1])

	else:

		atoms = atoms.select(sel)

	return atoms

# Let's parse the log file to get simulation parameters!

def readNAMDLog(pathName, fileName):

	logFileName = pathName + '/' + fileName + '.log'

	logFile = open(logFileName, "r")

	for line in logFile:

		if line.startswith("Info: TIMESTEP"):

			for s in line.split():

				if s.isdigit():

					timeStep = int(s)

		elif line.startswith("Info: DCD FREQUENCY"):

			for s in line.split():

				if s.isdigit():

					dcdFreq = int(s)

		elif line.startswith("TCL: Running for"):

			for s in line.split():

				if s.isdigit():

					numSteps = int(s)

	logFile.close()

	return (timeStep, dcdFreq, numSteps)
