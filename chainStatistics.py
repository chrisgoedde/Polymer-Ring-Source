import numpy as np

# Some constants for the dihedral groups.
# trans -> trans
# gP -> gauche plus
# gM -> gauche minus

trans = 0
gP = +1
gM = -1

#%% These three procedures calculate statistics for the chain, not the zones.

# Calculate some of the important statistics a bout the chain, its segments,
# and the dihedral angles.

def getSegmentStatistics(numFrames, numAtoms, phi):

	# The input arguments are the number of frames, the number of angles, and a
	# (numFrames x numAtoms) array holding the dihedral angles as a function of time.

	# Set up the dictionary of statistics. The keys are:
		# 'kinks': the distribution of kinks (rotated dihedrals) around the ring
		# 'dist': the segment distribution
		# 'frac': the fraction of the ring taken up by each segment length
		# 'num': the number of segments
		# 'avgLen': the average length of the segments
		# 'max': the most common segment
		# 'rev': the number of reversals for terminal dihedral
		# 'long': the longest segment
		# 'even': the fraction of the ring composed of even segments
		# 'odd': the fraction of the ring composed of odd segments
	# The first three of these are (numFrames) x (numAtoms) arrays, the rest
	# are arrays with numFrames elements.
	# For the first three:
	# Each frame of 'kinks' is a numAtoms array of (-1, 0, 1) corresponding
	# to (gauche minus, trans, gauche plus) for each dihedral.
	# Each frame of 'dist' is a numAtoms array of positive and negative integers
	# corresponding to the length of the segment that each atom belongs to.
	# Positive integers mean that the segment began (or ended) with a gauche
	# plus dihedral, negative integers means that the segment began (or ended)
	# with a gauche minus dihedral.
	# Each frame of 'frac' is an array holding the fraction of the chain occupied
	# by each possible segment length, starting with 0 and counting upward.
	
	d = getSegmentDictionary(numFrames, numAtoms)

	d['kinks'][phi-180 > 60] = gP
	d['kinks'][180-phi > 60] = gM

	for t in range(numFrames):

		d['dist'][t,:], d['frac'][t,:], d['num'][t], d['rev'][t] \
			= getKinkSegments(d['kinks'][t,:])
		d['max'][t] = np.argsort(-d['frac'][t,:])[0]
		d['avgLen'][t] = numAtoms/d['num'][t]
		d['long'][t] = np.max(np.abs(d['dist'][t,:]))
		d['even'][t] = np.sum(d['frac'][t, 2:numAtoms+1:2])
		d['odd'][t] = np.sum(d['frac'][t, 1:numAtoms+1:2])

	return d

# A convenience method to initial a dictionary of statistics.

def getSegmentDictionary(numFrames, numAtoms):

	d = {}

	d['kinks'] = np.zeros((numFrames, numAtoms))
	d['dist'] = np.zeros((numFrames, numAtoms))
	d['frac'] = np.zeros((numFrames, numAtoms+1))

	for k in ('num', 'avgLen', 'max', 'rev', 'long', 'even', 'odd'):

		d[k] = np.zeros((numFrames,))

	return d

# Analyze the chain in terms of the number of segments and their length.

def getKinkSegments(kinkDist, kinkFirst = True):

	numAtoms, = kinkDist.shape

	# We define a segment to be a sequence of consecutive zeros followed or
	# preceded by a +1 or a -1, depending on the value of kinkFirst.
	# The shortest possible sequence is 1. The next shortest
	# is 01 or 10, and the next shortest is 001 or 100, etc.

	# We partition the ring up into segments, and keep track of which segment
	# each atom is part of.
	#
	# An element of segment has value +N if that atom is part of a segment of
	# length N that is terminated by +1. An element of segment has value -N
	# if it is part of a segment of length N terminated by -1.
	#
	# segmentCount keeps track of the total number of atoms as a function of
	# the length of the segment. We return segmentCount/numAtoms, which is the
	# fraction of the ring that is composed of each length of segment.

	segment = np.zeros(numAtoms,)
	segmentCount = np.zeros(numAtoms+1,)

	# numSegments is the number of segments
	# numReversals is the number of times we switch from a segment that ends
	# with a +1 (-1) to a segment that ends with a -1 (+1)

	numSegments = 0
	numReversals = 0

	# We march forward if kinkFirst is true, otherwise we march backward

	if kinkFirst:

		kinkStep = +1

	else:

		kinkStep = -1

	# First we make sure that there is at least one kink

	if np.sum(abs(kinkDist)) == 0:

		segment[:] = numAtoms
		segmentCount[numAtoms] = numAtoms
		numSegments = 1

		return segment, segmentCount/numAtoms, numSegments, numReversals

	# Look for the first kink, going backward if kinkFirst is true, and
	# forward if kinkFirst is false.

	first = 0
	while kinkDist[first] == 0:

		first = first - kinkStep

	# Now we march through the ring, going forwards or backwards, to take
	# advantage of negative indices. Our stopping point will be last.

	if kinkFirst:

		last = first + len(kinkDist)

	else:

		last = first - len(kinkDist)

	current = first
	next = first + kinkStep

	# We haven't identified the sign of the first segment, so we
	# set a marker

	lastSegmentSign = 0

	# March down the current segment, adding to its length so long
	# as the next dihedral is in group 2

	while current != last:

		# We know the segment has at least length 1

		segmentLength = 1

		# Look for the start of the next segment

		while next < numAtoms and kinkDist[next] == 0:

			segmentLength = segmentLength + 1
			next = next + kinkStep

		# We've gotten to the start of the next segment

		if kinkFirst:

			theRange = range(current, next)

		else:

			theRange = range(next+1, current+1)

		if kinkDist[current] == gM:

			segment[theRange] = -segmentLength

			if lastSegmentSign == gP:

				numReversals = numReversals + 1

			lastSegmentSign = gM

		else:

			segment[theRange] = segmentLength

			if lastSegmentSign == gM:

				numReversals = numReversals + 1

			lastSegmentSign = gP

		numSegments = numSegments + 1
		segmentCount[segmentLength] = segmentCount[segmentLength] + abs(current - next)

		current = next
		next = current + kinkStep

	return segment, segmentCount/numAtoms, numSegments, numReversals
