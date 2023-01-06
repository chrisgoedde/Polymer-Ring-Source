import numpy as np
import copy
import spanningGroups as sG

def groupZones(M, zoneDict, time, theLevel):
    """Conduct grouping based on the inching algorithm"""

    Si = None
    inching = True
    first = True
    terminated = []
    residuals = []

    # Inherrited variables from prior algorithm iterations
    colors = []

    topDict = genTopDict(time, M)

    (groupList, groupSizeList, groupMap, currentGroups) \
        = newGroupList(M)

    (jumpMatrix, transMatrix, balMatrix, netFluxMatrix) \
        = newMatrices(groupList, groupSizeList, groupMap, M)

    # Main while loop, as long as we are inching forward we are in this loop
    while inching:

        # We'll place our index at the first element of M
        if first:

            # Instantiate our root span
            Si = sG.Span(M[0], M)

            # i will be iteratively moved as we inch forward,
            # think of it as a pointer to the elemnt that we are interested in
            i = 0
            first = False


        elif i == Si.mains[1]:

            # Have we ammased a group of residual spans?
            if residuals:

                handled = []

                # Looping through the residuals
                for idx in residuals:

                    # Sj is the current residual span that we are interested in
                    Sj = sG.Span(idx, M)
                    # Check if the delta is now negative
                    delta = sG.compDelta(Si, Sj, M)

                    # If it is, we need to join it to the root span
                    if delta < 0:

                        combineGroups(groupMap[int(Sj.S[0])], groupMap[int(Si.S[0])], groupList,
                                      groupSizeList, groupMap, currentGroups,
                                      jumpMatrix, transMatrix, balMatrix,
                                      netFluxMatrix)

                        # Replace all instances of element j with the root span element, i
                        instances = np.where(M == Sj.S[0])
                        M[instances] = Si.S[0]

                        # Recompute the root span
                        Si = sG.Span(Si.S[0], M)

                        handled.append(idx)

                # Remove the residuals that we have joined to Si
                # from the considered set of residual spans
                for entry in handled:
                    residuals.remove(entry)

                # If, after joining some residual spans, our last
                # main index has not changed we need to study
                # the rest of the residuals and see which one is
                # the optimal candidate for being the root span
                if i == Si.mains[1]:

                    if residuals:

                        # Remove the root span element from any future joining consideration
                        terminated.append(Si.S[0])

                        # Use a dissimilarity measure to pick out the next root span
                        nextI = sG.disMax(Si, residuals, M)

                        # Move the pointer i to the first main index of the new
                        # root span
                        i = np.where(M == nextI)
                        i = i[0][0]

                        # Instantiate new root span
                        Si = sG.Span(M[i], M)

                        # Clear out the residual groups
                        residuals = []

                    # If the last main index has not changed, and we don't have
                    # any residual spans to consider we'll simply choose the next
                    # root span to be the span associated with element M_{i+1}
                    else:
                        if (i + 1) < len(M) - 1:
                            terminated.append(Si.S[0])
                            i += 1
                            Si =sG.Span(M[i], M)

                        else:
                            # If we are here then we have met the end of
                            # of the main span, i.e. i == N
                            inching = False
            else:
                # If we have no residuals, then assign the next root
                # to be the span associated with element M_{i+1}
                if (i + 1) < len(M) - 1:
                    terminated.append(Si.S[0])
                    i += 1
                    Si =sG.Span(M[i], M)
                else:
                    # We cannot inch forward any further, stop inching
                    inching = False

        elif i >= len(M):
            # We have met the end of the main span, stop inching
            inching = False

        else:
            # We'll continue to inch forward and group using the current root span
            while i < Si.mains[1]:

                # "Inch Forward"
                i += 1
                print(i)

                # If the current element is not associated with a temrinated span or our
                # root span, we'll consider it for grouping
                if (M[i] not in terminated) and (M[i] not in residuals) and (M[i] != Si.S[0]):

                    # Instantiate Sj
                    Sj = sG.Span(M[i], M)
                    # Compute the activity delta
                    delta = sG.compDelta(Si, Sj, M)

                    # Is the activity delta negative?
                    if delta < 0:
                        # Join the two groups and change all instances of element j to
                        # element i

                        combineGroups(groupMap[int(Sj.S[0])], groupMap[int(Si.S[0])], groupList,
                                      groupSizeList, groupMap, currentGroups,
                                      jumpMatrix, transMatrix, balMatrix,
                                      netFluxMatrix)

                        instances = np.where(M == Sj.S[0])
                        M[instances] = Si.S[0]
                        Si = sG.Span(Si.S[0], M)
                        terminated.append(M[i])
                    else:
                        # If not, add it to the residual group,
                        # we will revisit later
                        residuals.append(M[i])

    addGrouptoDict(topDict, M, time, groupList,
                   groupSizeList, currentGroups, jumpMatrix,
                   transMatrix, balMatrix, netFluxMatrix,
                   theLevel, colors)

    return topDict

def newGroupList(zoneList):
    groupList = []
    groupSizeList = []
    groupMap = {}
    groupNum = 0

    # groupNum tracks the index of the list of groups.

    # Find the largest zone number that is occupied.

    maxNum = int(np.max(zoneList))

    # Iterate through all the zone numbers up to the maximum occupied zone.

    for z in range(maxNum + 1):

        # Find the number of data points in the current zone.

        num = np.sum(zoneList == z)

        # If the zone is occupied, make it a group and record the number of
        # data points in the zone.

        if num > 0:
            groupList.append([z])

            groupSizeList.append(num)

            groupMap[z] = groupNum

            groupNum = groupNum + 1

    currentGroups = [i for i in range(len(groupList))]

    # Return the list of groups and the list of group sizes.

    return groupList, groupSizeList, groupMap, currentGroups

def newMatrices(groupList, groupSizeList, groupMap, zoneList):
    # groupList is the list of groups.
    # groupSizeList is the list of group sizes.
    # groupMap is a map of groups to zones.
    # zoneList is the data set of zones for the trajectory being analyzed.

    # The size of the jump matrix depends on the number of groups.

    numGroups = len(groupList)

    # We need to know the total number of data points for the detailed balance matrix.

    totalPoints = sum(groupSizeList)

    # Initialize the jump matrix as all zeros. When finished, this matrix holds
    # the number of jumps from group A to group B.

    jumpMatrix = np.zeros((numGroups, numGroups))

    # Go through the zone arrays in zoneDict one-by-one, adding each trajectory
    # to the jump matrix. This means that the jump matrix combines the data from
    # all the runs contained in zoneDict.

    # Find the group of the first data point. sG is the starting group, eG is the ending
    # group. The matrices are column stochastic, so jumpMatrix[eG, sG] is the number of
    # jumps from starting group sG to ending group eG.

    start = 0
    sG = groupMap[zoneList[start]]

    # Go through the data point-by-point.

    for end in range(1, len(zoneList)):
        # Find the group of the next point and update the jump matrix.

        eG = groupMap[zoneList[end]]

        jumpMatrix[eG, sG] = jumpMatrix[eG, sG] + 1

        # Move on to the next point.

        sG = eG

    # Calculate the transition and balance matrices from the jump matrix.

    transMatrix = np.zeros(jumpMatrix.shape)
    balMatrix = np.zeros(jumpMatrix.shape)
    netFluxMatrix = np.zeros(jumpMatrix.shape)

    # transMatrix[eG, sG] is the probability of jumping from starting group sG to ending
    # group eG. To calculate this, we divide the elements of each column of jumpMatrix
    # by the sum of that column.

    # balMatrix[eG, sG] is the flux from starting group sG to ending group eG. It is the
    # transition rate from sG to eG times the probability of being in group sG.

    for sG in range(numGroups):
        transMatrix[:, sG] = jumpMatrix[:, sG] / np.sum(jumpMatrix[:, sG])
        balMatrix[:, sG] = transMatrix[:, sG] * groupSizeList[sG] / totalPoints

    # netFluxMatrix[eG, sG] is the net flux from starting group sG to ending group eG.
    # If it is postive, then points are moving from sG to eG. This matrix is always
    # skew symmetric.

    netFluxMatrix = balMatrix - np.transpose(balMatrix)

    return jumpMatrix, transMatrix, balMatrix, netFluxMatrix


def combineGroups(g1, g2, groupList, groupSizeList, groupMap, \
                  currentGroups, jumpMatrix, transMatrix, balMatrix, netFluxMatrix):
    # np.set_printoptions(precision=0, suppress=True, linewidth=200)

    # print('Combining groups ' + str(g1) + ' and ' + str(g2))

    totalPoints = sum(groupSizeList)

    #

    # Add the first group to the second and update groupMap.

    for g in groupList[g1]:
        groupList[g2].append(g)
        groupMap[g] = g2

    groupSizeList[g2] = groupSizeList[g2] + groupSizeList[g1]

    groupList[g1] = []

    groupSizeList[g1] = 0
    currentGroups.remove(g1)

    # Add the jumps from the first group to the second group

    jumpMatrix[:, g2] = jumpMatrix[:, g2] + jumpMatrix[:, g1]
    jumpMatrix[g2, :] = jumpMatrix[g2, :] + jumpMatrix[g1, :]

    # Set the removed row and column to zero.

    jumpMatrix[:, g1] = 0
    jumpMatrix[g1, :] = 0

    # Calculate the new rows and columns of the transition and balance matrices.

    transMatrix[:, g2] = jumpMatrix[:, g2] / np.sum(jumpMatrix[:, g2])
    balMatrix[:, g2] = transMatrix[:, g2] * groupSizeList[g2] / totalPoints

    # print('Currently have ' + str(len(currentGroups)) + ' groups: ' + str(currentGroups))

    for sG in currentGroups:
        transMatrix[g2, sG] = jumpMatrix[g2, sG] / np.sum(jumpMatrix[:, sG])
        balMatrix[g2, sG] = transMatrix[g2, sG] * groupSizeList[sG] / totalPoints

    # Set the probabilities for the removed groups to zero in all the probability matrices.

    transMatrix[g1, :] = 0
    transMatrix[:, g1] = 0
    balMatrix[g1, :] = 0
    balMatrix[:, g1] = 0
    netFluxMatrix[g1, :] = 0
    netFluxMatrix[:, g1] = 0

    for sG in currentGroups:
        netFluxMatrix[:, sG] = balMatrix[:, sG] - balMatrix[sG, :]


def addGrouptoDict(topDict, zoneList, time, groupList,
                   groupSizeList, currentGroups, jumpMatrix, transMatrix,
                   balMatrix, netFluxMatrix, levels, colors, reorder=False):
    # Find the number of active groups.

    numCurrent = len(currentGroups)

    print('Adding ' + str(numCurrent) + ' groups to dictionary')

    newTransMatrix = np.zeros((numCurrent, numCurrent))
    newJumpMatrix = np.zeros((numCurrent, numCurrent))
    newBalMatrix = np.zeros((numCurrent, numCurrent))
    newConMatrix = np.zeros((numCurrent, numCurrent))
    netFlux = np.zeros((1, numCurrent))

    # This array will hold the group number for each point in the trajectory.

    groups = np.zeros(zoneList.shape)

    # Go through the currently active groups.

    for i in range(numCurrent):

        c = currentGroups[i]

        # Go through the zones in each group

        for g in groupList[c]:
            # Assign a group number to the appropriate points in the trajectory.

            groups[zoneList == g] = i

        # Create the matrices to be saved.

        for j in range(numCurrent):
            d = currentGroups[j]

            newTransMatrix[i, j] = transMatrix[c, d]
            newBalMatrix[i, j] = balMatrix[c, d]
            newJumpMatrix[i, j] = jumpMatrix[c, d]
            newConMatrix[i, j] = netFluxMatrix[c, d]

    netFlux = np.sum(newBalMatrix, axis=0) - np.sum(newBalMatrix, axis=1)
    netFlux = netFlux * time[-1]

    topDict['groups'][numCurrent] = groups
    topDict['transMatrix'][numCurrent] = newTransMatrix
    topDict['balMatrix'][numCurrent] = newBalMatrix
    topDict['jumpMatrix'][numCurrent] = newJumpMatrix
    topDict['netFluxMatrix'][numCurrent] = newConMatrix
    topDict['groupList'][numCurrent] = copy.deepcopy([groupList[i] for i in currentGroups])
    topDict['groupSizeList'][numCurrent] \
        = copy.deepcopy([groupSizeList[i] for i in currentGroups])
    topDict['netFlux'][numCurrent] = netFlux

    if colors != []:
        topDict['colors'][numCurrent] = copy.deepcopy([colors[i] for i in currentGroups])


def genTopDict(time, zoneList):
    """Create an empty topDict object"""

    topDict = {'time': time, 'zones': zoneList, 'groups': {}, 'jumpMatrix': {}, 'transMatrix': {}, 'balMatrix': {},
               'conMatrix': {}, 'netFlux': {}, 'netFluxMatrix': {}, 'colors': {}, 'groupList': {}, 'groupSizeList': {},
               'currentGroups': {}}

    return topDict