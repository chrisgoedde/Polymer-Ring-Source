import numpy as np
import copy
import plotZones as pz


class Span(object):

    def __init__(self, zone, M):
        """Constructor function"""

        # Instances is a tuple of idexes.
        # (i,j,k,...) | finite, lenght n
        # These indexes mark the appearance of "zone" in M
        instances = np.where(M == zone)

        if len(instances[0]) == 1:
            # Do we have a span of unit length? Or, there is one occurrence of zone
            self.mains = tuple((instances[0][0], instances[0][0]))
            self.homogenous = True
        elif len(instances[0]) == 0:
            # Was the zone number that was passed correct?
            raise "A wrong zone value has been passed"
        else:
            # Resorting case, there is at least two insrances of zone.
            # We will use the first and last index in "instances" as
            # time indices of the span.
            self.mains = tuple((instances[0][0], instances[0][-1]))

        # S is a numpy array, and thus has all of its functionalities.
        # It will always have dimensions (1xn) where
        #           n=(mains[1]-mains[0]) + 1
        self.S = M[self.mains[0]:self.mains[1] + 1]

        # The activity is a quantity that characterizes the span.
        # see the compA definition to see its mathematical definition.
        # It is used to capture the stability of the span.
        activity = compA(self.S)
        if activity:
            self.A = activity
            self.homogenous = False
        else:
            self.A = activity
            self.homogenous = True

    def plot(self, outDir, fName, zoneDict, title='', color=None):
        """Visualize the span in the R-Theta Plane"""
        if color is None:
            color = [1, 0, 0, 0.5]
        pz.drawSpan(outDir, fName, title, self, zoneDict, color)


def compA(span):
    """Compute the activity value of the passed span, S"""
    if len(span) == 1:
        return 0
    else:
        S = copy.deepcopy(span)
        # We need to add 1 to each zone because 0 will be
        # a relevant value in the difference between l and r
        S = S + 1
        l = S[0:len(S) - 1]
        r = S[1:len(S)]
        # Compute the number of transitions contained within the span,
        # Beta
        trans = l - r
        beta = np.sum(trans != 0)
        # Compute the number of unique zones in the span,
        # alpha
        alpha = len(set(S))
        # Return the activity value
        return (alpha * beta) / ((len(S) - 1) ** 2)


def getSpans(M):
    """Return the set of spans contained in the time series M|(set)"""

    # Instantiate return object
    retSet = set()

    # Create a set of the unique zones present in the time series
    zones = set(M)

    # Iterate through the zones, create a Span associated with each one
    for z in zones:
        retSet.add(Span(z, M))

    return retSet


def getSiMap(SiSet):
    """Return a mapping between the zone labels and a set of Spans"""

    # Instantiate return object
    retDict = {}

    # Let n denote the number of spans contained in the span set
    n = len(SiSet)

    # temporary list object I'll use for accessing the spans
    SiList = list(SiSet)

    # Check if we were passed some nonsense, i.e. an empty span set
    if n == 0:
        raise "Passed empty span set"
    else:
        # The passed set is non-empty, proceed
        for i in range(1, n + 1):
            # Loop through the neutral numbers, up to n, and assign
            # a label to each span in the past set corresponding
            # to the number that "i" is.
            retDict.update({SiList[i - 1].S[0]: SiList[i - 1]})

    return retDict


def getSiTmap(SiMap, M):
    """Return a mapping between the time steps in a time series and a set of spans"""

    # How many time steps are in the time series we were passed?
    T = len(M)

    # temporary variable, holds time step labels (1, 2, 3,...,T)
    t = np.arange(0, T)

    # Instantiate return object
    retDict = dict.fromkeys(t, [])

    # Now, we will procedurally build out the map by iterating
    # through each span and "placing" it under any time in between
    # its time indices

    for zone in SiMap.keys():
        # t1 and t2 denote the first and last appearance of this span, respectively
        t1 = SiMap[zone].mains[0]
        t2 = SiMap[zone].mains[1]

        query = np.arange(t1, t2 + 1)

        for t in query:
            retDict[t].append(zone)

    # Given that the keys of the dictionary at the top loop are all unique,
    # we do not need to worry about any instances of double recording entries.

    return retDict


def getContainers(M, SiMap):
    """Return the set of containing spans present in the spanSet"""

    # Recorded will hold all of the defining zones that we have
    # already looked at
    recorded = []

    # This set will contain all the spans that are contained within
    # a container
    contained = set()

    # containDict will be diciontary with keys as zones, thus containDict[key]
    # return the spans that are contained in particular container associated
    # with the key
    containDict = {}

    # Iterate through the time series
    for i in range(0, len(M)):
        if M[i] not in recorded:
            # Let Si refer to the current span that were concerning ourselves with
            Si = SiMap[M[i]]
            # Let contains be the set holding all the spans that Si contains, if any
            contains = set()
            # Get the beginning time of Si, t1, and the ending time of Si, t2
            t1 = Si.mains[0]
            t2 = Si.mains[1]

            # Get all the zones appearing to the left of t1
            left = set(M[0:t1])
            # Get all the zones appearing to the right of t2
            right = set(M[t2 + 1:len(M)])
            # Get all the zones appearing between t1 and t2
            inter = set(Si.S)

            # Make sure we have no instance of the current zone
            # in any of the sets
            left = left.difference({M[i]})
            right = right.difference({M[i]})
            inter = inter.difference({M[i]})

            # Now, any span that is contained in Si will, by definition,
            # only be contained in "inter". How can we find the spans that
            # satisfy this criterion? We use some basic set theory:
            #
            # contained = (inter-right) /\ (inter-left)
            # Where the "-" refers to a set difference

            # (inter - right)
            temp1 = inter.difference(right)
            # (inter - left)
            temp2 = inter.difference(left)
            # (inter-right) /\ (inter-left)
            temp3 = temp1.intersection(temp2)

            # add all the spans in tmep3 to contains
            contains = contains.union(temp3)
            # add all the spans in temp3 to the gloabl contained set
            contained = contained.union(contains)

            containDict.update({M[i]: contains})
            recorded.append(M[i])

    # Remove all the spans which are contained in a container as a key
    # from the containDict key set
    containers = set(SiMap.keys())
    containers = containers.difference(contained)
    for key in contained:
        containDict.pop(key, None)

    retContainers = {}

    # Return a zone map of all the containers
    for item in containers:
        retContainers.update({item: SiMap[item]})

    return retContainers, containDict


def getDense(SiMap, M):
    """Return the density array for the given span set"""

    # The D array will have dimensions 1 x T, where
    # T denotes the number of time steps present in the
    # time series
    D = np.zeros((len(M),))

    for key in SiMap.keys():
        # Let Si denote the current span we are considering
        Si = SiMap[key]

        # Get the beginning time of Si, t1, and the ending time of Si, t2
        t1 = Si.mains[0]
        t2 = Si.mains[1]

        # Add 1 to the times in the density array, indicating that
        # span Si is present at those times
        D[t1:t2 + 1] += 1

    return D


def getLength(SiMap, M):
    """Return the length array for the given span set"""

    # The L array will have dimensions 1 x T, where
    # T denotes the number of time steps present in the
    # time series
    L = np.zeros((len(M),))

    for key in SiMap.keys():
        # Let Si denote the current span we are considering
        Si = SiMap[key]

        # Get the beginning time of Si, t1, and the ending time of Si, t2
        t1 = Si.mains[0]
        t2 = Si.mains[1]

        # Add the length of Si to all the times upon which it is defined
        L[t1:t2 + 1] += len(Si.S)

    # Return the summed lengths averaged over all spans
    return L / len(SiMap.keys())


def getActivity(SiMap, M):
    """Return the activity array for the given span set"""

    # The A array will have dimensions 1 x T, where
    # T denotes the number of time steps present in the
    # time series
    A = np.zeros((len(M),))

    for key in SiMap.keys():
        # Let Si denote the current span we are considering
        Si = SiMap[key]

        # Get the beginning time of Si, t1, and the ending time of Si, t2
        t1 = Si.mains[0]
        t2 = Si.mains[1]

        # Add the activity of Si to all the times upon which it is defined
        A[t1:t2 + 1] += Si.A

    # Return the summed activity values over all spans
    return A / len(SiMap.keys())


def plotSpans(outDir, fName, title, spans, zoneDict, colors):
    """Plot multiple spans in the R-Theta Plane"""
    pz.drawSpan(outDir, fName, title, spans, zoneDict, colors, multi=True)
