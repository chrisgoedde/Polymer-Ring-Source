import chainZones as cZ
import plotUtils as plU

import matplotlib.pyplot as plt
import numpy as np

def drawZones(picturePathName, zoneDict, shape = 'Wedge', fileName = 'Zones', \
				titleString = ''):

	if shape == 'Wedge':

		f, ax = plU.makeFigure(figsize = (6,6), textSize = 10, polar = True)

		ax.set_xticks([np.pi/3, 2*np.pi/3])
		ax.set_xticklabels(['Oblate','Prolate'])
		ax.set_thetamin(60)
		ax.set_thetamax(120)
		ax.set_rmax(1)
		ax.set_rticks([0.25, 0.5, 0.75, 1])
		plt.text(0.75, 0.5, 'R', rotation=60)

		theLevel = int(np.log2(len(zoneDict))/2)

		for z in zoneDict.keys():

			plU.drawWedge(ax, zoneDict[z]['xmin'], zoneDict[z]['xmax'], \
					zoneDict[z]['ymin'], zoneDict[z]['ymax'], lined = True)

			plt.text(2*np.pi/3-(zoneDict[z]['xmin'] + zoneDict[z]['xmax'])/2, \
					(zoneDict[z]['ymin'] + zoneDict[z]['ymax'])/2, \
					cZ.intToKey(z, theLevel), ha = 'center', va = 'center', size = 10-2*theLevel)

	if titleString != '':

		plt.title(titleString, fontsize = 10)

	if fileName != '':

		plU.savePlot(picturePathName, fileName, \
					type = 'png', dpiArg = 600, closeFigure = True)
