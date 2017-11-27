from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from math import atan2,degrees,sin,cos,radians,copysign
from scipy.stats import norm
import warnings 
from matplotlib import collections as mc
"""
**********************************************
File: galtonWatsonTrees.py
Author: Luke Burks
Date: November 2017

Just plotting Poisson 
Distributions for a 
range of means

And Galton-Watson Tree plotting

**********************************************
"""


class Node:

	def __init__(self,parent,pose):
		self.parent = parent; 
		self.pose = pose; 
		self.children = [];
		self.dead = False
		self.depth = 0; 

		if(parent is not None):
			self.angle = self.findAngle(); 

	def findAngle(self):
		xDiff = self.pose[0]-self.parent.pose[0]; 
		yDiff = self.pose[1]-self.parent.pose[1]; 
		return degrees(atan2(yDiff,xDiff)); 

	def addChild(self,pose):
		a = Node(self,pose); 
		self.children.append(a); 
		return a; 
 	
 	def prune(self): 
 		self.dead = True;
 		self.parent.children.remove(self);  
 		self.parent = None; 

	def convertForPlot(self):
		a = [self.parent.pose[0],self.pose[0]]; 
		b = [self.parent.pose[1],self.pose[1]]; 
		return [a,b]; 

def plotTree(root,ax=None,leaves = True,pausing = 0):
	plt.cla();
	allNodes = runDFSAll(root);  
	allEx = runDFSEx(root); 

	allLines = []
	allWidths = []; 
	for nodee in allNodes:
		if(nodee is not root and nodee.dead == False):
			tmp = nodee.convertForPlot(); 
			if(ax is not None):
				#plt.plot(tmp[0],tmp[1],linewidth=10-nodee.depth,color='xkcd:brown',solid_capstyle='round');
				allLines.append([(tmp[0][0],tmp[1][0]),(tmp[0][1],tmp[1][1])]); 
				allWidths.append(10-nodee.depth); 
			else:
				allLines.append([(tmp[0][0],tmp[1][0]),(tmp[0][1],tmp[1][1])]); 
				#plt.plot(tmp[0],tmp[1],color = 'b');


			if(leaves and nodee in allEx and ax is not None):
				circ = plt.Circle((nodee.pose[0],nodee.pose[1]),1.5,facecolor='g',alpha = 0.75); 
				ax.add_patch(circ);
	if(len(allWidths) == 0):
		allWidths = 1; 
	lc = mc.LineCollection(allLines,colors = 'xkcd:brown',linewidths=allWidths); 
	#lc = mc.LineCollection(allLines); 
	ax.add_collection(lc); 

	if(pausing == 0):
		ax.autoscale();
		plt.show(); 
	else:
		plt.xlim([-12,12]); 
		plt.ylim([0,16]); 
		plt.pause(pausing); 

#grabs all nodes in tree
def runDFSAll(root):
	allNodes = []; 
	DFSAll(root,allNodes); 
	return allNodes; 

def DFSAll(root,allNodes):
	allNodes.append(root); 
	if(len(root.children) == 0): 
		return; 
	for child in root.children:
		DFSAll(child,allNodes); 

#grabs edge nodes in tree
def runDFSEx(root):
	exposed = []; 
	DFSEx(root,exposed);  
	return exposed

def DFSEx(root,exposed):
	if(len(root.children) == 0):
		exposed.append(root); 
		return; 
	for child in root.children:
		DFSEx(child,exposed); 


def makePoissonPlot():
	H = 10; 
	lambs = [(H-i)/(H/1.68) for i in range(0,H+1)]
	k = [i for i in range(0,6)]; 

	pmf = np.zeros(shape=(len(lambs),len(k)));  
	for i in range(0,len(lambs)):
		for j in range(0,len(k)):
			pmf[i][j] = (lambs[i]**k[j])*np.exp(-lambs[i])/np.math.factorial(k[j]); 
		plt.plot(k,pmf[i]); 
		#print(sum([pmf[i][l]*k[l] for l in range(0,len(k))])); 
	plt.show();

def make2SplitTree(depth = 7):
	trunkHeight = depth; 
	root = Node(None,[0,0]);
	node1 = root.addChild([0,trunkHeight]);  
	layerLengths = [5,4,3,2,1]; 
	layerLengths = [np.sqrt(trunkHeight)-np.sqrt(i) for i in range(0,trunkHeight-1)]; 
	layerNodes = []; 

	fig = plt.figure(); 
	ax = fig.add_subplot(111); 


	layerNodes.append([node1]); 
	for i in range(0,len(layerLengths)): 
		thisLayer = []; 
		for nod in layerNodes[i]:
			cordx = nod.pose[0] + layerLengths[i]*cos(radians(45 + nod.angle)); 
			cordy = nod.pose[1] + layerLengths[i]*sin(radians(45 + nod.angle)); 
			tmp = nod.addChild([cordx,cordy]); 
			thisLayer.append(tmp); 

			cordx = nod.pose[0] + layerLengths[i]*cos(radians(-45 + nod.angle)); 
			cordy = nod.pose[1] + layerLengths[i]*sin(radians(-45 + nod.angle)); 
			tmp = nod.addChild([cordx,cordy]); 
			thisLayer.append(tmp); 
		layerNodes.append(thisLayer); 


	plotTree(root,ax,False,0); 

def makeProbSplitTree(depth = 6,steps = 10):
	warnings.filterwarnings("ignore")
	trunkHeight = depth; 
	root = Node(None,[0,0]);
	node1 = root.addChild([0,trunkHeight/2]);  
	node1.depth = 0;  
	layerLengths = [np.sqrt(depth+2)-np.sqrt(i) for i in range(0,depth+2)]; 

	k = [i for i in range(0,depth)]; 

	# Tuning Nobs
	#dispersalTuner = 5; #rate of damping at distance
	#variance = 1.2; #rate of growth
	dispersalTuner = 3; 
	variance = .5;
	degSplit = 33;#tree dispersal
	lev = True; #plot leaves

	probs = np.zeros(shape=(depth+1,len(k)));  
	for i in range(0,depth+1):
		suma = 0; 
		for j in range(0,len(k)):
			probs[i][j] = norm((depth-i)/dispersalTuner,variance).pdf(k[j]); 
			suma+=probs[i][j]; 
		for j in range(0,len(k)):
			probs[i][j] = probs[i][j]/suma; 

	#print(probs);

	fig = plt.figure(); 
	ax = fig.add_subplot(111); 

	 
	degBounds = [-degSplit,degSplit]; 
	sign = lambda x:copysign(1,x); 

	for i in range(0,steps): 
		print(i); 
		thisLayer = runDFSEx(root); 
		for nod in thisLayer:
			if(nod.depth >= depth):
				nod.prune(); 
				continue
			#print(len(probs[nod.depth])); 
			branches = np.random.choice([j for j in range(0,depth)],p=probs[nod.depth]);
			#print(branches)
			if(branches == 0 and not nod.depth == 0): 
				nod.prune(); 
			if(branches == 1):
				continue; 
			elif(branches == 0 and nod.depth == 0):
				thisLayer.append(nod);  
			else:
				splits = [degBounds[0]+x*((degBounds[1]-degBounds[0])/(branches-1)) for x in range(0,branches)];  
				for j in range(0,branches):
					cordx = nod.pose[0] + layerLengths[nod.depth+1]*cos(radians(splits[j] + nod.angle))# + sign(90-splits[j]+nod.angle)*15)); 
					cordy = nod.pose[1] + layerLengths[nod.depth+1]*sin(radians(splits[j] + nod.angle))# + sign(90-splits[j]+nod.angle)*15)); 
					tmp = nod.addChild([cordx,cordy]); 
					tmp.depth = nod.depth+1; 
					 

		#print(i); 
		plotTree(root,ax,lev,0.01); 

	plotTree(root,ax); 

if __name__ == '__main__':
	#make2SplitTree(8); 
	makeProbSplitTree(10,100); 

