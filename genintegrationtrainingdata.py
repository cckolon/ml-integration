# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:18:12 2021

Source for symbolic integration w/ neural networks: https://arxiv.org/pdf/1912.01412.pdf
Source for Seq2Seq how to: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
Source for trees in python: https://medium.com/swlh/making-data-trees-in-python-3a3ceb050cfd
Source for RPN comprehension: https://blog.klipse.tech/python/2016/09/22/python-reverse-polish-evaluator.html
Source for jit: http://numba.pydata.org/

@author: Carl Kolon
"""
import numpy as np
import sympy as sp
import itertools as itt
from numba import jit
import multiprocessing as mp
import json
import time
import os
from functools import lru_cache

fwdtimes = 1000
bwdtimes = 1000
ibptimes = 1000

deecache = {}

unaryoperators = ["log","exp","sin","cos","sqrt","asin","acos","atan",'li','Ei','exp_polar','Ci','Si']
binaryoperators = ['+','-','*','/','**']
variables = ['x','1','2','3']

unaryprobs = [.2,.2,.1,.1,.1,.1,.1,.1,0,0,0,0,0]
binaryprobs = [.2,.2,.2,.2,.2]
varprobs = [.4,.3,.2,.1]

problemvars = [sp.core.numbers.NaN,sp.core.numbers.ComplexInfinity,sp.core.numbers.Infinity]
problemstrs = ['nan','zoo','oo']
numCPUs = os.cpu_count()
if numCPUs == None:
    numCPUs = 4


spfns = {
	"<class 'sympy.core.power.Pow'>" : '**',
	"<class 'sympy.core.mul.Mul'>" : '*',
	"<class 'sympy.core.add.Add'>" : '+'
}

class Functree:
	def __init__(self):
		self.nodes = []
		self.leaves = [Node()]
	def addnode(self,k,a): # turn the kth leaf into a new node with arity a
		for i in range(a): #add one child if a=1, add two if a=2
			newnode = Node()
			self.leaves.append(newnode)
			self.leaves[k].addchild(newnode)
		if a == 1: #set a unary operator for a=1
			self.leaves[k].setcontent(np.random.default_rng().choice(unaryoperators,p=unaryprobs))
		else: #set a binary operator for a=2
			self.leaves[k].setcontent(np.random.default_rng().choice(binaryoperators,p=binaryprobs))
		self.nodes.append(self.leaves.pop(k)) # remove the kth item from leaves and put it in nodes
	def fillleaves(self):
		for i in self.leaves:
			i.setcontent(np.random.default_rng().choice(variables,p=varprobs))			
	def display(self):
		disparray = self.nodes[0].display()
		for i in disparray:
			print(i)
	def reversepolish(self):
		return self.nodes[0].reversepolish()
	def spoutput(self):
		return sp.sympify(self.infix())

class Node:
	def __init__(self):
		self.children = []
		self.content = ""
	def addchild(self,child):
		self.children.append(child)
	def setcontent(self,cont):
		self.content = cont
	def display(self):
		if self.children == []:
			return self.content
		else:
			dispcont = [self.content]
			for i in self.children:
				dispcont.extend(["\t" + i.display()[j] for j in range(len(i.display()))])
			return(dispcont)
	def reversepolish(self):
		if self.children == []:
			return self.content
		else:
			dispcont = []
			for i in self.children:
				childcont = i.reversepolish()
				dispcont.extend(childcont)
			dispcont.append(self.content)
			return dispcont
		
#this function returns the number of different binary subtrees that can be generated from e empty elements, with n internal nodes to generate
# e is the length of the leaves array
# k is the index of the array to choose from
def deecached(e,n,deehash):
    if (e,n) in deehash:
        return deehash[(e,n)]
    else:
        newdee = dee(e,n)
        deehash[(e,n)] = newdee
        return newdee

@jit
def dee(e,n):
	if (e <= 0):
		return 0
	elif (n == 0):
		return 1
	elif (n < 0):
		return 0
	else:
		return dee(e-1,n) + dee(e,n-1) + dee(e+1,n-1)

#this function returns the probability that the next internal node is in position k with arity a
def ell(e,n,k,a,deehash):
	if (a==1):
		return deecached(e-k,n-1,deehash)/deecached(e,n,deehash)
	elif (a==2):
		return deecached(e-k+1,n-1,deehash)/deecached(e,n,deehash)
	else:
		return 0

def randomtree(numnodes):
	n=numnodes
	tree = Functree()
	while n > 0:
		kalist = list(itt.product(range(len(tree.leaves)),[1,2]))
		kaprobs = [ell(len(tree.leaves),n,*i,deecache) for i in kalist]
		(k,a) = np.random.default_rng().choice(kalist,p=kaprobs)
		tree.addnode(k,a)
		n -= 1
	tree.fillleaves()
	return tree

def rpntoinfix(expr):
	stack = []
	for i in expr:
		if i in unaryoperators:
			stack.append(i + "(" + stack.pop() + ")")
		elif i in binaryoperators:
			stack.append("(" + stack.pop(-2) + i + stack.pop() + ")")
		else:
			stack.append(i)
	return stack.pop()

def spfn_convert(expr):
	try:
		return spfns[str(expr)]
	except:
		return str(expr)

def sptorpn(expr):
	if expr.args == ():
		if type(expr) == sp.core.numbers.Rational: # split rational numbers into numerator and denominator
			return [str(expr.p),str(expr.q),"/"]
		elif type(expr) == sp.core.numbers.Half:
			return ['1','2','/']
		else:
			return [str(expr)]
	elif len(expr.args) == 1:
		return [*sptorpn(expr.args[0]),spfn_convert(expr.func)]
	elif len(expr.args) == 2:
		return [*sptorpn(expr.args[0]),*sptorpn(expr.args[1]),spfn_convert(expr.func)]
	else:
		return [*sptorpn(expr.args[0]),*sptorpn(expr.func(*expr.args[1:])),spfn_convert(expr.func)]
		
def bwdgenerate(numnodes,queue):
	integral = sp.core.numbers.Zero()
	while integral.is_constant():
		tree = randomtree(numnodes)
		integral = sp.sympify(rpntoinfix(tree.reversepolish()))
	integral = sp.expand(integral)
	if integral.func == sp.core.add.Add:
		for i in integral.args:
			if i.is_constant():
				integral = integral - i;
			elif type(i) in problemvars: # return zeros if the integral contains a problematic variable
				queue.put([['0'],['0']])
				return[['0'],['0']]
	integrand = sp.expand(sp.diff(integral,sp.symbols('x')))
	print('integral of: ')
	print(integrand)
	print('is')
	print(integral)
	integralseq = sptorpn(integral)
	integrandseq = sptorpn(integrand)
	if (sp.sympify(rpntoinfix(integralseq)) == integral) & (sp.sympify(rpntoinfix(integrandseq))==integrand):
		print("Self test passed.")
		queue.put([integrandseq,integralseq])
		return[integrandseq,integralseq]
	else:
		print("Self test failed.")
		print(str(sp.sympify(rpntoinfix(integralseq))) + " != " + str(integral))
		print("or")
		print(str(sp.sympify(rpntoinfix(integrandseq))) + " != " + str(integrand))
		print([integrandseq,integralseq])
		queue.put([['0'],['0']])
		return[['0'],['0']]

def fwdgenerate(numnodes,queue):
	tree = randomtree(numnodes)
	integrand = sp.expand(sp.sympify(rpntoinfix(tree.reversepolish())))
	print('Attempting to integrate:')
	print(integrand)
	try:
		integral = sp.integrate(integrand,sp.symbols('x'))
	except:
		print("Integration failed.")
		queue.put([['0'],['0']])
		return [['0'],['0']]
	if integral.has(sp.integrals.Integral):
		print("I could not find a closed-form integral.")
		queue.put([['0'],['0']])
		return [['0'],['0']]
	try:
		simplifiedintegral = sp.expand(integral)
		integral = simplifiedintegral
	except:
		pass
	print('Integration successful. Integral is:')
	print(integral)
	try:
		integralseq = sptorpn(integral)
		integrandseq = sptorpn(integrand)
	except:
		print('Simplification failed. Function arguments may be exotic.')
		queue.put([['0'],['0']])
		return [['0'],['0']]
	try:
		selftest = (sp.sympify(rpntoinfix(integralseq)) == integral) & (sp.sympify(rpntoinfix(integrandseq))==integrand)
	except:
		selftest = False
		print("Failed to conduct self test for the following function and integral:")
		print("Function: " + str(integrand))
		print("Integral: " + str(integral))
	if selftest:
		print("Self test passed.")
		queue.put([integrandseq,integralseq])
		return[integrandseq,integralseq]
	else:
		print("Self test failed.")
		print(str(rpntoinfix(integralseq)) + " != " + str(integral))
		print("or")
		print(str(rpntoinfix(integrandseq)) + " != " + str(integrand))
		print([integrandseq,integralseq])
		queue.put([['0'],['0']])
		return[['0'],['0']]

def ibpselftest(integrandseq,integralseq):
	return sp.diff(sp.sympify(rpntoinfix(integralseq))) == sp.sympify(rpntoinfix(integrandseq))
	
def ibpgenerate(numnodes,queue,pairs,integrandset):
	f = sp.core.numbers.Zero()
	while f.is_constant():
		ftree = randomtree(numnodes)
		f = sp.sympify(rpntoinfix(ftree.reversepolish()))
	g = sp.core.numbers.Zero()
	while g.is_constant():
		gtree = randomtree(numnodes)
		g = sp.sympify(rpntoinfix(gtree.reversepolish()))
	df = sp.expand(sp.diff(f,sp.symbols('x')))
	dg = sp.expand(sp.diff(g,sp.symbols('x')))
	fdgseq = sptorpn(sp.expand(f*dg))
	gdfseq = sptorpn(sp.expand(g*df))
	print(f"Searching for ({g})*({df}) or ({f})*({dg}) in {len(pairs)} training pairs.")
	if tuple(fdgseq) in integrandset:
		print(f"I found {f}*{dg} in the training set.")
		print()
		for pair in pairs:
			if pair[0] == fdgseq:
				integrandseq = gdfseq
				integral = sp.expand(g*f-sp.sympify(rpntoinfix(pair[1])))
				integralseq = sptorpn(integral)
				if ibpselftest(integrandseq,integralseq):
					print('Self test passed.')
					queue.put([integrandseq,integralseq])
					return [integrandseq,integralseq]
				else:
					print('Self test failed.')
					print(rpntoinfix(integralseq) + ' does not differentiate to ' + rpntoinfix(integrandseq))
					return [['0'],['0']]
	elif tuple(gdfseq) in integrandset:
		print(f"I found {g}*{df} in the training set.")
		for pair in pairs:
			if pair[0] == gdfseq:
				integrandseq = fdgseq
				integral = sp.expand(g*f-sp.sympify(rpntoinfix(pair[1])))
				integralseq = sptorpn(integral)
				if ibpselftest(integrandseq,integralseq):
					queue.put([integrandseq,integralseq])
					return [integrandseq,integralseq]
				else:
					print('Self test failed.')
					print(rpntoinfix(integralseq) + ' does not differentiate to ' + rpntoinfix(integrandseq))
					return [['0'],['0']]
	else:
		return [['0'],['0']]

def savetrainingdata(pairs,filepath):
	originallength = len(pairs)
	print(f'Saving {originallength} pairs to: {filepath}')
	with open(filepath,'a') as file:
		pass #Create the file if it doesn't exist
	with open(filepath,'r') as file:
		for line in file.readlines():
			i=0
			while i < len(pairs):
				if pairs[i][0] == json.loads(line)[0]:
					pairs.pop(i)
				else:
					i += 1
		print(f'I found and deleted {originallength - len(pairs)} duplicates.')
	with open(filepath,'a') as file:
		while pairs:
			writepair = True
			pair = pairs.pop()
			for i in pair:
				for j in i:
					if j in problemstrs:
						writepair = False
			if writepair:
				file.write(json.dumps(pair))
				file.write("\n")
			else:
				print(f'I am deleting {pair} due to exotic variables.')

def asHours(s):
	m = np.floor(s / 60)
	s -= m * 60
	h = np.floor(m / 60)
	m -= h * 60
	return '%dh %dm %ds' % (h, m, s)


def timeSince(since, percent):
	now = time.time()
	s = now - since
	es = s / (percent)
	rs = es - s
	return 'Elapsed: %s. Remaining: %s.' % (asHours(s), asHours(rs))

if __name__ == '__main__':
	fwdcompleted = 0
	bwdcompleted = 0
	ibpcompleted = 0
	numnodes = 8
	savepath = 'trainingdata.txt'
	outpath = None
	timeout = 10
	pairs = []
	starttime = time.time()
	with mp.Manager() as manager:
		q = manager.Queue()
		try:
			fwdstart = time.time()
			while fwdcompleted < fwdtimes:
				processes = min(numCPUs,fwdtimes-fwdcompleted)
				p = [mp.Process(target = fwdgenerate, args = (numnodes,q)) for i in range(processes)]
				for proc in p:
					proc.start()
				for i in range(timeout):
					time.sleep(1)
					if not any(proc.is_alive() for proc in p):
						break
				for i in range(len(p)):
					if p[i].is_alive():
						p[i].terminate()
						print(f"Timeout on process {i}. Terminating.")
				fwdcompleted += processes
				while not q.empty():
					pair = q.get(timeout = 3)
					print(rpntoinfix(pair[0]) + '   --->   ' + rpntoinfix(pair[1]))
					if pair not in pairs:
						pairs.append(pair)
				print(f"Forward functions completed: {fwdcompleted}/{fwdtimes}")
				print(f"Total functions completed: {bwdcompleted+fwdcompleted}/{bwdtimes+fwdtimes+ibptimes}")
				print(f"Average seconds per fwd function call: {(time.time()-fwdstart)/(fwdcompleted)}")
				print(timeSince(fwdstart, fwdcompleted/fwdtimes))
				if len(pairs)>100:
					savetrainingdata(pairs,savepath)
					print('Save complete. Resuming...')
			if fwdcompleted > 0:
				fwdavgtime = (time.time()-fwdstart)/(fwdcompleted)
			else:
				fwdavgtime = 'Not calculated'
			bwdstart = time.time()
			while bwdcompleted < bwdtimes:
				processes = min(numCPUs,bwdtimes-bwdcompleted)
				p = [mp.Process(target = bwdgenerate, args = (numnodes,q)) for i in range(processes)]
				for proc in p:
					proc.start()
				for i in range(timeout):
					time.sleep(1)
					if not any(proc.is_alive() for proc in p):
						break
				for i in range(len(p)):
					if p[i].is_alive():
						p[i].terminate()
						print(f"Timeout on process {i}. Terminating.")
				bwdcompleted += processes
				while not q.empty():
					pair = q.get(timeout = 3)
					print(rpntoinfix(pair[0]) + '   --->   ' + rpntoinfix(pair[1]))
					if pair not in pairs:
						pairs.append(pair)
				print(f"Backwards functions completed: {bwdcompleted}/{bwdtimes}")
				print(f"Total functions completed: {bwdcompleted+fwdcompleted}/{bwdtimes+fwdtimes+ibptimes}")
				print(f"Average seconds per bwd function call: {(time.time()-bwdstart)/(bwdcompleted+fwdcompleted+ibpcompleted)}")
				print(timeSince(bwdstart, bwdcompleted/bwdtimes))
				if len(pairs)>100:
					savetrainingdata(pairs,savepath)
					print('Save complete. Resuming...')
			if bwdcompleted > 0:
				bwdavgtime = (time.time()-bwdstart)/(bwdcompleted)
			else:
				bwdavgtime = 'Not calculated'
		finally:
			savetrainingdata(pairs,savepath)
		pairs = []
		newpairs = []
		integrandset = set([])
		ibpstart = time.time()
		with open(savepath, 'r') as file:
			for line in file.readlines():
				pairs.append(json.loads(line))
				integrandset.add(tuple(json.loads(line)[0]))
		print(f"I imported {len(pairs)} pairs.")
		try:
			while ibpcompleted < ibptimes:
				processes = min(numCPUs,ibptimes-ibpcompleted)
				p = [mp.Process(target = ibpgenerate, args = (numnodes,q,pairs,integrandset)) for i in range(processes)]
				for proc in p:
					proc.start()
				for i in range(timeout):
					time.sleep(1)
					if not any(proc.is_alive() for proc in p):
						break
				for i in range(len(p)):
					if p[i].is_alive():
						p[i].terminate()
						print(f"Timeout on process {i}. Terminating.")
				ibpcompleted += processes
				while not q.empty():
					pair = q.get(timeout = 3)
					print(rpntoinfix(pair[0]) + '   --->   ' + rpntoinfix(pair[1]))
					if pair not in newpairs:
						newpairs.append(pair)
				print(f"IBP functions completed: {ibpcompleted}/{ibptimes}")
				print(f"Total functions completed: {ibpcompleted+bwdcompleted+fwdcompleted}/{bwdtimes+fwdtimes+ibptimes}")
				print(f"Average seconds per IBP function call: {(time.time()-ibpstart)/(ibpcompleted)}")
				print(timeSince(ibpstart, ibpcompleted/ibptimes))
			if ibpcompleted > 0:
				ibpavgtime = (time.time()-ibpstart)/(ibpcompleted)
			else:
				ibpavgtime = 'Not calculated'
		finally:
			savetrainingdata(newpairs,savepath)
			print('Program complete.')
			print(f'Average forward function time: {fwdavgtime}')
			print(f'Average backward function time: {bwdavgtime}')
			print(f'Average IBP function time: {ibpavgtime}')
