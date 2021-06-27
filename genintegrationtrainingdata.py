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



unaryoperators = ["log","exp","sin","cos","sqrt","asin","acos","atan",'li','Ei','exp_polar']
binaryoperators = ['+','-','*','/','**']
variables = ['x','1','2','3']

unaryprobs = [.2,.2,.1,.1,.1,.1,.1,.1,0,0,0]
binaryprobs = [.2,.2,.2,.2,.2]
varprobs = [.4,.3,.2,.1]

problemvars = [sp.core.numbers.NaN,sp.core.numbers.ComplexInfinity,sp.core.numbers.Infinity]


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
@jit
def ell(e,n,k,a):
	if (a==1):
		return dee(e-k,n-1)/dee(e,n)
	elif (a==2):
		return dee(e-k+1,n-1)/dee(e,n)
	else:
		return 0

def randomtree(numnodes):
	n=numnodes
	tree = Functree()
	while n > 0:
		kalist = list(itt.product(range(len(tree.leaves)),[1,2]))
		kaprobs = [ell(len(tree.leaves),n,*i) for i in kalist]
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
	integral = sp.simplify(integral)
	if integral.func == sp.core.add.Add:
		for i in integral.args:
			if i.is_constant():
				integral = integral - i;
			elif type(i) in problemvars: # return zeros if the integral contains a problematic variable
				queue.put([['0'],['0']])
				return[['0'],['0']]
	integrand = sp.diff(integral,sp.symbols('x'))
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
	integrand = sp.sympify(rpntoinfix(tree.reversepolish()))
	integrand = sp.simplify(integrand)
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
		simplifiedintegral = sp.simplify(integral)
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

def loadtrainingdata(filepath):
	pairs = []
	with open(filepath) as file:
		for line in file.readlines():
			pairs.append(json.loads(line))
	return pairs

def savetrainingdata(pairs,filepath):
	with open(filepath,'w') as file:
		for i in pairs:
			file.write(json.dumps(i))
			file.write("\n")

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
	fwdtimes = 7200
	bwdtimes = 7200
	numnodes = 3
	inpath = 'trainingdata.txt'
	outpath = None
	timeout = 10
	pairs = loadtrainingdata(inpath)
	starttime = time.time()
	with mp.Manager() as manager:
		q=mp.Queue()
		try:
			for i in range(max(fwdtimes,bwdtimes)):
				if i < fwdtimes:
					try:
						p=mp.Process(target=fwdgenerate, args = (numnodes,q))
						p.start()
						p.join(timeout)
						if p.is_alive():
							print("Timeout")
							pair = [['0'],['0']]
						else:
							pair = q.get()
						if pair in pairs:
							pass
						else:
							pairs.append(pair)
					finally:
						p.terminate()
						avgsecs = (time.time()-starttime)/(2*i+1)
						if i < bwdtimes:
							callscompleted = 2*i + 1
						else:
							callscompleted = bwdtimes + i + 1
						print(rpntoinfix(pair[0]) + '   --->   ' + rpntoinfix(pair[1]))
						print(f"Forward functions computed: {i+1}/{fwdtimes}")
						print(f"Total functions computed: {callscompleted}/{fwdtimes+bwdtimes}")
						print(f"Average seconds per function call: {avgsecs}")
						print(timeSince(starttime, callscompleted/(fwdtimes + bwdtimes)))
				if i < bwdtimes:
					try:
						p=mp.Process(target=bwdgenerate, args = (numnodes,q))
						p.start()
						p.join(timeout)
						if p.is_alive():
							print("Timeout")
							pair = [['0'],['0']]
						else:
							pair = q.get()
						if pair in pairs:
							pass
						else:
							pairs.append(pair)
					finally:
						p.terminate()
						avgsecs = (time.time()-starttime)/(2*i+1)
						if i < fwdtimes:
							callscompleted = 2*i + 2
						else:
							callscompleted = fwdtimes + i + 1
						print(rpntoinfix(pair[0]) + '   --->   ' + rpntoinfix(pair[1]))
						print(f"Backward functions computed: {i+1}/{bwdtimes}")
						print(f"Total functions computed: {callscompleted}/{fwdtimes+bwdtimes}")
						print(f"Average seconds per function call: {(time.time()-starttime)/(2*i+2)}")
						print(timeSince(starttime, callscompleted/(fwdtimes + bwdtimes)))
		finally:
			if outpath == None:
				savetrainingdata(pairs,inpath)
			else:
				savetrainingdata(pairs,outpath)


"""
fix this after we implement the training set

def ibpgenerate(numnodes):
	f = sp.core.numbers.Zero()
	while f.is_constant():
		ftree = randomtree(numnodes)
		f = sp.sympify(rpntoinfix(ftree.reversepolish()))
	g = sp.core.numbers.Zero()
	while g.is_constant():
		gtree = randomtree(numnodes)
		g = sp.sympify(rpntoinfix(ftree.reversepolish()))
	df = sp.diff(f,sp.symbols('x'))
	dg = sp.diff(g,sp.symbols('x'))
	print('integral of:')
	print(f*dg)
	print('is')
	print(f*g-)
"""

