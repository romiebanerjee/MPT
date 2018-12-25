import numpy as np
import sympy as sp 
import pandas as pd
import statistics as stats 
import json
import math


def flatten(A): #flattens a nested list    
        if A == []: return A
        if type(A[0]) == list:
            return flatten(A[0]) + flatten(A[1:])
        else: return [A[0]] + flatten(A[1:])
        

def cart(A,B): # cartesian product of two lists
   
    if A == []: return B
    elif B == []: return A
    else: return [flatten([x,y]) for x in A for y in B]

def mode(A): #Find most common element
    df = pd.DataFrame({"A":A})
    return df.mode().A.values.tolist()[0]



##.....Some linear agebra 


def rref(A): 
	X = sp.Matrix(A).rref()[0]
	return np.array(X).astype(np.float64)


def is_in_range(A,b):
    A = np.matrix(A)
    b = np.matrix(b)
    codomain_dim = np.shape(A)[0]
    domain_dim = np.shape(A)[1]
    if codomain_dim == np.shape(b)[1]: pass
    else: return None
          
    b_rref = rref(np.append(A,b.transpose(), axis=1))[:,-1]
    rank = np.linalg.matrix_rank(A)
      
    if rank == codomain_dim: return True
    elif np.array(b_rref[rank:]).tolist() == [0]*(codomain_dim - rank):
    	return True
    else: return False


##......Network class 

class Network():
	def __init__(self, vertices = (), edges = []):
		self.vertices = vertices
		e1 = [edge for edge in edges if vertices.index(edge[0]) <= vertices.index(edge[1])]
		e2 = [edge[::-1] for edge in edges if vertices.index(edge[0]) > vertices.index(edge[1])]
		self.edges = e1 + e2

	def adjacency_matrix(self):
		V = self.vertices
		n = len(self.vertices)
		A = np.zeros((n,n))
		for i in range(n):
			for j in range(n):
				if (V[i],V[j]) in self.edges or (V[j],V[i]) in self.edges:
					A[i,j] = 1
				else:
					A[i,j] = 0
		return A


	def remove_vertices(self, vertex_subset):
		V = [v for v in self.vertices if v not in vertex_subset]
		E = [e for e in self.edges if set(vertex_subset).intersection(set(e)) == set()]
		return Network(V,E)

	def delta_0(self):
		return np.zeros(len(self.vertices))

	def delta_1(self):
		m = len(self.vertices)
		n = len(self.edges)
		A = np.zeros((m,n))
		for i in range(m):
			for j in range(n):
				for k in [0,1]:
					if self.vertices[i] == self.edges[j][k]:
						A[i,j] = (-1)**k
					else: pass
		return np.matrix(A)


	def betti_0(self):
		if len(self.edges) == 0: return len(self.vertices)
		else:
			return len(self.vertices) - np.linalg.matrix_rank(self.delta_1())

	def betti_1(self):
		if len(self.edges) == 0: return 0
		else: 
			return len(self.edges) - np.linalg.matrix_rank(self.delta_1())

	def is_connected(self, vertex1, vertex2):
		if len(self.edges) == 0: return False
		b = [0]*len(self.vertices)
		V = self.vertices 
		path = [vertex1, vertex2]
		if V.index(vertex1) <= V.index(vertex2): pass
		else: path = path[::-1]

		b[V.index(path[0])] = 1
		b[V.index(path[1])] = -1

		return is_in_range(self.delta_1(), b)

	#def connected_component(self, vertex):
	#	if self.vertices == (): return ()
	#	connected_component = [vertex] + [v for v in self.vertices if self.is_connected(vertex,v)]
	#	return tuple(connected_component)

	def connected_component(self,vertex):
		if self.vertices == (): return ()
		n = len(self.vertices)
		index = self.vertices.index(vertex)
		A = self.adjacency_matrix()
		B = sum([np.linalg.matrix_power(A,i) for i in range(100)])
		connected_indices = [j for j in range(n) if B[index,j] != 0]
		return tuple([self.vertices[k] for k in connected_indices])

	def components(self):
		if len(self.vertices) == 0: return []
		v = self.vertices[0]
		component_v = self.connected_component(v)
		X = self.remove_vertices(component_v)
		return [component_v] + X.components()

	def draw(self):
		V = self.vertices
		E = self.edges

		nodes = [{"id": v, "label": v} for v in V]
		links = [{"source": V.index(link[0]), "target": V.index(link[1]), "value": 1} for link in E]

		viz = {"nodes":nodes, "links": links} #"paths": paths}

		viz_json = json.dumps(viz)
		file = open("network.json", 'w')
		file.write(viz_json)
		file.close()


		

## ........Point Cloud Class


class PointCloud():
	def __init__(self, points = [], metric = {}):
		self.points = points
		self.pairs = [(x,y) for x in self.points for y in self.points if self.points.index(x) < self.points.index(y)]
		self.metric = {e:metric[e] for e in self.pairs} 

	def dist_matrix(self):
		V = self.points
		n = len(self.points)
		A = np.zeros((n,n))
		for i in range(n):
			for j in range(n):
				if i<j:
					A[i,j] = self.metric[(V[i],V[j])]
					A[j,i] = self.metric[(V[i],V[j])]
				elif i == j:
					A[i,j] = 0
				else: pass
		return A

	def l2_inner(self):
		n = len(self.points)
		A = self.dist_matrix()
		M = np.zeros((n,n))
		for i in range(n):
			for j in range(n):
				M[i,j] = 0.5*(A[i,0]**2 + A[j,0]**2 - A[i,j]**2)
		return np.matrix(M)

	def l2_embed(self):
		n = len(self.points)
		M = self.l2_inner()
		eig, U = np.linalg.eigh(M)
		eig = np.absolute(eig)
		D = np.zeros((n,n))
		for i in range(n):
			D[i,i] = eig[i]
		S = np.sqrt(D)
		V = np.dot(U,S)
		return V
		

	def sub_pc(self, points_subset):
		points = points_subset
		metric = {e:self.metric[e] for e in list(self.metric.keys()) if e[0] in points_subset and e[1] in points_subset}
		return PointCloud(points, metric)


	def network(self, scale):
		V = self.points
		E = [e for e in self.pairs if self.metric[e] < scale]
		return Network(V,E)

	def PH_0(self, steps = 100):
		if self.points == []: return 0, 0
		if len(self.metric) == 0: return 1,0
		#n = min(list(self.metric.values()))
		n = 0
		m = max(list(self.metric.values()))
		l = []
		scales = np.arange(n,m,(m - n)/steps).tolist()
		for scale in scales: 
			l.append(self.network(scale).betti_0())
		ph0_scale = scales[l.index(mode(l))]
		return mode(l), ph0_scale

	def PH_1(self, steps = 100):
		#n = min(list(self.metric.values()))
		n = 0
		m = max(list(self.metric.values()))
		l = []
		scales = np.arange(n,m,(m-n)/steps).tolist()
		for scale in scales: 
			l.append(self.network(scale).betti_1())
		ph1_scale = scales[l.index(mode(l))]
		return mode(l), ph1_scale

	

	def ph0_network(self, steps = 10):
		return self.network(scale = self.PH_0(steps)[1])

	def ph0_components(self, steps = 10):
		return self.ph0_network(steps).components()

	def rbfkernel(self):
		A = np.exp(-np.square(self.dist_matrix()))
		values = [np.sum(A[i,:]) for i in range(len(self.points))]
		return values

	def rbf_covering(self, rcover):
		values = self.rbfkernel()
		
		a,b = np.amin(values), np.amax(values)
		N, overlap = rcover
		eps = ((b - a)/N)*overlap
		rcover = [a + (b - a)*i/N for i in range(N)] + [b]
		print(rcover, eps)
		covering = []
		for i in range(N):
			preimage = [x for x in self.points if values[self.points.index(x)]> rcover[i] - eps and values[self.points.index(x)] < rcover[i+1] + eps]
			print(preimage)
			preimage_subpc = self.sub_pc(preimage)
			print(preimage_subpc.PH_0())
			components = preimage_subpc.ph0_components()
			print(components)
			for component in components:
				covering.append(component)
		return covering, N

	def rbf_mapper(self, rcover = [5,0.3]):
		V = self.rbf_covering(rcover)[0]
		values = self.rbfkernel()

		pairs = [(x,y) for x in V for y in V if V.index(x) < V.index(y)]
		#print([len(v) for v in V])

		E = [(x,y) for (x,y) in pairs if set(x).intersection(set(y)) != set()]
		#return Network(V,E)

		max_value = max(values)
		max_weight = max([len(v) for v in V])
		nodes = [{"id": v, "label": v, "weight": len(v), "rbf_value": math.floor(np.average([values[self.points.index(x)] for x in v])) } for v in V]
		
		#print(nodes)
		
		links = [{"source": V.index(link[0]), "target": V.index(link[1]), "value": 1} for link in E]
		
		#print(links)

		viz = {"max_rbfvalue":math.floor(max_value), "max_weight": max_weight, "nodes":nodes, "links": links}

		viz_json = json.dumps(viz)
		file = open("network.json", 'w')
		file.write(viz_json)
		file.close()




	
		







		














