#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 14:16:20 2018

@author: yuyingjie
"""

import logging
import sys
from os import path
from time import time
from glob import glob
from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict, Iterable
from multiprocessing import cpu_count
import random
import collections
from random import shuffle
from itertools import product,permutations
from scipy.io import loadmat
from scipy.sparse import issparse

from concurrent.futures import ProcessPoolExecutor

from multiprocessing import Pool
from multiprocessing import cpu_count



logger = logging.getLogger("deepwalk")


__author__ = "Yingjie Yu"
__email__ = "yyu2@scu.edu"

LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

'''
All ratings are contained in the file "ratings.dat" and are in the
following format:

UserID,ProfileID,Rating
'''
class Node(object):
    def __init__(self, id, name, type='user'):
        self.id = str(id)
        self.neighbors = []
        self.name = name
        self.type = type
        self.rating = {}

class User(object):
    def __init__(self, userid):
        self.userid= userid
        self.gender = None
        
def load_gender_data():
    gender_filename = './libimseti/gender.dat'
    users = {}
    with open(gender_filename,'r+') as f:
        f.__next__() # burn meta data line
        for line in f:
            u_id,gender = line.strip().split(',')
            users['u'+u_id] = User(u_id)
            users['u'+u_id].gender = gender
    return users
            
def records_to_graph():
    """
    Creates a graph from the datasets.
    A node is created for each entity: user, gender, profile, rating.
    The rating nodes created as one node for each possible 1-10 rating and for each profile(rated user).
        e.g., The profile 124 will lead to the nodes 124_1, 124_2,..., 124_10.
    The profile rating node 124_2, for example, will be connected to movie 124 and any users who rated 124 as a 2.
    """
    # Output files for the graph
    adjlist_file = open("./out.adj", 'w')
    node_list_file = open("./nodelist.txt", 'w')
    
    # Load all the ratings for every user into a dictionary
    # The dictionary maps a user to a list of (movie, rating) pairs
    #   e.g., ratings[75] = [(3,1), (32,4.5), ...]
    
    num_ratings = 0
    ratings = collections.defaultdict(dict)
    with open('./libimseti/ratings.dat','r') as f:
        f.next() # ignore metadata line
        for line in f:
            ls = line.strip().split('\t')
            user, profile, rating = ls[:3]
            rating = str(int(round(float(rating))))
            ratings['u'+user]['p'+profile] = rating
            num_ratings+=1
    users = load_gender_data()
    """
    Create nodes for the different entities in the graph
    Keep all the nodes that you make in nodelist.
    nodedict should map node IDs to their respective node object.
    The node IDs should be the ID of that node in the graph; the IDs need to range from 0 to n-1 incrementally.
      e.g., the node u75's ID may be 12 => nodedict["u75"].id = 12
     """ 
    nodelist=[]
    nodedict={}
     # adding profile to the nodedict
    for key,value in users.items():
        newNode = Node(getNextId(), value.name,'profile')
        nodedict[key] = newNode
        nodelist.append(newNode)
        for r in range(1,11):
            ratingNode = Node(getNextId(), key+'_' + str(r), 'rating')
            nodedict[key+'_'+str(r)] = ratingNode
            nodelist.append(ratingNode)
        
    # adding users to the nodedict
    for user in ratings:
        newUser = Node(getNextId(), user, 'user')
        nodedict[user] = newUser
        nodelist.append(newUser)
        
    # adding gender 
    for g in ['U','F','M']:
        newNode = Node(getNextId(), g, 'gender')
        nodedict[g] = newNode
        nodelist.append(newNode)
    # Add edges between users and profile-rating nodes
    # Add edges between user and gender
    # Add edges between profile ratings and profiles
    # By "add an edge" we mean to update the neighbors list of the nodes in both directions:
    #   e.g., 
    #           user_node.neighbors.append(gender_node)
    #           gender_node.neighbors.append(user_node)        
    
    # add edges beween user-rating/profile - profile rating
    for user, rating in ratings.items():
        for p,r in rating.items():
            userNode = nodedict[user]
            ratingNode = nodedict['p'+'_'+r]
            profileNode = nodedict[p]
            userNode.neighbors.append(ratingNode)
            ratingNode.neighbors.append(userNode)
            profileNode.neighbors.append(ratingNode)
            ratingNode.neighbors.append(profileNode)
            
    # user- gender
    for k, v in users.items():
        userNode = nodedict[k]
        if users[k].gender!=None:
            genderNode= nodedict[v.gender]
            userNode.neighbors.append(genderNode)
            genderNode.neighbors.append(userNode)
    
    # Write out the graph
    for node in nodelist:
        node_list_file.write("%s\t%s\t%s\n" % (node.id, node.name, node.type))
        adjlist_file.write("%s " % node.id)
        for n in node.neighbors:
            adjlist_file.write("%s " % n.id)
        adjlist_file.write("\n")
    adjlist_file.close()
    node_list_file.close()
    
    return nodedict

nId = -1
def getNextnId():
  global nId
  nId = nId + 1
  return nId   


class Graph(defaultdict):
  """Efficient basic implementation of nx `Graph' â€“ Undirected graphs with self loops"""  
  def __init__(self):
    super(Graph, self).__init__(list)

  def nodes(self):
    return self.keys()

  def adjacency_iter(self):
    return self.iteritems()

  def subgraph(self, nodes={}):
    subgraph = Graph()
    
    for n in nodes:
      if n in self:
        subgraph[n] = [x for x in self[n] if x in nodes]
        
    return subgraph

  def make_undirected(self):
  
    t0 = time()

    for v in self.keys():
      for other in self[v]:
        if v != other:
          self[other].append(v)
    
    t1 = time()
    logger.info('make_directed: added missing edges {}s'.format(t1-t0))

    self.make_consistent()
    return self

  def make_consistent(self):
    t0 = time()
    for k in iterkeys(self):
      self[k] = list(sorted(set(self[k])))
    
    t1 = time()
    logger.info('make_consistent: made consistent in {}s'.format(t1-t0))

    self.remove_self_loops()

    return self

  def remove_self_loops(self):

    removed = 0
    t0 = time()

    for x in self:
      if x in self[x]: 
        self[x].remove(x)
        removed += 1
    
    t1 = time()

    logger.info('remove_self_loops: removed {} loops in {}s'.format(removed, (t1-t0)))
    return self

  def check_self_loops(self):
    for x in self:
      for y in self[x]:
        if x == y:
          return True
    
    return False

  def has_edge(self, v1, v2):
    if v2 in self[v1] or v1 in self[v2]:
      return True
    return False

  def degree(self, nodes=None):
    if isinstance(nodes, Iterable):
      return {v:len(self[v]) for v in nodes}
    else:
      return len(self[nodes])

  def order(self):
    "Returns the number of nodes in the graph"
    return len(self)    

  def number_of_edges(self):
    "Returns the number of nodes in the graph"
    return sum([self.degree(x) for x in self.keys()])/2

  def number_of_nodes(self):
    "Returns the number of nodes in the graph"
    return order()

  def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
    """ Returns a truncated random walk.
        path_length: Length of the random walk.
        alpha: probability of restarts.
        start: the start node of the random walk.
    """
    G = self
    if start:
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(G.keys())]

    while len(path) < path_length:
      cur = path[-1]
      if len(G[cur]) > 0:
        if rand.random() >= alpha:
          path.append(rand.choice(G[cur]))
        else:
          path.append(path[0])
      else:
        break
    return path

def build_deepwalk_corpus(G, num_paths, path_length, alpha=0,
                      rand=random.Random(0)):
  walks = []

  nodes = list(G.nodes())
  
  for cnt in range(num_paths):
    rand.shuffle(nodes)
    for node in nodes:
      walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))
  
  return walks

def build_deepwalk_corpus_iter(G, num_paths, path_length, alpha=0,
                      rand=random.Random(0)):
  walks = []

  nodes = list(G.nodes())

  for cnt in range(num_paths):
    rand.shuffle(nodes)
    for node in nodes:
      yield G.random_walk(path_length, rand=rand, alpha=alpha, start=node)

def clique(size):
    return from_adjlist(permutations(range(1,size+1)))


def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

def parse_adjacencylist(f):
  adjlist = []
  for l in f:
    if l and l[0] != "#":
      introw = [int(x) for x in l.strip().split()]
      row = [introw[0]]
      row.extend(set(sorted(introw[1:])))
      adjlist.extend([row])
  
  return adjlist

def parse_adjacencylist_unchecked(f):
  adjlist = []
  for l in f:
    if l and l[0] != "#":
      adjlist.extend([[int(x) for x in l.strip().split()]])
  
  return adjlist

def load_adjacencylist(file_, undirected=False, chunksize=10000, unchecked=True):

  if unchecked:
    parse_func = parse_adjacencylist_unchecked
    convert_func = from_adjlist_unchecked
  else:
    parse_func = parse_adjacencylist
    convert_func = from_adjlist

  adjlist = []

  t0 = time()

  with open(file_) as f:
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
      total = 0 
      for idx, adj_chunk in enumerate(executor.map(parse_func, grouper(int(chunksize), f))):
          adjlist.extend(adj_chunk)
          total += len(adj_chunk)
  
  t1 = time()

  logger.info('Parsed {} edges with {} chunks in {}s'.format(total, idx, t1-t0))

  t0 = time()
  G = convert_func(adjlist)
  t1 = time()

  logger.info('Converted edges to graph in {}s'.format(t1-t0))

  if undirected:
    t0 = time()
    G = G.make_undirected()
    t1 = time()
    logger.info('Made graph undirected in {}s'.format(t1-t0))

  return G 


def load_edgelist(file_, undirected=True):
  G = Graph()
  with open(file_) as f:
    for l in f:
      x, y = l.strip().split()[:2]
      x = int(x)
      y = int(y)
      G[x].append(y)
      if undirected:
        G[y].append(x)
  
  G.make_consistent()
  return G


def load_matfile(file_, variable_name="network", undirected=True):
  mat_varables = loadmat(file_)
  mat_matrix = mat_varables[variable_name]

  return from_numpy(mat_matrix, undirected)


def from_networkx(G_input, undirected=True):
    G = Graph()

    for idx, x in enumerate(G_input.nodes_iter()):
        for y in iterkeys(G_input[x]):
            G[x].append(y)

    if undirected:
        G.make_undirected()

    return G


def from_numpy(x, undirected=True):
    G = Graph()

    if issparse(x):
        cx = x.tocoo()
        for i,j,v in zip(cx.row, cx.col, cx.data):
            G[i].append(j)
    else:
      raise Exception("Dense matrices not yet supported.")

    if undirected:
        G.make_undirected()

    G.make_consistent()
    return G


def from_adjlist(adjlist):
    G = Graph()
    
    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = list(sorted(set(neighbors)))

    return G


def from_adjlist_unchecked(adjlist):
    G = Graph()
    
    for row in adjlist:
        node = str(row[0])
        neighbors = map(str, row[1:])
        G[node] = neighbors

    return G

    
            
