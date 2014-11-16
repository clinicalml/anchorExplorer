import sys
import networkx as nx
import cPickle as pickle
import re

class Structure(object):
    def __init__(self, structure, root):
        self.structure = structure
        self.root = root
        
    def getStructure(self):
        nodes = []
        for n in nx.dfs_preorder_nodes(self.structure, self.root):
            p = self.structure.in_edges(n)
            if len(p) == 0:
                p = ""
            else:
                p = p[0][0]
            nodes.append((n, self.structure.node[n]['name'], p))
        return nodes

    def getDescendents(self, root):
        nodes = []
        edges = []
        names = []
        for n in nx.dfs_preorder_nodes(self.structure, root):
            p = self.structure.in_edges(n)
            if len(p) == 0:
                p = ""
            else:
                p = p[0][0]

            nodes.append(str(n))
            names.append(self.structure.node[n]['name'])
            if n == root:
                continue

            edges.append((p,str(n)))
        return nodes, names, edges
        

