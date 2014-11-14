import networkx as nx
import re
import itertools
import cPickle as pickle
from Structures import Structure

#TODO: use settings to direct this reference
vocab,inv_vocab,display_vocab = pickle.load(file('vocab.pk'))

def subsets(l):
    sets = []
    for k in xrange(1,len(l)+1):
        for s in itertools.combinations(l, k):
            sets.append(s)
    return sets

class Anchor(object):
    def __init__(self, id, members=None, edges=None, display_names=None):
        self.id = id

        try:
            t = id.split('_')[0]
            members, display_names, edges = structures[t].getDescendents(id)

        except:
            pass


        if re.search('age_\d+-\d+', id):
            start = int(id.split('_')[1].split('-')[0])
            end = int(id.split('-')[1])
            members = [id] + ['age_'+str(i) for i in xrange(start, end)]
            edges = [(id, m) for m in members[1:]]

        if members == None:
            members = [id]

        if edges == None:
            edges = []

        if display_names == None:
            display_names = list(members)

        self.structure = nx.DiGraph()
        for m,name in zip(members, display_names):
            self.structure.add_node(m)
            self.structure.node[m]['name']=name

        for e in edges:
            self.structure.add_edge(*e)

        print 'new anchor', id, members, edges
        print 'members are', self.getMembers()

        e = set()
        for n in self.structure.nodes():
            n_set = set(n.split())
            for v in vocab:
                if len(set(v.split()) & n_set):
                    e.add(v)

        self.exclusions = e
        print 'exclusions', self.exclusions

    def getMembers(self):
        return self.structure.nodes()

    def getExclusions(self):
        return self.exclusions

    def getStructure(self):
        nodes = []
        for n in nx.dfs_preorder_nodes(self.structure, self.id):
            p = self.structure.in_edges(n)
            if len(p) == 0:
                p = ""
            else:
                p = p[0][0]
            name = self.structure.node[n]['name']
            nodes.append((n,name,p))
        return nodes
