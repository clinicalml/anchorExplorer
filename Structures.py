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
            #print nodes[-1]
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
        
if __name__ == "__main__":
#########
#ICD9 structure
    import Structures

    graph = nx.DiGraph()
    paren = re.compile('\([EV0-9\.\-]+\)')
    f = file('../icd9Tree.txt')
    current_parent = None
    graph.add_node('root', name='root')
    for l in f:
        if not '\t' in l:
            try:
                highest = l.strip()
                if highest=="":
                    continue
                if len(highest.split('.')[0]) == 1:
                    highest = '0'+highest
            except:
                print l, 'error'
                sys.exit()
            graph.add_edge('root', highest)
            graph.add_node(highest, name=highest)

        try:
            pat = paren.search(l).group(0)
            l = l.replace(pat, '').strip()
            pat = pat.strip('()')
            pat = 'code_'+pat

            if '.' in pat:
                continue
            
            if '-' in pat:
                current_parent = pat
                graph.add_node(current_parent, name=l)
                graph.add_edge(highest, current_parent)

            else:
                graph.add_node(pat, name=l)
                graph.add_edge(current_parent, pat)

        except:
            pass


    f.close()
    f = file('/m/sepsis_data/incoming/icd9.tsv')
    for l in f:
        code, name = l.strip().split('\t')
        code = 'code_'+code
        if not '.' in code:
            continue
        graph.add_node(code, name=name) 
        parent = code.split('.')[0]
        graph.add_edge(parent, code)

    codeStruct = Structures.Structure(graph, 'root')
    print codeStruct.getStructure()
    pickle.dump(codeStruct, file('codeStruct.pk', 'w'))

#Med structure
    lookup = pickle.load(file('../medDict.pk'))
    graph = nx.DiGraph()
    f = file('/m/sepsis_data/incoming/etc.tsv')
    f.readline()

    graph.add_node('med_etc_00000000', name='root')
    for l in f:
        id, name, parentid = l.strip().split('\t')
        for p in parentid.split():
            graph.add_edge('med_etc_'+p, 'med_etc_'+id)
            graph.node['med_etc_'+id]['name'] = name

    f.close()

    f = file('/m/sepsis_data/incoming/gsn.tsv')
    f.readline()

    for l in f:
        id, parentid = l.strip().split('\t')
        for p in parentid.split():
            graph.add_edge('med_etc_'+p, 'med_'+id)
            try:
                graph.node['med_'+id]['name'] = lookup['med_'+id]
            except:
                graph.node['med_'+id]['name'] = 'med_'+id

    medStruct = Structures.Structure(graph, 'med_etc_00000000')
    pickle.dump(medStruct, file('medStruct.pk', 'w'))


#pyx structure
    lookup = pickle.load(file('../pyxDict.pk'))
    graph = nx.DiGraph()
    f = file('/m/sepsis_data/incoming/etc.tsv')
    f.readline()

    graph.add_node('pyx_etc_00000000', name='root')
    for l in f:
        id, name, parentid = l.strip().split('\t')
        for p in parentid.split():
            graph.add_edge('pyx_etc_'+p, 'pyx_etc_'+id)
        graph.node['pyx_etc_'+id]['name'] = name

    f.close()

    f = file('/m/sepsis_data/incoming/gsn.tsv')
    f.readline()

    for l in f:
        id, parentid = l.strip().split('\t')
        for p in parentid.split():
            graph.add_edge('pyx_etc_'+p, 'pyx_'+id)
        try:
            graph.node['pyx_'+id]['name'] = lookup['pyx_'+id]
        except:
            graph.node['pyx_'+id]['name'] = 'pyx_'+id

    medStruct = Structures.Structure(graph, 'pyx_etc_00000000')
    pickle.dump(medStruct, file('pyxStruct.pk', 'w'))
