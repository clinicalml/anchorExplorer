import sys
import os
import networkx as nx
import cPickle as pickle
import re
from Structures import Structure

        
if __name__ == "__main__":
    
    try:
        datatype = sys.argv[1]
        names = sys.argv[2]+'.names'
        edges = sys.argv[2]+'.edges'
    except:
        print "usage: python build_structured_rep.py type src"
        sys.exit()
    

    try:
        os.makedirs('Structures')
    except:
        pass

    print 'building a structured representation of', datatype
    print 'assuming prefix', datatype+'_'
    prefix = datatype+'_'

    
    nameDict = {}
    f = file(names)
    for l in f:
        code, name = l.strip().split('\t')
        code = prefix+code
        nameDict[code] = name
    f.close()

    f = file(edges)
    graph = nx.DiGraph()

    for code,name in nameDict.items():
        graph.add_node(code, name=name)

    for l in f:
        parent,child = l.strip().split('\t')
        graph.add_edge(prefix+parent, prefix+child)
    f.close()

    struct = Structure(graph, prefix+'ROOT')
    struct.getStructure()
    pickle.dump(struct, file('Structures/'+datatype+'Struct.pk', 'w'))
    pickle.dump(nameDict, file('Structures/'+datatype+'Dict.pk', 'w'))


