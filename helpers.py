import re

def union(L):
    s = set()
    for l in L:
        s |= set(l)
    return s

phi_re = re.compile("PHI_PHI_PHI\S*PHI_PHI_PHI")

def removePHI(s):
    return phi_re.sub("", s)

def show_hand_cursor(event): 
    event.widget.configure(cursor="hand1") 

def show_arrow_cursor(event):
    event.widget.configure(cursor="")


def nop():
    pass
