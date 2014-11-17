import sys
import string
import re
import csv
import os


MAX_ENTRIES = 2
def collectInfo(datadir, patient, structure):
    print "<visit>"
    print "\t<index>vid_"+patient+"</index>"
    for fname,field in structure.items():
        if fname == "__index__":
            continue
        print "\t<"+fname.split(':')[1]+">",

        input = file(datadir+'/'+patient+'/'+fname.split(':')[0]+'-'+patient+'.txt')
        reader = csv.reader(input)
        try:
            header = reader.next()
            index = header.index(field[0])
            for n,l in enumerate(reader):
                data = l[index].strip() +'\n'
                if len(field[1]):
                    print "\t\t<"+field[1]+">" +re.sub("\&", "", data) +"</"+field[1]+">"
                    
                else:
                    if n < MAX_ENTRIES:
                        print re.sub("\&", "", data)
        except:
            pass

        print "\t</"+fname.split(':')[1]+">"
    print "</visit>"



if __name__ == "__main__":
    
    try:
        src = sys.argv[1]
        field_file = sys.argv[2]
    except:
        print 'usage: get_mimic_data.py src fields'
        sys.exit()

    fields = file(field_file)

    structure = {}
    while 1:
        l = fields.readline()
        if l.strip() == '':
            break
        if l.split(':')[0] == 'index_by':
            index = l.split(':')[1]
            structure['__index__'] = index
        else:
            fname = l.strip()
            l = fields.readline()
            fieldnames = l.strip().split(':')
            structure[fname] = fieldnames

    for n,pat in enumerate(os.listdir(src)):
        collectInfo(src, pat, structure)
