import sys
import re
import csv
import os

_illegal_unichrs = [(0x00, 0x08), (0x0B, 0x0C), (0x0E, 0x1F), 
                        (0x7F, 0x84), (0x86, 0x9F), 
                        (0xFDD0, 0xFDDF), (0xFFFE, 0xFFFF)] 
if sys.maxunicode >= 0x10000:  # not narrow build 
        _illegal_unichrs.extend([(0x1FFFE, 0x1FFFF), (0x2FFFE, 0x2FFFF), 
                                 (0x3FFFE, 0x3FFFF), (0x4FFFE, 0x4FFFF), 
                                 (0x5FFFE, 0x5FFFF), (0x6FFFE, 0x6FFFF), 
                                 (0x7FFFE, 0x7FFFF), (0x8FFFE, 0x8FFFF), 
                                 (0x9FFFE, 0x9FFFF), (0xAFFFE, 0xAFFFF), 
                                 (0xBFFFE, 0xBFFFF), (0xCFFFE, 0xCFFFF), 
                                 (0xDFFFE, 0xDFFFF), (0xEFFFE, 0xEFFFF), 
                                 (0xFFFFE, 0xFFFFF), (0x10FFFE, 0x10FFFF)]) 

_illegal_ranges = ["%s-%s" % (unichr(low), unichr(high)) 
                   for (low, high) in _illegal_unichrs] 
#_illegal_xml_chars_RE = re.compile(u'[%s]' % u''.join(_illegal_ranges)+)
_illegal_xml_chars_RE = re.compile('&')

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
                data = ''
                data += l[index].strip() +'\n'
                if len(field[1]):
                    print "\t\t<"+field[1]+">" +re.sub("\&", "", data) +"</"+field[1]+">"
                    #print "\t\t<"+field[1]+">" +data +"</"+field[1]+">"
                else:
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

    #collectInfo('00', '00077', structure)
    for n,pat in enumerate(os.listdir(src)):
        collectInfo(src, pat, structure)
