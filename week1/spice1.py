import sys
nonComment = lambda x: x.split('#',maxsplit=1)[0].strip()

try:
    netlistFile = sys.argv[1]  #take input from command line
except IndexError:
    print('please enter the netlist filename after .py filename.')
    quit()
else:
    if netlistFile[-8:] !=  '.netlist':
        print('please enter netlist filename')
        quit()

circuitFound, elements=False, []
try:
    with open(netlistFile,'r') as dataFile:
        for line in dataFile.read().split('\n'):
            if nonComment(line)=='.circuit':
                circuitFound = True
                continue
            elif nonComment(line)=='.end':
                break
            if circuitFound:
                elements.append(nonComment(line).split()) 

        for i in reversed(elements):
            print(' '.join(reversed(i)))
        #if len(elements)==0:
        #   print('This file has no valid data')

except (FileNotFoundError,IOError):
    print('the file you typed does not exist.\nTry again with a valid file')
