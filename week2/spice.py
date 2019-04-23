import sys,cmath
import numpy as np

nonComment = lambda line:line.split('#')[0].split()
#myRound = lambda x: '{0.real:0.6f} {1} {0.imag:0.6f}j'.format(x,'-' if x.imag<0 else '+')
def myRound(x,ndigits=6):
    return '{0.real:0.{2}f} {1} {0.imag:0.{2}f}j'.format(x, '' if x.imag<0 else '+',ndigits)

def valuate(x):
    '''converts string notation of magnitude to complex'''
    if isinstance(x,str) and x[-1] in 'pnumkM':
        return complex(x[:-1])*10**dict(zip('pnumkM',(-12,-9,-6,-3,3,6)))[x[-1]]
    return eval(x)

def polarForm(x,polar=True,ndigits=6):
    #return (abs(x),cmath.phase(x)*180/cmath.pi) if polar else x
    if polar:
        return '{:0.6f}, {:0.6f}'.format(abs(x),cmath.phase(x)*180/cmath.pi)
    return myRound(x,ndigits)

class Element(object):
    Nodes = set()
    RLCs = []
    Batteries = []
    CSources = []
    ACSources = 0
    def __init__(self,info):
        self.type = info[0][0]
        valid = False
        if self.type in 'RLC' and len(info)==4:
            self.name,*self.nodes,value = info
            self.value,valid = valuate(value),True
        elif self.type in 'VI':
            if info[3] == 'ac' and len(info)==6:
                Element.ACSources+=1
                self.name,*self.nodes, self._type, value,phase = info
                self.value,self.phase,valid = valuate(value)/2,valuate(phase),True
            elif info[3] == 'dc' and len(info)==5:
                self.name,*self.nodes, self._type, value = info
                self.value,self.phase,self.freq,valid = valuate(value),0,0,True
            
            if not valid:   raise Exception
            if self.type == 'V':    Element.Batteries.append(self);self.current = 'I({})'.format(self.name)
            if self.type == 'I':    Element.CSources.append(self)

        Element.Nodes.update(self.nodes)

    def admittance(self,W=0):
        if self.type == 'R':    return 1/self.value
        if self.type == 'C':    return 1j*W*self.value
        if self.type == 'L':    return -1j/(W*self.value)

    def _clearClassVariables():
        Element.Nodes,Element.Batteries,Element.RLCs = set(),[],[]
        Element.ACSources = 0

    __str__ = lambda self: ' '.join('%s: %s'% item for item in vars(self).items())
        #', '.join([attr for attr in dir(self) if not attr.startswith()]
    isRLC = lambda self: self.type in 'RLC'

class Circuit(object):
    def __init__(self,filename):
        self.w = 0
        # input for phase in 
        self.ReadFile(filename)
        self.SolveCircuit()
        self.PrintSolution()

    def ReadFile(self,filename):
        try:
            with open(filename,'r') as netlistFile:
                data = netlistFile.read().split('\n')
        except(FileNotFoundError,IOError):
            print('invalid filename: file not found');quit()
        
        CIRCUIT,END,AC = '.circuit','.end','.ac'
        Start,End = -1,-2
        for line in data:
            if line[:len(CIRCUIT)] == CIRCUIT:
                Start = data.index(line)
            if line[:len(END)] == END:
                End = data.index(line)

        try:
            self.elements = [Element(nonComment(line)) for line in data[Start+1:End]]
            if len(self.elements)==0:
                print('empty valid input in this file');quit()

            self.nodes = list(Element.Nodes)
            self.batteries = Element.Batteries
            self.cSources = Element.CSources
            variables = self.nodes + [i.current for i in self.batteries]
            self.vars = dict(zip(variables,range(len(variables))))

            for line in data[End+1:]:
                line = nonComment(line)
                if len(line)==3 and line[0] == '.ac':
                    #below using a dict instead of two loops
                    for element in {'I':self.cSources,'V':self.batteries}[line[1][0]]: 
                        if element.name == line[1]:
                            element.freq = valuate(line[2])
                            self.w = 2*np.pi*valuate(line[2])
                            Element.ACSources -=1

            if Element.ACSources!=0:
                print('unbalanced frequency inputs')
                raise Exception
        except Exception:
            print('invalid format.');quit()
        
        Element._clearClassVariables()

    def SolveCircuit(self):
        '''identify variables in x matrix'''
        lenX = len(self.vars)
        A,B = np.zeros((lenX,lenX),dtype=np.complex64),np.zeros((lenX,1),dtype=np.complex64)
        for e in self.elements:
            if e.isRLC():
                A[self.vars[e.nodes[0]],self.vars[e.nodes[0]]] += e.admittance(self.w)
                A[self.vars[e.nodes[0]],self.vars[e.nodes[1]]] -= e.admittance(self.w)
                A[self.vars[e.nodes[1]],self.vars[e.nodes[0]]] -= e.admittance(self.w)
                A[self.vars[e.nodes[1]],self.vars[e.nodes[1]]] += e.admittance(self.w)
            if e.type == 'V':
                #assume current is flowing out from +ve/first node
                A[self.vars[e.nodes[0]],self.vars[e.current]] = -1
                A[self.vars[e.nodes[1]],self.vars[e.current]] = 1
                A[self.vars[e.current],self.vars[e.nodes[0]]] = 1
                A[self.vars[e.current],self.vars[e.nodes[1]]] = -1
                B[self.vars[e.current],0] = cmath.rect(e.value,(e.phase*np.pi)/180)
            if e.type == 'I':
                #assume current is flowing out from +ve/first node(from node2 to node1)
                current  = cmath.rect(e.value,(e.phase*np.pi)/180)
                B[self.vars[e.nodes[0]],0] = current
                B[self.vars[e.nodes[1]],0] = -current

        A[self.vars['GND'],:], B[self.vars['GND'],0] = 0,0#overwriting GND equation
        #A[:,self.vars['GND']] = 0
        A[self.vars['GND'],self.vars['GND']] = 1

        solution = np.linalg.solve(A,B).reshape(lenX)
        self.solution = dict(zip(self.vars,solution))
        self.A,self.B = A,B

    def PrintSolution(self,fileObj=sys.stdout):
        print('the answer approximated to 6 decimal places is:')
        print('\n'.join(('{0}\t{1}'.format(i,myRound(self.solution[i]))
                    for i in sorted(self.vars))),end='\n\n',file=fileObj)
        
if __name__ == '__main__':
    if len(sys.argv)>1:
        ckts = [Circuit(i) for i in sys.argv[1:]]
        # this code can take multiple netlist files and parse,solve
        # them in one command line
    else:
        print('Please (proper) filename and try again')
