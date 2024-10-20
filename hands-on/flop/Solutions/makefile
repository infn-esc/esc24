#
#  USAGE:
#     make          ... to build the programs
#     make test     ... to run the default test case
#

CC          = gcc-14
CPP         = g++-14
CLINKER     = $(CC)
CPPLINKER   = $(CPP)
#OPTFLAGS   = -std=c11 -fopenmp -O3
OPTFLAGS    = -O3 -fopenmp -Wall -frounding-math
LIBS        = -lm
CFLAGS	  = $(OPTFLAGS)

EXES = AssocDistCases CounterAccum MachEpsilon expSumSeries quadraticSolve \
       refactorFuncCancelation sumHarmonicForwardAndBack  summation

SUMOBJS = summationWithKahan.o UtilityFunctions.o

all: $(EXES)

AssocDistCases: AssocDistCases.o
	$(CLINKER) $(OPTFLAGS) -o AssocDistCases AssocDistCases.o $(LIBS)

CounterAccum: CounterAccum.o
	$(CLINKER) $(OPTFLAGS) -o CounterAccum CounterAccum.o $(LIBS)

MachEpsilon: MachEpsilon.o
	$(CLINKER) $(OPTFLAGS) -o MachEpsilon MachEpsilon.o $(LIBS)

expSumSeries: expSumSeries.o
	$(CLINKER) $(OPTFLAGS) -o expSumSeries expSumSeries.o $(LIBS)

quadraticSolve: quadraticSolve.o
	$(CLINKER) $(OPTFLAGS) -o quadraticSolve quadraticSolve.o $(LIBS)

refactorFuncCancelation: refactorFuncCancelation.o
	$(CLINKER) $(OPTFLAGS) -o refactorFuncCancelation refactorFuncCancelation.o $(LIBS)

sumHarmonicForwardAndBack: sumHarmonicForwardAndBack.o
	$(CLINKER) $(OPTFLAGS) -o sumHarmonicForwardAndBack sumHarmonicForwardAndBack.o $(LIBS)

summation: $(SUMOBJS) UtilityFunctions.h
	$(CLINKER) $(OPTFLAGS) -o summation $(SUMOBJS)  $(LIBS)

test: $(EXES)
	./AssocDistCases
	./CounterAccum
	./MachEpsilon
	./quadraticSolve
	./expSumSeries
	./refactorFuncCancelation
	./sumHarmonicForwardAndBack
	./summation

clean:
	rm -rf $(EXES) *.o

.SUFFIXES:
.SUFFIXES: .c .cc .o

.c.o:
	$(CC) $(CFLAGS) -c $<

.cc.o:
	$(CC) $(CFLAGS) -c $<

