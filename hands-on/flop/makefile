#
#  USAGE:
#     make          ... to build the programs
#     make test     ... to run the default test case
#

CC          = gcc-13
CPP         = g++-13
CLINKER     = $(CC)
CPPLINKER   = $(CPP)
#OPTFLAGS   = -std=c11 -fopenmp -O3
OPTFLAGS    = -O0 -fopenmp -Wall -frounding-math
LIBS        = -lm
CFLAGS	  = $(OPTFLAGS)

EXES = roundingC roundingCpp  summation

SUMOBJS = summation.o UtilityFunctions.o

all: $(EXES)

roundingC: roundingC.o
	$(CLINKER) $(OPTFLAGS) -o roundingC roundingC.o $(LIBS)

roundingCpp: roundingCpp.o
	$(CPPLINKER) $(OPTFLAGS) -o roundingCpp roundingCpp.o $(LIBS)

summation: $(SUMOBJS)
	$(CLINKER) $(OPTFLAGS) -o summation $(SUMOBJS)  $(LIBS)

test: $(EXES)
	./roundingC
	./roundingCpp
	./summation

clean:
	rm -rf $(EXES) *.o

.SUFFIXES:
.SUFFIXES: .c .cc .o

.c.o:
	$(CC) $(CFLAGS) -c $<

.cc.o:
	$(CC) $(CFLAGS) -c $<

