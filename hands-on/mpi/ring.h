/*******************************************************************

NAME:    ring.h

PURPOSE: include file for the MPI ring tests.

HISTORY: Written by Tim Mattson, April 1999
*******************************************************************/

#include "mpi.h"
#include <stdlib.h>
//#include <malloc.h>
#include <stdio.h>

int ring_naive( double*, double*, int, int, int, int);

#define IS_ODD(x)  ((x)%2)           /* test for an odd int */
#define dabs(x) ((x)>0?(x):(-(x)))   /* absolute value */

#define False 0
#define True  1
#define TOL   0.001       /*  Tolerance used in floating point compares */
#define MB_CONV  1.0e-6   /*  conversion factor ... bytes to Megabytes */

