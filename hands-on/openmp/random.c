
//**********************************************************
// Pseudo random number generator:
//     double drandom
//     void seed (lower_limit, higher_limit)
//**********************************************************
//
// A simple multiplicative linear congruential random number generator
// (Numerical Recipies chapter 7, 1st ed.) with parameters
// from the table on page 198.
//
//  Uses a pseudo-random number generator to return a value between
//  0 and 1, then scales and shifts it to fill the desired range.  This
//  range is set when the random number generator seed is called.
// 
// USAGE:
//
//      pseudo random sequence is seeded with a range
//
//            void seed(lower_limit, higher_limit)
//   
//      and then subsequent calls to the random number generator generates values
//      in the sequence:
//
//            double drandom()
//
// History: 
//      Written by Tim Mattson, 9/2007.
//      changed to drandom() to avoid collision with standard libraries, 11/2011
 
// A much better set of MLCG parameters
// from Pierre L'Ecuyer, Mathematics of Computation, Vol 68, Jan. 1999, pp 249-260
//define PMOD        2147483647
//define MULTIPLIER  1583458089
//define SEED        483647

// The infamous RANDU generator from IBM
#define PMOD        2147483648
#define MULTIPLIER  65539
#define SEED        483647

long random_last =  SEED;
double random_low, random_hi;

double drandom()
{
    long random_next;
    double ret_val;

// 
// compute an integer random number from zero to mod
//
    random_next = (MULTIPLIER  * random_last)% PMOD;
    random_last = random_next;

//
// shift into preset range
//
    ret_val = ((double)random_next/(double)PMOD)*(random_hi-random_low)+random_low;
    return ret_val;
}
//
// set the seed and the range
//
void seed(double low_in, double hi_in)
{
   if(low_in < hi_in)
   { 
      random_low = low_in;
      random_hi  = hi_in;
   }
   else
   {
      random_low = hi_in;
      random_hi  = low_in;
   }
   random_last = SEED;

}
//**********************************************************
// end of pseudo random generator code.
//**********************************************************

