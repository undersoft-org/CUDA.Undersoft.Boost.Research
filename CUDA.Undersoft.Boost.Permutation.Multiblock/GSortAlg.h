#ifndef GSortAlgH
#define GSortAlgH
#include "permutation.h"

void quicksort(float *A, permutation &pi, int p, int r);
void sort_up(float *A, permutation &pi, int a, int b);
void sort_down(float *A, permutation &pi, int a, int b);


#endif
