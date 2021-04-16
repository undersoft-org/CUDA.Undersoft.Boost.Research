#include "permutation.h"

//----------------- partition --------------//
int partition(float *A, permutation &pi, int p, int r)
{
	int i, j;
	float x;

	x = A[pi[p]];
	i = p - 1;
	j = r + 1;

	while (1)
	{
		while (A[pi[--j]]>x);
		while (A[pi[++i]]<x);
		if (i<j)
			pi.Swap(i, j);
		else
			return j;
	}
}


//--------------- quicksort ----------------//
void quicksort(float *A, permutation &pi, int p, int r)
{
	int q;

	if (p<r)
	{
		q = partition(A, pi, p, r);
		quicksort(A, pi, p, q);
		quicksort(A, pi, q + 1, r);
	}
}

//----------------- sort_up ----------------//
void sort_up(float *A, permutation &pi, int a, int b)
{
	int tmp;

	if (a<0) a = 0;
	if (b>pi.Size()) b = pi.Size();
	if (a>b) { tmp = a; a = b; b = tmp; }

	quicksort(A, pi, a, b);
}

//--------------- sort-down ----------------//
void sort_down(float *A, permutation &pi, int a, int b)
{
	sort_up(A, pi, a, b);
	pi.Inverse(a, b);
}
