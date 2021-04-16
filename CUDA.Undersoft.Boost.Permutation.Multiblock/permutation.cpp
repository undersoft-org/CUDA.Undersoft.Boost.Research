#include "permutation.h"


#include"permutation.h"

//-------------------------------------------//
//------------------- sort ------------------//
//-------------------------------------------//

//----------------- partition --------------//
int partition(int *A, int p, int r)
{
	int i, j, x, tmp;

	x = A[p];
	i = p - 1;
	j = r + 1;

	while (1)
	{
		while (A[--j]>x);
		while (A[++i]<x);
		if (i<j)
		{
			tmp = A[i];
			A[i] = A[j];
			A[j] = tmp;
		}
		else
			return j;
	}
}

//--------------- quicksort ----------------//
void quicksort(int *A, int p, int r)
{
	int q;

	if (p<r)
	{
		q = partition(A, p, r);
		quicksort(A, p, q);
		quicksort(A, q + 1, r);
	}
}



//-------------------------------------------//
//--------------- permutation --------------//
//-------------------------------------------//

//--------------- permutation --------------//
permutation::permutation(int size)
{
	int i;

	n = size;
	perm = new int[n];

	for (i = 0; i<n; i++) perm[i] = i;
}


//-------------- ~permutation --------------//
permutation::~permutation()
{
	int i;

	n = 0;
	delete perm;
	perm = NULL;
}

//------------ copy constructor ------------//
permutation::permutation(const permutation &p)
{
	int i;

	n = p.n;
	perm = new int[n];

	for (i = 0; i<n; i++)
		perm[i] = p.perm[i];
}

//--------------- operator = ---------------//
permutation& permutation::operator =(const permutation &p)
{
	int i;

	if (this == &p) return *this;

	if (perm != NULL)
		delete perm;
	n = p.n;
	perm = new int[n];

	for (i = 0; i<n; i++)
		perm[i] = p.perm[i];

	return *this;
}

//--------------- operator == --------------//
bool permutation::operator ==(const permutation &p)
{
	int i;

	if (this == &p) return 1;
	if (perm == NULL) return 0;
	if (p.perm == NULL) return 0;
	if (p.n != n) return 0;

	for (i = 0; i<n; i++)
		if (perm[i] != p.perm[i])
			return 0;

	return 1;
}

//-------------- operator << ---------------//
ostream& operator <<(ostream& out_data, const permutation& p)
{
	int i;

	out_data << "[";
	for (i = 0; i<p.n - 1; i++)
		out_data << p.perm[i] << ",";

	out_data << p.perm[i] << "]";
	return out_data;
}


//---------------- InitPerm ----------------//
void permutation::InitPerm(int size)
{
	int i;

	if (perm != NULL)
		delete perm;

	n = size;
	perm = new int[n];

	for (i = 0; i<n; i++) perm[i] = i;
}

//---------------- NextPerm ----------------//
bool permutation::NextPerm(int a, int b)
{
	int i, j, k, tmp, el_min;

	if ((a<0) || (b >= n)) return 1;                 //out of range

	i = b;

	while ((perm[i - 1]>perm[i]) && (i>a + 1)) i--;

	if ((i == a + 1) && (perm[a]>perm[a + 1])) return 1; //last permutation

	j = i;
	i--;

	tmp = perm[i];

	for (k = i + 2; k <= b; k++)
		if ((tmp<perm[k]) && (perm[k]<perm[j])) j = k;

	perm[i] = perm[j];
	perm[j] = tmp;

	for (k = 0; k<((b - i) / 2); k++)
	{
		tmp = perm[i + 1 + k];
		perm[i + 1 + k] = perm[b - k];
		perm[b - k] = tmp;
	}
	return 0;
}

//-------------- SortDec -------------------//
void permutation::SortDec(int a, int b)
{
	int i, tmp;

	quicksort(perm, a, b);
	Inverse(a, b);
}


//--------------- Inverse ------------------//
void permutation::Inverse(int a, int b)
{
	int i, tmp;

	for (i = 0; i<((b - a + 1) / 2); i++)				// decreasing sort
	{
		tmp = perm[a + i];
		perm[a + i] = perm[b - i];
		perm[b - i] = tmp;
	}
}


//----------------- Insert -----------------//
void permutation::Insert(int a, int b)
{
	int i;

	if (a == b) return;

	if (a<b)
		for (i = a; i<b; i++)
			Swap(i, i + 1);
	else
		for (i = a; i>b; i--)
			Swap(i - 1, i);
}






