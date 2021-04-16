//#pragma once
#ifndef PermH
#define PermH

#include"stdafx.h"

class permutation
{
private:
	int n;
	int *perm;

public:

	permutation() { n = 0; perm = NULL; };
	permutation(int size);
	~permutation();

	permutation(const permutation &p);
	inline const int& operator [](int i) { return perm[i]; };
	permutation& operator =(const permutation &p);
	bool operator ==(const permutation &p);

	// mozna to zrobic przesz przypisanie tablicy i uszeregowanie jej - bezpieczniejsze -> zawsze bedzie permutacja
	inline void Set(int i, int a) { perm[i] = a; };

	inline int Size() { return n; };

	void InitPerm(int size);

	inline void Swap(int a, int b) { int tmp; tmp = perm[a]; perm[a] = perm[b]; perm[b] = tmp; }
	void Insert(int a, int b);			   // [...,a,.....,b,...]->[........,a,b,...]
	void Inverse(int a, int b);            // Inverse betwen 'a' and 'b'
	inline void Inverse() { Inverse(0, n - 1); };

	void SortDec(int a, int b);

	bool NextPerm(int a, int b);        //NextPerm [...,a<--->b,....]
	inline bool NextPerm() { return(NextPerm(0, n - 1)); };

	friend ostream& operator << (ostream& out_data, const permutation& p);
};

#endif
