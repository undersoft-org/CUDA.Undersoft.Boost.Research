#ifndef AlgH
#define AlgH

#include"stdafx.h"

#include"permutation.h"

using namespace std;

struct solution;

struct job
{
	//------ step function ------//
	int k;
	float *pj;   //n
	float *aji;  //k
	int *gji;    //k
				 //---------------------------//

	float aj;
	float bj;
	float gj;
	float alphaj;
};


//------------------------------------------//
//----------------- problem ----------------//
//------------------------------------------//

class problem
{
public:
	int n;
	int m;
	job **jobs;	//[i][j] parameters of job j on machine i
	float *Cj;

public:

	int K_KB_MB;

	problem();
	~problem();

	inline int Size() { return n; };

	void SetProblem(int m, int n);
	void ClearProblem();

	//must be abstract in future
	void DrawJobs(int bj_min, int bj_max, float bj_s, int alphaj_min, int alphaj_max, float alphaj_s,
		int gj_min, int gj_max, float gj_s, int aj_min, int aj_max, float aj_s);

	// i - step, j - job, v - position = sum pj
	inline float p_j(int i, int j, int v)     //in v =1,...,n, pj v=0,..., n-1
	{
		//int pj;

		float pj;

		v = max(v, 0);
		
		pj = ((jobs[0][j].aj* (pow(v + 1, -jobs[0][j].alphaj))));


		return (pj);		//dla wszystkich identyczna krzywa
	};



	float Criterion(permutation pi, int h);
	inline float Criterion(permutation pi) { return Criterion(pi, n - 1); };  //[0<---->a...]	
																					//------ algorithms ------//
public:



	//--------- GPU -------//
	void AlgTSonGPU_FU_BaT(solution &result, const solution sUB, int listN, int iterN, int threadsN, int blocksN);
	void AlgTSonGPU_FFU_BaT(solution &result, const solution sUB, int listN, int iterN, int threadsN, int blocksN);	
	void AlgTSonGPU_FFU_BaT_block_sync(solution &result, const solution sUB, int listN, int iterN, int threadsN, int blocksN);
	void PrintJobs();

};


struct solution
{
	permutation pi;
	float value;
	long long int param;
	long long int param2;

	clock_t t_alloc;
	clock_t t_del;
	clock_t t_calc;

	SIZE_T mem_alloc;
	SIZE_T mem_diff;
	SIZE_T mem_total;


};

#endif

