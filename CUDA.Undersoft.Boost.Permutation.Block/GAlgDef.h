#ifndef GAlgDefH
#define GAlgDefH

#include"problem.h"

typedef void(*fPointer)(problem *, solution&);

struct AlgExp
{
	fPointer fun;
	char *name;
	int active;
	clock_t t, tmin, tmax;

	clock_t t_alloc, tmin_alloc, tmax_alloc;
	clock_t t_del, tmin_del, tmax_del;
	clock_t t_calc, tmin_calc, tmax_calc;

	SIZE_T mem_alloc, mem_alloc_min, mem_alloc_max;
	SIZE_T mem_diff, mem_diff_min, mem_diff_max;
	SIZE_T mem_total, mem_total_min, mem_total_max;

	float err, errmin, errmax;
	long long int k, kmin, kmax;
	long long int k2, k2min, k2max;
	float value;

	AlgExp() {
		active = 0;
		t = 0; tmin = 0; tmax = 0;
		k = 0; kmin = 0; kmax = 0;
		k2 = 0; k2min = 0; k2max = 0;
		t_alloc = 0; tmin_alloc = 0; tmax_alloc = 0;
		t_del = 0; tmin_del = 0; tmax_del = 0;
		t_calc = 0; tmin_calc = 0; tmax_calc = 0;
		mem_alloc = 0; mem_alloc_min = 0; mem_alloc_max = 0;
		mem_diff = 0; mem_diff_min = 0; mem_diff_max = 0;
		mem_total = 0; mem_total_min = 0; mem_total_max = 0;
	};

	void Clear() {
		t = 0; tmin = 0; tmax = 0;
		k = 0; kmin = 0; kmax = 0;
		k2 = 0; k2min = 0; k2max = 0;
		err = 0; errmin = 0; errmax = 0;
		t_alloc = 0; tmin_alloc = 0; tmax_alloc = 0;
		t_del = 0; tmin_del = 0; tmax_del = 0;
		t_calc = 0; tmin_calc = 0; tmax_calc = 0;
		mem_alloc = 0; mem_alloc_min = 0; mem_alloc_max = 0;
		mem_diff = 0; mem_diff_min = 0; mem_diff_max = 0;
		mem_total = 0; mem_total_min = 0; mem_total_max = 0;

	}
};


//------------- Primary Algorithms ---------------//


//inline void gAlgExtSearch(problem *p, solution &result)
//{
//	result.pi.InitPerm(p->Size());
//	p->ExtensiveSearch(result);
//};



//------------- Approximation Algorithms ---------------//


//------ Pm|LE|Cmax ------//




inline void gAlgTS_neigh_iter(problem *p, solution &result)
{
	result.pi.InitPerm(p->Size());
	result.value = p->Criterion(result.pi);
	p->AlgTS_neigh_iter(result, result, 10, 50);
};

inline void gAlgTS_neigh_iter_tab(problem *p, solution &result)
{
	result.pi.InitPerm(p->Size());
	result.value = p->Criterion(result.pi);
	p->AlgTS_neigh_iter_tab(result, result, 10, 50);
};

//inline void gGPU_AlgTSonGPU(problem *p, solution &result)
//{
//	result.pi.InitPerm(p->Size());
//	result.value = p->Criterion(result.pi);
//	p->AlgTSonGPU(result, result, 10, 50);
//
//};

inline void gGPU_AlgTSonGPU_neigh_iter(problem *p, solution &result)
{
	result.pi.InitPerm(p->Size());
	result.value = p->Criterion(result.pi);
	p->AlgTSonGPU_neigh_iter(result, result, 10, 100, 1024);
};

inline void gGPU_AlgTSonGPU_BaT(problem *p, solution &result)
{
	result.pi.InitPerm(p->Size());
	result.value = p->Criterion(result.pi);
	p->AlgTSonGPU_BaT(result, result, 10, 100, 1024, 2);

};

//---------------------//


//------ Metaheuristics ------//


#endif
