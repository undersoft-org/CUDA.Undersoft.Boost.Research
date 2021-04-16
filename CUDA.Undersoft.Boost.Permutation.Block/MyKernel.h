
#ifndef KERNEL_CUH_
#define KERNEL_CUH_

void GPU_AlgTS_neigh_iter(problem *p, solution &result, const solution sUB, int listN, int iterN, int threadsN);

void GPU_AlgTS_BaT(problem *p, solution &result, const solution sUB, int listN, int iterN, int threadsN, int blocksN);

#endif

