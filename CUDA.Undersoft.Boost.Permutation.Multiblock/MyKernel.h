
#ifndef KERNEL_CUH_
#define KERNEL_CUH_

void GPU_AlgTS_FU_BaT(problem *p, solution &result, const solution sUB, int listN, int iterN, int threadsN, int blocksN);

void GPU_AlgTS_FFU_BaT(problem *p, solution &result, const solution sUB, int listN, int iterN, int threadsN, int blocksN);

void GPU_AlgTS_FFU_BaT_block_sync(problem *p, solution &result, const solution sUB, int listN, int iterN, int threadsN, int blocksN);


#endif

