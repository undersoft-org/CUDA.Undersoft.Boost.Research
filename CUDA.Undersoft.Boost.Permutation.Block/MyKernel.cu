#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_device_runtime_api.h"

#include <curand.h>
#include <curand_kernel.h>

#include "device_functions.h"	//atomicAdd - required for mutex


#include <stdio.h>
#include <iostream>
#include"problem.h"

//liczba bokow tylko potegi 2, (blocks 12 dziala jak blocks 8)

#define MAX_BLOCKS 16				//GeForce 770GTX nie ma sensu wiecej niz 16, wiecej tylko wolniej
#define MAX_THREADS 1024			//if greater then GPU does not calculate at all, if more than 1024 - alg on GPU does not run
#define MAX_PROBLEM_SIZE 1024
#define MAX_MACHINE_SIZE 128

#define MAX_TABU_LIST 128
#define BIG_NUMBER 1000000

using namespace std;

__device__ int g_mutex;

__device__ int g_blocksCompleted[2];		//completed calculations
__device__ int g_blocksReady[1];			//copied global to shared and ready for next iteration ([MAX_BLOCKS], ale 1 zeby miesjca nie zajmowalo podczas testow
__device__ int g_blocksActive[1];			//can go futher
__device__ float g_value[2];
__device__ int g_best_j[2];
__device__ int g_best_v[2];

__device__ float g_value_overall;
__device__ int g_best_j_overall;
__device__ int g_best_v_overall;

__device__ int l_mutex;
__device__ inline void my_lock(void) {
	while (atomicCAS(&l_mutex, 0, 1) != 0);
}
__device__ inline void my_unlock(void) {
	atomicExch(&l_mutex, 0);
}

void HostSwapTable(int *perm, int a, int b)
{ 
	int tmp; 
	tmp = perm[a]; 
	perm[a] = perm[b]; 
	perm[b] = tmp; 
}

void HostInsertTable(int *perm, int a, int b)
{
	int i;

	if (a == b) return;

	if (a < b)
	{
		for (i = a; i < b; i++)
		{
			HostSwapTable(perm, i, i + 1);
		}
	}
	else
	{
		for (i = a; i > b; i--)
		{
			HostSwapTable(perm, i - 1, i);
		}
	}
}


__device__ 	inline void SwapTable(int *perm, int a, int b)
{
	int tmp;
	tmp = perm[a];
	perm[a] = perm[b];
	perm[b] = tmp;
}

__device__ inline void InsertTable(int *perm, int a, int b)
{
	int i;

	if (a == b) return;

	if (a < b)
	{
		for (i = a; i < b; i++)
		{
			SwapTable(perm, i, i + 1);
		}
	}
	else
	{
		for (i = a; i > b; i--)
		{
			SwapTable(perm, i - 1, i);
		}
	}
}



__device__  inline float Criterion(int *n_s, int *m_s, int *pi, float *d_aj_s, float *d_alphaj_s, float *CP_shared)
{
	//float CP[10];
	int i, j;
	float v;
	float C;
	int lindex = threadIdx.x;

	
	//for (i = 0; i<m_s[lindex]; i++) CP[i] = 0;

	for (i = 0; i<m_s[lindex]; i++) CP_shared[lindex*MAX_MACHINE_SIZE + i] = 0;

	v = 0;
	i = 0;
	for (j = 0; j<n_s[lindex]; j++)
	{
		if (pi[j] == n_s[lindex] - 1)
		//if (pi_shared[lindex * MAX_PROBLEM_SIZE + j] == *d_n - 1)
		{
			i = 1;
			v = 0;
		}
		else
		{
			CP_shared[lindex * MAX_MACHINE_SIZE + i] += d_aj_s[pi[j]] * pow(float(v + 1), -d_alphaj_s[pi[j]]);
			//CP[i] += d_aj_s[pi[j]] * pow(float(v + 1), -d_alphaj_s[pi[j]]);
			v += d_aj_s[pi[j]];	//sum pj

			//CP[i] += d_aj[pi_shared[lindex * MAX_PROBLEM_SIZE + j]] * pow(float(v + 1), -d_alphaj[pi_shared[lindex * MAX_PROBLEM_SIZE + j]]);
			//v += d_aj[pi_shared[lindex * MAX_PROBLEM_SIZE + j]];	//sum pj
		}
	}

	//C = CP[0];
	C = CP_shared[lindex * MAX_MACHINE_SIZE];

	for (i = 1; i<m_s[lindex]; i++)
	{
		//if (C<CP[i]) C = CP[i];
		if (C<CP_shared[lindex * MAX_MACHINE_SIZE + i]) C = CP_shared[lindex * MAX_MACHINE_SIZE + i];
	}

	return(C);
}


__device__  inline float Criterion2(int *n, int *m, int *pi, float *d_aj_s, float *d_alphaj_s, float *CP)
{
	int i, j;
	float v;
	float C;
	int lindex = threadIdx.x;

	for (i = 0; i<*m; i++) CP[i] = 0;

	v = 0;
	i = 0;
	for (j = 0; j<*n; j++)
	{
		if (pi[j] >= (*(n)) - (*(m)) + 1)		
		{
			i++;			
			v = 0;
		}
		else
		{
			CP[i] += d_aj_s[pi[j]] * pow(float(v + 1.0), -d_alphaj_s[pi[j]]);
			v += d_aj_s[pi[j]];	
		}
	}

	C = CP[0];

	for (i = 1; i<*m; i++)
	{
		if (C < CP[i]) C = CP[i];
	}

	return(C);

}


//------------ Kernel TS 3 -------------//
// insert by swap - multithread = number of jobs
// shared tabu list
// shared pi_neigh_best - wolniejsze niz w TS2 bez shared tylko kazdy watek ma swoj pi_neigh_best (1024 - 51s vs 53s)
// shared pi_best
//
// parallel moze dawac inny wynik niz jednowatkowy,
// gdyz to samo kryterium moze byc dla roznych j,v, wowczas inne j,v trafiaja na liste tabu (w innej kolejnosci) 
////
// mozna zrobic, zeby tylko lindex == 0 sprawdzal najlepsze wstawienie i przebudowal shared_pi_neigh_best
// trzeba wtedy dodac shared_best_j, shared_best_v
//
// !!! zerowac wpisy, moze byc pozostaly z wczesniejszego, np. 5, 5, 4, 4, watki nie zmienia swoich wartosci
__global__ void KernelTS3_neigh_iter(int *d_threadsN, int *d_n, int *d_m, int *d_pi, float *d_aj, float *d_alphaj, float *d_parametersTS, unsigned int seed)
{
	int i, l;
	int jobi, jobv1, jobv2, best_jobv1, best_jobv2, j_best, v_best;
	int iter, iterN;
	int listN, isInTabu;
	int index_best;

	int firstNeigh;

	//int listTabu[2 * MAX_TABU_LIST];		//[1,1; 2,2; ...]

	float CP[MAX_MACHINE_SIZE];				//for criterion value

	float value_C, value_neigh_best;
	int pi[MAX_PROBLEM_SIZE];				// temporary pi for analysing neighbourhood

	__shared__ int shared_listIdx;
	__shared__ int listTabu[2 * MAX_TABU_LIST];		//[1,1; 2,2; ...]

	__shared__ int shared_pi_neigh_best[MAX_PROBLEM_SIZE];	//the best for the neighourhood
	__shared__ int shared_pi_best[MAX_PROBLEM_SIZE];			//overall the best - result

	__shared__ float value_shared[MAX_THREADS];	//the best solution value for each thread

	__shared__ float aj_shared[MAX_PROBLEM_SIZE];			//much faster than local float aj_shared[MAX_PROBLEM_SIZE];
	__shared__ float alphaj_shared[MAX_PROBLEM_SIZE];

	__shared__ float shared_value_neigh_best;
	__shared__ float shared_value_best;

	//float aj_shared[MAX_PROBLEM_SIZE];
	//float alphaj_shared[MAX_PROBLEM_SIZE];


	

	int all_completed = 0;
	int lindex = threadIdx.x;

	int threadsN = *d_threadsN;
	int n = *d_n;
	int m = *d_m;

	int j, v, k;
	int N;				//number of moves per neighbourhood
	int N_k;					// number of moves per this thread
	int rem_N;					//reminder
	int div_N;					//division
	int index_start_move_k;		//the index of a start move of this thread, e.g., (0,0) is 0, (0,1) is 1, etc. (0,n-1) is n-1, (1,0) is n
	int job_j;
	int job_v;

	
	//for random values
	// if random values required 
	curandState_t state;
	curand_init(seed + 17 * lindex, 0, 0, &state);	//it should be a time(NULL)

	listN = d_parametersTS[0];
	iterN = d_parametersTS[1];

	//initialize shared parameters
	if (lindex == 0)
	{

		//problem parameters
		for (j = 0; j<n; j++)
		{
			aj_shared[j] = d_aj[j];
			alphaj_shared[j] = d_alphaj[j];
		}

		//tabu list
		shared_listIdx = 0;
		for (i = 0; i<listN * 2; i++)
		{
			listTabu[i] = 0;
		}

		//solutions
		for (j = 0; j < n; j++)
		{
			shared_pi_best[j] = d_pi[j];
			shared_pi_neigh_best[j] = shared_pi_best[j];
		}

		//caclulate the best value
		shared_value_best = Criterion2(&n, &m, shared_pi_best, aj_shared, alphaj_shared, CP);
		shared_value_neigh_best = shared_value_best;

		//firstNeigh = 1;
	}

	firstNeigh = 1;

	// wait for all threads to reach the barrier
	__syncthreads();

	value_neigh_best = shared_value_neigh_best;

	N = n*n;
	rem_N = N % threadsN;
	div_N = (int)(N / threadsN);
	N_k = div_N + (lindex < rem_N);
	index_start_move_k = lindex*div_N + rem_N*(lindex >= rem_N) + lindex*(lindex < rem_N); //lindex >= rem_N->rem_n otherwise lindex


	// okreslone tylko liczy dany watek
	for (iter = 0; iter<iterN; iter++)
	{
		best_jobv1 = 0;	//means that nothing has changed in reference to the previouse solution (po iteracji 0,0 oznacza, ze bez zmian)
		best_jobv2 = 0;

		//firstNeigh = 1;

		firstNeigh = 0;

		value_neigh_best = BIG_NUMBER;

		//pi = pi_neigh_best
		//for (l = 0; l < n; l++)
		//{
		//	pi[l] = shared_pi_neigh_best[l];
		//}

		j = (int)(index_start_move_k / n);
		v = index_start_move_k % n;
		job_j = -1;

		for (k = 0; k < N_k; k++)
		{
			//------- obtain move --------//
			if (job_j == j)
			{
				SwapTable(pi, v - 1, v);	//here v is > 0 - always
				job_j = j;
				job_v = v;
			}
			else
			{
				//pi = pi_neigh_best
				for (l = 0; l < n; l++)
				{
					pi[l] = shared_pi_neigh_best[l];
				}

				InsertTable(pi, j, v);
				job_j = j;
				job_v = v;
			}
			//----------------------------//

			if (j != v)
			{
				value_C = Criterion2(&n, &m, pi, aj_shared, alphaj_shared, CP);

				//---------- check Tabu List -------------//
				if ((job_j != job_v) && (firstNeigh == 1 || value_C < value_neigh_best))
				{
					isInTabu = 0;
					for (l = 0; l < listN; l++)
					{
						if ((job_j == listTabu[0 * listN + l]) && (job_v == listTabu[1 * listN + l])) //[0][l], [1][l]
						{
							isInTabu = 1;
							break;
						}
					}                     

					if (!isInTabu)
					{
						value_neigh_best = value_C;
						firstNeigh = 0;
						best_jobv1 = job_j;
						best_jobv2 = job_v;
					}
				}
				//----------------------------------------//
			}

			//---- update j and v ---//
			v++;
			if (v >= n)
			{
				v = 0;
				j++;
			}
			//-----------------------//
		}

		value_shared[lindex] = value_neigh_best;

		// wait for all threads to reach the barrier
		__syncthreads();

		//choose the best insert
		index_best = lindex;		//if value_i = value_k, then solution i is chosen instead of solution k (i<k)
		for (i = 0; i < threadsN; i++)	//to the number of threads, check if the currecn thread is the best
		{
			if ((value_shared[i] <  value_neigh_best) || (value_shared[i] == value_neigh_best) && (i < lindex)) //this thread (lindex) is not the best
			{
				index_best = -1;
				break;
			}
		}

		//only the best update pi, criterion, tabuList
		if (lindex == index_best)
		{
			shared_value_neigh_best = value_neigh_best;
			InsertTable(shared_pi_neigh_best, best_jobv1, best_jobv2);

			if (shared_value_best > shared_value_neigh_best)
			{
				shared_value_best = shared_value_neigh_best;
				for (l = 0;l < n; l++)
				{
					shared_pi_best[l] = shared_pi_neigh_best[l];
				}
			}

			//------ add to tabu list -------//
			//- list can be shared later on -//
			listTabu[0 * listN + shared_listIdx] = best_jobv1;
			listTabu[1 * listN + shared_listIdx] = best_jobv2;

			shared_listIdx = (shared_listIdx + 1) % listN;
			//------------------------------//
			//firstNeigh = 1;
		}
		//shared_neigh_best mozna w shared value pamietac, dla kazdego w osobnej kolumnie

		//firstNeigh = 1;

		//value_shared[lindex] = BIG_NUMBER;

		// wait for all threads to reach the barrier
		__syncthreads();

		//--- update local from shared values ---//
		//value_neigh_best = shared_value_neigh_best;
		//---------------------------------------//
	}


	//--------- copy best solution to host -----------//	
	if (lindex == 0)
	{
		for (j = 0; j < n; j++)
		{
			d_pi[j] = shared_pi_best[j];
		}
	}
}


//------------ GPU AlgTS -------------//
// one block TS
void GPU_AlgTS_neigh_iter(problem *p, solution &result, const solution sUB, int listN, int iterN, int threadsN)
{
	int i, j, v;

	int *pi, n, m;
	float *aj, *alphaj;
	float *parametersTS;

	int *d_pi, *d_n, *d_m;
	float *d_aj, *d_alphaj;
	float *d_parametersTS;

	int *d_threadsN;

	clock_t t1, t2;



	result = sUB;

	n = p->n;
	m = p->m;

	aj = new float[n];
	alphaj = new float[n];

	//--------------------//

	//threadsN = MAX_THREADS;
	if (threadsN <= 0)
	{
		threadsN = MAX_THREADS;
	}
	threadsN = min(threadsN, MAX_THREADS);
	threadsN = min(threadsN, n*n);		// cannot be greater than neighourhood size, N= n*n

										//--------------------//

	for (j = 0; j < n; j++)
	{
		aj[j] = p->jobs[0][j].aj;
		alphaj[j] = p->jobs[0][j].alphaj;
	}

	parametersTS = new float[2];
	parametersTS[0] = listN;
	parametersTS[1] = iterN;


	cout << "===== TS GPU iter =====" << endl;
	cout << "threadsN " << threadsN << endl;
	cout << "listN " << parametersTS[0] << endl;
	cout << "iterN " << parametersTS[1] << endl;
	cout << "value " << result.value << endl;


	pi = new int[n];

	for (j = 0; j < n; j++)
	{
		pi[j] = result.pi[j];
	}


	t1 = clock();

	cudaMalloc(&d_threadsN, sizeof(int));
	cudaMalloc(&d_pi, n * sizeof(int));
	cudaMalloc(&d_aj, n * sizeof(float));
	cudaMalloc(&d_alphaj, n * sizeof(float));
	cudaMalloc(&d_n, sizeof(int));
	cudaMalloc(&d_m, sizeof(int));
	cudaMalloc(&d_parametersTS, 2 * sizeof(float));

	t2 = clock();

	cout << "cuda alloc " << t2 - t1 << endl;

	t1 = clock();

	cudaMemcpy(d_threadsN, &threadsN, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pi, pi, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_aj, aj, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_alphaj, alphaj, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_m, &m, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_parametersTS, parametersTS, 2 * sizeof(float), cudaMemcpyHostToDevice);

	t2 = clock();

	cout << "cuda mem copy " << t2 - t1 << endl;

	t1 = clock();

	int blocksN;

	blocksN = MAX_BLOCKS;


	KernelTS3_neigh_iter << < 1, threadsN >> > (d_threadsN, d_n, d_m, d_pi, d_aj, d_alphaj, d_parametersTS, time(NULL));

	t2 = clock();

	cout << "cuda run " << t2 - t1 << endl;

	t1 = clock();

	cudaMemcpy(pi, d_pi, n * sizeof(int), cudaMemcpyDeviceToHost);


	t2 = clock();

	cout << "cuda mem to host " << t2 - t1 << endl;

	//cudaMemcpy(pi, d_pi, n * sizeof(int), cudaMemcpyDeviceToHost);

	for (j = 0; j < n; j++)
	{
		//cout << pi[j] << " ";
		result.pi.Set(j, pi[j]);
	}
	//cout << endl;

	//cout << "pi " << endl;
	//cout << result.pi << endl;

	result.value = p->Criterion(result.pi);

	t1 = clock();

	cudaFree(d_threadsN);
	cudaFree(d_pi);
	cudaFree(d_aj);
	cudaFree(d_alphaj);
	cudaFree(d_n);
	cudaFree(d_m);
	cudaFree(d_parametersTS);

	t2 = clock();

	cout << "cuda free " << t2 - t1 << endl << endl;


	delete[]parametersTS;
	delete[]aj;
	delete[]alphaj;
	delete[]pi;
}


//--------- global variable for block to synchronize blocks -----------//


//------------ Kernel TS BaT -------------//
// blocks and threads 
//
// teoretycznie powinno dzialac, ale nie dziala, gdyz nie zawsze bloki moga byc zsynchronizowane
// moze byc tak, ze scheduler czeka na zakonczenie bloku, zanim kolejny zacznie sie wykonywac
// 
// insert by swap - multithread = number of jobs
// shared tabu list
// shared pi_neigh_best - wolniejsze niz w TS2 bez shared tylko kazdy watek ma swoj pi_neigh_best (1024 - 51s vs 53s)
// shared pi_best
//
// parallel moze dawac inny wynik niz jednowatkowy,
// gdyz to samo kryterium moze byc dla roznych j,v, wowczas inne j,v trafiaja na liste tabu (w innej kolejnosci) 
////
// mozna zrobic, zeby tylko lindex == 0 sprawdzal najlepsze wstawienie i przebudowal shared_pi_neigh_best
// trzeba wtedy dodac shared_best_j, shared_best_v
//
// ??? czy n wlicza m?, czy musze robic (n+m)*(n+m), ale chyba n wlicza m
// niebezpieczna sytuacja do sprawdzanie 111|100, wtedy block drugi jak watki wyladaja?, albo 111|000 czy wtedy block 2 powstanie?
//??? a jak wszystkie ruchy danego watku na liscie tabu? czy cos bedzie mial lepszego niz wczesniej?, czy zamazywac duza wartoscia 10000000
// 
// 
__global__ void KernelTS_BaT1(int *d_blocksN, int *d_threads_per_block, int *d_n, int *d_m, int *d_pi, float *d_aj, float *d_alphaj, float *d_parametersTS, unsigned int seed, int *d_err_note)
{
	int i, l, r;
	int jobi, jobv1, jobv2, best_job_j, best_job_v, j_best, v_best;
	int iter, iterN;
	int listIdx, listN, isInTabu;
	int index_best;
	int index_block_best;

	int firstNeigh;

	//int listTabu[2 * MAX_TABU_LIST];		//[1,1; 2,2; ...]

	float CP[MAX_MACHINE_SIZE];				//for criterion value

	float value_C, value_neigh_best;
	int pi[MAX_PROBLEM_SIZE];				// temporary pi for analysing neighbourhood

	__shared__ int listTabu[2 * MAX_TABU_LIST];		//[1,1; 2,2; ...]

	__shared__ int shared_pi_neigh_best[MAX_PROBLEM_SIZE];	//the best for the neighourhood
	__shared__ int shared_pi_best[MAX_PROBLEM_SIZE];			//overall the best - result

	__shared__ float value_shared[MAX_THREADS];	//the best solution value for each thread
	__shared__ int shared_best_j[MAX_THREADS];
	__shared__ int shared_best_v[MAX_THREADS];

	__shared__ float aj_shared[MAX_PROBLEM_SIZE];			//much faster than local float aj_shared[MAX_PROBLEM_SIZE];
	__shared__ float alphaj_shared[MAX_PROBLEM_SIZE];

	__shared__ float shared_value_neigh_best;
	__shared__ float shared_value_best;

	//float aj_shared[MAX_PROBLEM_SIZE];
	//float alphaj_shared[MAX_PROBLEM_SIZE];

	int threads_per_blockN = *d_threads_per_block;
	int blocksN = *d_blocksN;
	int threadsN = (*d_threads_per_block)*(*d_blocksN);
	int n = *d_n;
	int m = *d_m;

	int j, v, k;
	int N;				//number of moves per neighbourhood
	int N_k;					// number of moves per this thread
	int rem_N;					//reminder
	int div_N;					//division
	int index_start_move_k;		//the index of a start move of this thread, e.g., (0,0) is 0, (0,1) is 1, etc. (0,n-1) is n-1, (1,0) is n
	int job_j;
	int job_v;


	int all_completed = 0;
	int all_ready = 0;
	int gindex;
	int block_index = blockIdx.x;
	
	int wait_couter = 0;

	gindex = threadIdx.x + blockIdx.x*threads_per_blockN;

	//gindex = blockIdx.x * blockDim.x + threadIdx.x;

	int lindex = threadIdx.x;	//local index for each block
	//gindex = threadIdx.x;
	//threadsN = threadsN;
	//threadsN = blockDim.x * gridDim.x;	//numeber of threads

	
	//for random values
	// if random values required 
	//curandState_t state;
	//curand_init(seed + 17 * lindex, 0, 0, &state);	//it should be a time(NULL)

	listN = d_parametersTS[0];
	iterN = d_parametersTS[1];

	//initialize shared parameters
	
	if (lindex == 0)
	{
		//for block synchronization
		atomicExch(&l_mutex, 0);		
		atomicExch(&g_mutex, 0);	//nie jest teraz uzywany
		
		g_blocksCompleted[block_index] = 0;
		g_blocksReady[block_index] = 0;
		g_blocksActive[block_index] = 0;

		g_value[block_index] = BIG_NUMBER;
		g_best_j[block_index] = 0;
		g_best_v[block_index] = 0;

		//problem parameters
		for (j = 0; j<n; j++)
		{
			aj_shared[j] = d_aj[j];
			alphaj_shared[j] = d_alphaj[j];
		}

		//tabu list
		for (i = 0; i<listN * 2; i++)
		{
			listTabu[i] = 0;
		}

		//solutions
		for (j = 0; j < n; j++)
		{
			shared_pi_best[j] = d_pi[j];
			shared_pi_neigh_best[j] = shared_pi_best[j];
		}

		//caclulate the best value
		shared_value_best = Criterion2(&n, &m, shared_pi_best, aj_shared, alphaj_shared, CP);
		shared_value_neigh_best = shared_value_best;

		//firstNeigh = 1;
	}

	firstNeigh = 1;

	// wait for all threads to reach the barrier
	__syncthreads();

	value_neigh_best = shared_value_neigh_best;

	listIdx = 0;
	N = n*n;
	rem_N = N % threadsN;
	div_N = (int)(N / threadsN);
	N_k = div_N + (gindex < rem_N);
	index_start_move_k = gindex*div_N + rem_N*(gindex >= rem_N) + gindex*(gindex < rem_N); //gindex >= rem_N->rem_n otherwise gindex


																						   // okreslone tylko liczy dany watek
	for (iter = 0; iter < iterN; iter++)
	{
		
		//do testow
		if (lindex == 0)
		{
			g_blocksCompleted[block_index] = 0;
			g_blocksReady[block_index] = 0;
			g_blocksActive[block_index] = 0;
			g_value[block_index] = BIG_NUMBER;				
		}
		__syncthreads();
		


		best_job_j = 0;	//means that nothing has changed in reference to the previouse solution (po iteracji 0,0 oznacza, ze bez zmian)
		best_job_v = 0;

		//firstNeigh = 1;

		firstNeigh = 0;	//moze nie byc potrzebny

		value_neigh_best = BIG_NUMBER;

		//pi = pi_neigh_best
		for (l = 0; l < n; l++)
		{
			pi[l] = shared_pi_neigh_best[l];
		}

		j = (int)(index_start_move_k / n);
		v = index_start_move_k % n;
		job_j = -1;

		
		for (k = 0; k < N_k; k++)
		{
			//------- obtain move --------//
			if (job_j == j)
			{
				SwapTable(pi, v - 1, v);	//here v is > 0 - always
				job_j = j;
				job_v = v;
			}
			else
			{
				//pi = pi_neigh_best
				for (l = 0; l < n; l++)
				{
					pi[l] = shared_pi_neigh_best[l];
				}

				InsertTable(pi, j, v);
				job_j = j;
				job_v = v;
			}
			//----------------------------//

			if (j != v)
			{
				value_C = Criterion2(&n, &m, pi, aj_shared, alphaj_shared, CP);

				//---------- check Tabu List -------------//
				if ((job_j != job_v) && (firstNeigh == 1 || value_C < value_neigh_best))
				{
					isInTabu = 0;
					for (l = 0; l < listN; l++)
					{
						if ((job_j == listTabu[0 * listN + l]) && (job_v == listTabu[1 * listN + l])) //[0][l], [1][l]
						{
							isInTabu = 1;
							break;
						}
					}

					if (!isInTabu)
					{
						value_neigh_best = value_C;
						firstNeigh = 0;
						best_job_j = job_j;
						best_job_v = job_v;
					}
				}
				//----------------------------------------//
			}

			//---- update j and v ---//
			v++;
			if (v >= n)
			{
				v = 0;
				j++;
			}
			//-----------------------//
		}

		value_shared[lindex] = value_neigh_best;
		shared_best_j[lindex] = best_job_j;
		shared_best_v[lindex] = best_job_v;

		// wait for all threads to reach the barrier		
		__syncthreads();
		//<-------------------- barrier -------//


		//sprawdzic czy takie podejscie nie jest wolniejsze niz index_best, przez np. zajmowanie pamieci
		//choose the best insert by local threads
		if (lindex == 0)
		{
			index_best = 0;		
			value_neigh_best = value_shared[index_best];

			for (i = 1; i < threads_per_blockN; i++)	//to the number of threads, check if the currecn thread is the best
			{
				if (value_shared[i] < value_neigh_best)
				{
					value_neigh_best = value_shared[i];
					index_best = i;
				}
			}

			value_neigh_best = value_shared[index_best];
			best_job_j = shared_best_j[index_best];
			best_job_v = shared_best_v[index_best];		
		//}

		//insert local best to global and find the best among global
		//if (lindex == 0)
		//{
			// announce own the best solution
			my_lock();
			g_value[block_index] = value_neigh_best;
			g_best_j[block_index] = best_job_j;
			g_best_v[block_index] = best_job_v;

			///*
			// calculations completed and copied to global 

			g_blocksActive[block_index] = 0;
			g_blocksCompleted[block_index] = 1;
			my_unlock();
			
			//if (block_index == 0) b0 = 1;
			//if (block_index == 1) b1 = 1;
			//if (block_index == 2) b2 = 1;
			//if (block_index == 3) b3 = 1;

			//atomicExch(&g_blocksCompleted[block_index], 1);
			
			//my_unlock();

			//--- synchronize threads by the main thread gindex==0 ---//
			if (gindex == 0)
			{
				all_completed = 0; //wait all to complete
				//while (all_completed == 0)
				
				for (r = 0; r < 50; r++)
				{
					my_lock();
					all_completed = 1;
					for (i = 0; i < blocksN; i++)
					{
						all_completed *= g_blocksCompleted[i];
					}
					my_unlock();

					if (all_completed == 1)
					{
						break;
					}
				}
				//all blocks have completed

				//find the best move
				index_block_best = 0;
				value_neigh_best = g_value[index_block_best];

				for (i = 1; i < blocksN; i++)	//to the number of threads, check if the currecn thread is the best
				{
					if (g_value[i] < value_neigh_best) //this thread (lindex) is not the best
					{
						index_block_best = i;
					}
				}
				
				best_job_j = g_best_j[index_block_best];
				best_job_v = g_best_v[index_block_best];
				value_neigh_best = g_value[index_block_best];
				
				// assign the best move
				g_best_j_overall = best_job_j;
				g_best_v_overall = best_job_v;
				g_value_overall = value_neigh_best;
				
				//allow other block to start (to be active again)
				for (i = 0; i < blocksN; i++)
				{
					atomicExch(&g_blocksActive[i], 1);
				}										
			}

			//other local main threads (lindex == 0) wait for the global main thread gindex (finding the best move)
			for (r = 0; r < 100; r++)
			{
				if (g_blocksActive[block_index] == 1)
				{
					break;
				}
			}
			
			//g_blocksCompleted[block_index] = 0;

			 //*/
		
			//synchronize all blocks
			//atomicAdd(&g_mutex, 1);
			//while (g_mutex != blocksN)		//zeby nie czyscie g_mutex (g_mutex = 0) mozna sprawdzac czy != blocksN*liter
			//g_mutex = 0;
			//while (g_mutex < 10000000)		//zeby nie czyscie g_mutex (g_mutex = 0) mozna sprawdzac czy != blocksN*liter
			//{
			//	g_mutex = g_mutex + 1;
			//}
			//all blocks completed


			//the best j,v, value from blocks // if equal then the first from the list
			//my_lock();
			//index_block_best = 0;
			//value_neigh_best = g_value[index_block_best];
			
			//for (i = 1; i < blocksN; i++)	//to the number of threads, check if the currecn thread is the best
			//{
			//	if (g_value[i] < value_neigh_best) //this thread (lindex) is not the best
			//	{
			//		index_block_best = i;
			//	}
			//}
			
			
			//my_lock();
			best_job_j = g_best_j_overall;
			best_job_v = g_best_v_overall;
			value_neigh_best = g_value_overall;			
			//my_unlock();

			//update shared  (for given block) according to global values j, v
			shared_value_neigh_best = value_neigh_best;
			InsertTable(shared_pi_neigh_best, best_job_j, best_job_v);

			if (shared_value_best > shared_value_neigh_best)
			{
				shared_value_best = shared_value_neigh_best;
				for (i = 0; i < n; i++)
				{
					shared_pi_best[i] = shared_pi_neigh_best[i];
				}
			}

			//------ add to tabu list -------//
			//- list can be shared later on -//
			listTabu[0 * listN + listIdx] = best_job_j;
			listTabu[1 * listN + listIdx] = best_job_v;

			listIdx = (listIdx + 1) % listN;
			//------------------------------//
			//firstNeigh = 1;
		
			//wait for all blocks to finish copying from global to shared
			///*
			// data copied from global to shared
			//my_lock();
			g_blocksActive[block_index] = 0;
			g_blocksReady[block_index] = 1;
			//my_unlock();


			if (gindex == 0)
			{
				//--- synchronize threads ---//
				all_ready = 0; //wait all to complete
				//while (all_ready == 0)
				for (r = 0; r < 50; r++)
				{
					all_ready = 1;
					for (i = 0; i < blocksN; i++)
					{
						all_ready *= g_blocksReady[i];
					}

					if (all_ready == 1)
					{
						break;
					}
				}

				for (i = 0; i < blocksN; i++)
				{
					atomicExch(&g_blocksActive[i], 1);
				}
			}

			//other local main threads (lindex == 0) wait for the global main thread gindex
			for (r = 0; r < 100; r++)
			{
				if (g_blocksActive[block_index] == 1)
				{
					break;
				}
			}


			//g_blocksActive[block_index] = 0;

			//
			//my_lock();
			//g_blocksCompleted[block_index] = 0;
			//g_blocksReady[block_index] = 0;
			//g_value[block_index] = BIG_NUMBER;
			//my_unlock();
			//*/
			//----------------------------//
		}
		//shared_neigh_best mozna w shared value pamietac, dla kazdego w osobnej kolumnie

		//firstNeigh = 1;

		// wait for all threads to reach the barrier

		//value_shared[lindex] = BIG_NUMBER;

		//each block has to wait for each at least one thread
		//inne nie musza czekac i moga zaczac odlicznie
		// block 0 wejdzie w czekanie, a wtedy block 1 ustawi 
		// to nie musi dzialac !!! naprawic trzeba
		// nie bedzie takiej sytuacji, gdyz kolejna iteracja idzie, jak skonczyl to bedzie czekal i przepisywal z shared_pi
		// poza tym nie wszystkie watki beda wykonywaly sie tyle samo iteracji, np. 6666|5555 block 0 trwa, ale blok 1 nie skonczy sie i nie powoli kontunuowac blokowi 0



		//wait for all blocks to finish copying from global to shared //


		//<-------------------- global barrier -------//


		// wait for all threads to reach the barrier (to next iteration) to copy global to shared
		__syncthreads();
		//<-------------------- barrier -------//


		//--- update local from shared values ---//
		//value_neigh_best = shared_value_neigh_best;	//nie wiadomo czy potrzeba tego value_neigh_best w kolejnej iteracji, skoro jest ustawiana BIG_NUMBER
		//---------------------------------------//

		// wait for all threads to reach the barrier (to next iteration)	
		//__syncthreads();	//tutaj juz nie jest potrzebne
		//<-------------------- barrier -------//
	}

	//useful functions
	//atomicExch(&g_mutex, 1);
	//atomicCAS(&g_mutex, 0, 1);	//compare and swap
	
	//__syncthreads();
	// bezpiecznie jezeli jeden watek bedzie zawsze wpisywal, ale watek 0 zawsze sie wykona 666665|555555 - podzial na bloki to zawsze pierwszy i tak wykona sie nie mniej niz inne
	// trzeba poczekac na wszystkie bloki i watki
	//--------- copy best solution to host -----------//	
	if (gindex == 0)
	{
		for (j = 0; j < n; j++)
		{
			d_pi[j] = shared_pi_best[j];
		}		
	}

	if ((block_index == 1))//&&(lindex == 0))
	{
		d_err_note[0] = 8+10*g_blocksCompleted[0] + g_blocksCompleted[1]*100+ g_blocksCompleted[2]*1000+ g_blocksCompleted[3]*10000 +  all_completed*100000+ 8*1000000;

		//d_err_note[0] = 8 + 10 * b0 + b1 * 100 + b2 * 1000 + b3 * 10000 + all_completed * 100000 + 8 * 1000000;

		//d_err_note[0] = 8 + 10 * b0 + b1 * 100 + 8 * 1000;
	
	//index_start_move_k
	}

}




//------------ Kernel TS BaT -------------//
// blocks and threads 
//
// work if blocks = 8, 
// insert by swap - multithread = number of jobs
// shared tabu list
// shared pi_neigh_best - wolniejsze niz w TS2 bez shared tylko kazdy watek ma swoj pi_neigh_best (1024 - 51s vs 53s)
// shared pi_best
//
// parallel moze dawac inny wynik niz jednowatkowy,
// gdyz to samo kryterium moze byc dla roznych j,v, wowczas inne j,v trafiaja na liste tabu (w innej kolejnosci) 
////
// mozna zrobic, zeby tylko lindex == 0 sprawdzal najlepsze wstawienie i przebudowal shared_pi_neigh_best
// trzeba wtedy dodac shared_best_j, shared_best_v
//
// ??? czy n wlicza m?, czy musze robic (n+m)*(n+m), ale chyba n wlicza m
// niebezpieczna sytuacja do sprawdzanie 111|100, wtedy block drugi jak watki wyladaja?, albo 111|000 czy wtedy block 2 powstanie?
//??? a jak wszystkie ruchy danego watku na liscie tabu? czy cos bedzie mial lepszego niz wczesniej?, czy zamazywac duza wartoscia 10000000
__global__ void KernelTS_BaT(int *d_blocksN, int *d_threads_per_block, int *d_n, int *d_m, int *d_pi, float *d_aj, float *d_alphaj, float *d_parametersTS, unsigned int seed, float *d_err_note)
{
	int i, l, r;
	int jobi, jobv1, jobv2, best_job_j, best_job_v, j_best, v_best;
	int iter, iterN;
	int listN, isInTabu;
	int index_best;
	int index_block_best;

	int firstNeigh;

	//int listTabu[2 * MAX_TABU_LIST];		//[1,1; 2,2; ...]

	float CP[MAX_MACHINE_SIZE];				//for criterion value

	float value_C, value_neigh_best;
	int pi[MAX_PROBLEM_SIZE];				// temporary pi for analysing neighbourhood

	__shared__ int shared_listIdx;					//jezeli rozne watki moga zmieniac, ale wiadomo, ze zwiekszy sie o jeden tez po kazdej iteracji, dla kazdego bloku, zatem moze byc jedno shared na block tylko

	__shared__ int listTabu[2 * MAX_TABU_LIST];		//[1,1; 2,2; ...]

	__shared__ int shared_pi_neigh_best[MAX_PROBLEM_SIZE];	//the best for the neighourhood
	__shared__ int shared_pi_best[MAX_PROBLEM_SIZE];			//overall the best - result

	__shared__ float value_shared[MAX_THREADS];	//the best solution value for each thread

	__shared__ float aj_shared[MAX_PROBLEM_SIZE];			//much faster than local float aj_shared[MAX_PROBLEM_SIZE];
	__shared__ float alphaj_shared[MAX_PROBLEM_SIZE];

	__shared__ float shared_value_neigh_best;
	__shared__ float shared_value_best;

	//float aj_shared[MAX_PROBLEM_SIZE];
	//float alphaj_shared[MAX_PROBLEM_SIZE];

	int threads_per_blockN = *d_threads_per_block;
	int blocksN = *d_blocksN;
	int threadsN = (*d_threads_per_block)*(*d_blocksN);
	int n = *d_n;
	int m = *d_m;

	int j, v, k;
	int N;				//number of moves per neighbourhood
	int N_k;					// number of moves per this thread
	int rem_N;					//reminder
	int div_N;					//division
	int index_start_move_k;		//the index of a start move of this thread, e.g., (0,0) is 0, (0,1) is 1, etc. (0,n-1) is n-1, (1,0) is n
	int job_j;
	int job_v;


	int all_completed = 0;
	int all_ready = 0;
	int gindex;
	int block_index = blockIdx.x;

	int wait_couter = 0;

	//gindex = threadIdx.x + blockIdx.x*threads_per_blockN;

	gindex = blockIdx.x * blockDim.x + threadIdx.x;

	int lindex = threadIdx.x;	//local index for each block
								//gindex = threadIdx.x;
								//threadsN = threadsN;
								//threadsN = blockDim.x * gridDim.x;	//numeber of threads


								//for random values
								// if random values required 
								//curandState_t state;
								//curand_init(seed + 17 * lindex, 0, 0, &state);	//it should be a time(NULL)

	listN = d_parametersTS[0];
	iterN = d_parametersTS[1];

	//initialize shared parameters

	if (lindex == 0)
	{
		//for block synchronization

		g_blocksCompleted[block_index] = 0;

		g_value[block_index] = BIG_NUMBER;
		g_best_j[block_index] = 0;
		g_best_v[block_index] = 0;

		//problem parameters
		for (j = 0; j<n; j++)
		{
			aj_shared[j] = d_aj[j];
			alphaj_shared[j] = d_alphaj[j];
		}
		
		//tabu list
		shared_listIdx = 0;
		for (i = 0; i<listN * 2; i++)
		{
			listTabu[i] = 0;
		}

		//solutions
		for (j = 0; j < n; j++)
		{
			shared_pi_best[j] = d_pi[j];
			shared_pi_neigh_best[j] = shared_pi_best[j];
		}

		//caclulate the best value
		shared_value_best = Criterion2(&n, &m, shared_pi_best, aj_shared, alphaj_shared, CP);
		shared_value_neigh_best = shared_value_best;

		//firstNeigh = 1;
	}

	firstNeigh = 1;

	// wait for all threads to reach the barrier
	__syncthreads();

	value_neigh_best = shared_value_neigh_best;

	N = n*n;
	rem_N = N % threadsN;
	div_N = (int)(N / threadsN);
	N_k = div_N + (gindex < rem_N);
	index_start_move_k = gindex*div_N + rem_N*(gindex >= rem_N) + gindex*(gindex < rem_N); //gindex >= rem_N->rem_n otherwise gindex


																						   // okreslone tylko liczy dany watek
	for (iter = 0; iter < iterN; iter++)
	{		
		//<-------------------- barrier -------//
		if (lindex == 0)
		{
			g_blocksCompleted[block_index] = 0;
			g_value[block_index] = BIG_NUMBER;
		}
		__syncthreads();

		best_job_j = 0;	//means that nothing has changed in reference to the previouse solution (po iteracji 0,0 oznacza, ze bez zmian)
		best_job_v = 0;

		//firstNeigh = 1;

		firstNeigh = 0;	//moze nie byc potrzebny

		value_neigh_best = BIG_NUMBER;

		//pi = pi_neigh_best
		for (l = 0; l < n; l++)
		{
			pi[l] = shared_pi_neigh_best[l];
		}

		j = (int)(index_start_move_k / n);
		v = index_start_move_k % n;
		job_j = -1;


		for (k = 0; k < N_k; k++)
		{
			//------- obtain move --------//
			if (job_j == j)
			{
				SwapTable(pi, v - 1, v);	//here v is > 0 - always
				job_j = j;
				job_v = v;
			}
			else
			{
				//pi = pi_neigh_best
				for (l = 0; l < n; l++)
				{
					pi[l] = shared_pi_neigh_best[l];
				}

				InsertTable(pi, j, v);
				job_j = j;
				job_v = v;
			}
			//----------------------------//

			if (j != v)
			{
				value_C = Criterion2(&n, &m, pi, aj_shared, alphaj_shared, CP);

				//---------- check Tabu List -------------//
				if ((job_j != job_v) && (firstNeigh == 1 || value_C < value_neigh_best))
				{
					isInTabu = 0;
					for (l = 0; l < listN; l++)
					{
						if ((job_j == listTabu[0 * listN + l]) && (job_v == listTabu[1 * listN + l])) //[0][l], [1][l]
						{
							isInTabu = 1;
							break;
						}
					}

					if (!isInTabu)
					{
						value_neigh_best = value_C;
						firstNeigh = 0;
						best_job_j = job_j;
						best_job_v = job_v;
					}
				}
				//----------------------------------------//
			}

			if ((j == 13) && (v == 9) && (iter == 8))
			{
				d_err_note[0] = shared_listIdx;//listTabu[1 * listN + 0];
			}

			//---- update j and v ---//
			v++;
			if (v >= n)
			{
				v = 0;
				j++;
			}
			//-----------------------//

		}

		value_shared[lindex] = value_neigh_best;
		// wait for all threads to reach the barrier		
		__syncthreads();
		//<-------------------- barrier -------//

		//choose the best insert
		index_best = lindex;		//if value_i = value_k, then solution i is chosen instead of solution k (i<k)
		for (i = 0; i < threads_per_blockN; i++)	//to the number of threads, check if the currecn thread is the best
		{
			if ((value_shared[i] <  value_neigh_best) || (value_shared[i] == value_neigh_best) && (i < lindex)) //this thread (lindex) is not the best
			{
				index_best = -1;
				break;
			}
		}

		//only the best update pi, criterion, tabuList
		if (lindex == index_best)
		{
			//my_lock();
			g_value[block_index] = value_neigh_best;
			g_best_j[block_index] = best_job_j;
			g_best_v[block_index] = best_job_v;
			g_blocksCompleted[block_index] = 1;
			//my_unlock();

			//2 blocks, 100000 sufficient, 4 block somethimes different than 1024 threads
			for (r = 0; r < 1000000*blocksN; r++)
			{
				//my_lock();
				all_completed = 1;
				for (i = 0; i < blocksN; i++)
				{
					all_completed *= g_blocksCompleted[i];
				}
				//my_unlock();

				if (all_completed == 1)
				{
					break;
				}
			}

			index_block_best = 0;
			value_neigh_best = g_value[index_block_best];

			for (i = 1; i < blocksN; i++)	//to the number of threads, check if the currecn thread is the best
			{
				if (g_value[i] < value_neigh_best) //this thread (lindex) is not the best
				{
					index_block_best = i;
				}
			}
			value_neigh_best = g_value[index_block_best];
			best_job_j = g_best_j[index_block_best];
			best_job_v = g_best_v[index_block_best];

			shared_value_neigh_best = value_neigh_best;
			InsertTable(shared_pi_neigh_best, best_job_j, best_job_v);

			if (shared_value_best > shared_value_neigh_best)
			{
				shared_value_best = shared_value_neigh_best;
				for (l = 0;l < n; l++)
				{
					shared_pi_best[l] = shared_pi_neigh_best[l];
				}
			}

			//------ add to tabu list -------//
			//- list can be shared later on -//
			listTabu[0 * listN + shared_listIdx] = best_job_j;
			listTabu[1 * listN + shared_listIdx] = best_job_v;

			shared_listIdx = (shared_listIdx + 1) % listN;
			//------------------------------//
			//firstNeigh = 1;
		}
		//shared_neigh_best mozna w shared value pamietac, dla kazdego w osobnej kolumnie

		//__syncthreads();

		//if (lindex == 0)
		//{
		//	g_blocksCompleted[block_index] = 2;
		//}

		// wait for all threads to reach the barrier (to next iteration) to copy global to shared
		__syncthreads();

		/*
		threadsCompleted[lindex] = 1;
		//--- synchronize threads ---//		
		all_completed = 0; //wait all to complete

		while (all_completed == 0)
		{
			all_completed = 1;
			for (i = 0; i < threadsN; i++)
			{
				all_completed *= threadsCompleted[i];
			}
		}
		threadsCompleted[lindex] = 0;
		*/

		//<-------------------- barrier -------//

	}

	//useful functions
	//atomicExch(&g_mutex, 1);
	//atomicCAS(&g_mutex, 0, 1);	//compare and swap

	//__syncthreads();
	// bezpiecznie jezeli jeden watek bedzie zawsze wpisywal, ale watek 0 zawsze sie wykona 666665|555555 - podzial na bloki to zawsze pierwszy i tak wykona sie nie mniej niz inne
	// trzeba poczekac na wszystkie bloki i watki
	//--------- copy best solution to host -----------//	
	if (gindex == 0)
	{
		for (j = 0; j < n; j++)
		{
			d_pi[j] = shared_pi_best[j];
		}
	}

	if ((block_index == 0)&&(lindex == 0))
	{
		//d_err_note[0] = value_neigh_best;

		//d_err_note[0] = 8 + 10 * g_blocksCompleted[0] + g_blocksCompleted[1] * 100 + g_blocksCompleted[2] * 1000 + g_blocksCompleted[3] * 10000 + all_completed * 100000 + 8 * 1000000;
		//d_err_note[0] = g_blocksCompleted[0];

		//d_err_note[0] = 8 + 10 * b0 + b1 * 100 + b2 * 1000 + b3 * 10000 + all_completed * 100000 + 8 * 1000000;

		//d_err_note[0] = 8 + 10 * b0 + b1 * 100 + 8 * 1000;

		//index_start_move_k
	}

}



//------------ GPU AlgTS_BaT -------------//
// one block TS
void GPU_AlgTS_BaT(problem *p, solution &result, const solution sUB, int listN, int iterN, int threadsN, int blocksN)	//threads per block, blocks
{
	int i, j, v;

	int *pi, n, m;
	float *aj, *alphaj;
	float *parametersTS;

	int *d_pi, *d_n, *d_m;
	float *d_aj, *d_alphaj;
	float *d_parametersTS;

	int *d_threadsN;
	int *d_blocksN;

	float *d_err_note;

	clock_t t1, t2;


	//??????!!!!!!!!!!!!!!!!!
	//blocksN = 4;
	//threadsN = 1;
	//??????!!!!!!!!!!!!!!!!!


	result = sUB;

	n = p->n;
	m = p->m;

	aj = new float[n];
	alphaj = new float[n];

	//--------------------//

	//threadsN = MAX_THREADS;
	if (threadsN <= 0)
	{
		threadsN = MAX_THREADS;
	}
	threadsN = min(threadsN, MAX_THREADS);
	threadsN = min(threadsN, n*n);		// cannot be greater than neighourhood size, N= n*n

										//--------------------//

										//threadsN = MAX_THREADS;
	if (blocksN <= 0)
	{
		blocksN = MAX_BLOCKS;
	}
	blocksN = min(blocksN, MAX_BLOCKS);
	blocksN = min(blocksN, (int)((n*n)/threadsN));		// blocksN*threadsN cannot be greater than neighourhood size, N= n*n
	blocksN = max(blocksN, 1);

										//--------------------//


	for (j = 0; j < n; j++)
	{
		aj[j] = p->jobs[0][j].aj;
		alphaj[j] = p->jobs[0][j].alphaj;
	}

	parametersTS = new float[2];
	parametersTS[0] = listN;
	parametersTS[1] = iterN;


	cout << "===== TS GPU BaT =====" << endl;
	cout << "blocksN " << blocksN << endl;
	cout << "threadsN " << threadsN << endl;
	cout << "listN " << parametersTS[0] << endl;
	cout << "iterN " << parametersTS[1] << endl;


	pi = new int[n];

	for (j = 0; j < n; j++)
	{
		pi[j] = result.pi[j];
	}


	t1 = clock();

	cudaMalloc(&d_blocksN, sizeof(int));
	cudaMalloc(&d_threadsN, sizeof(int));
	cudaMalloc(&d_pi, n * sizeof(int));
	cudaMalloc(&d_aj, n * sizeof(float));
	cudaMalloc(&d_alphaj, n * sizeof(float));
	cudaMalloc(&d_n, sizeof(int));
	cudaMalloc(&d_m, sizeof(int));
	cudaMalloc(&d_parametersTS, 2 * sizeof(float));
	cudaMalloc(&d_err_note, sizeof(float));
		

	t2 = clock();

	cout << "cuda alloc " << t2 - t1 << endl;

	t1 = clock();

	float err_note = -1;

	cudaMemcpy(d_err_note, &err_note, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_blocksN, &blocksN, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_threadsN, &threadsN, sizeof(int), cudaMemcpyHostToDevice);	
	cudaMemcpy(d_pi, pi, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_aj, aj, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_alphaj, alphaj, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_m, &m, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_parametersTS, parametersTS, 2 * sizeof(float), cudaMemcpyHostToDevice);

	t2 = clock();

	cout << "cuda mem copy " << t2 - t1 << endl;

	//system("PAUSE");

	t1 = clock();


	// !!!??? zobaczyc jak jest w innych programach ustalane threads/blocks, etc.

	

	KernelTS_BaT <<< blocksN, threadsN >>> (d_blocksN, d_threadsN, d_n, d_m, d_pi, d_aj, d_alphaj, d_parametersTS, time(NULL), d_err_note);

	t2 = clock();

	cout << "cuda run " << t2 - t1 << endl;

	t1 = clock();

	cudaMemcpy(pi, d_pi, n * sizeof(int), cudaMemcpyDeviceToHost);

	//for tests only
	cudaMemcpy(&err_note, d_err_note, sizeof(float), cudaMemcpyDeviceToHost);
	

	t2 = clock();
	cout << "err --------------- note : " << err_note << endl;
	cout << "cuda mem to host " << t2 - t1 << endl;

	//cudaMemcpy(pi, d_pi, n * sizeof(int), cudaMemcpyDeviceToHost);

	for (j = 0; j < n; j++)
	{
		//cout << pi[j] << " ";
		result.pi.Set(j, pi[j]);
	}
	//cout << endl;

	//cout << "pi " << endl;
	//cout << result.pi << endl;

	result.value = p->Criterion(result.pi);

	t1 = clock();

	cudaFree(d_blocksN);
	cudaFree(d_threadsN);
	cudaFree(d_pi);
	cudaFree(d_aj);
	cudaFree(d_alphaj);
	cudaFree(d_n);
	cudaFree(d_m);
	cudaFree(d_parametersTS);
	cudaFree(d_err_note);


	t2 = clock();

	cout << "cuda free " << t2 - t1 << endl << endl;
	

	delete[]parametersTS;
	delete[]aj;
	delete[]alphaj;
	delete[]pi;
}







