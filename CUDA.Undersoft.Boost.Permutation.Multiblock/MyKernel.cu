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
__global__ void KernelTS_FU_BaT(int *d_blocksN, int *d_threads_per_block, int *d_n, int *d_m, int *d_pi, float *d_aj, float *d_alphaj, int *d_parametersTS, int *d_listTabu, float *d_best_value_neigh, int *d_best_job_j, int *d_best_job_v, unsigned int seed, float *d_err_note)
{
	int i, l, r;
	int best_job_j, best_job_v, j_best, v_best;
	int iterN;
	int listN, isInTabu;
	int index_best;
	int index_block_best;

	int firstNeigh;

	//int listTabu[2 * MAX_TABU_LIST];		//[1,1; 2,2; ...]

	float CP[MAX_MACHINE_SIZE];				//for criterion value

	float value_C, value_neigh_best;
	int pi[MAX_PROBLEM_SIZE];				// temporary pi for analysing neighbourhood

	__shared__ int listTabu[2 * MAX_TABU_LIST];		//[1,1; 2,2; ...]

	__shared__ int shared_pi_neigh_best[MAX_PROBLEM_SIZE];	//the best for the neighourhood

	__shared__ float shared_value[MAX_THREADS];	//the best solution value for each thread
	//__shared__ int shared_best_j[MAX_THREADS];
	//__shared__ int shared_best_v[MAX_THREADS];

	__shared__ float aj_shared[MAX_PROBLEM_SIZE];			//much faster than local float aj_shared[MAX_PROBLEM_SIZE];
	__shared__ float alphaj_shared[MAX_PROBLEM_SIZE];

	//float aj_shared[MAX_PROBLEM_SIZE];
	//float alphaj_shared[MAX_PROBLEM_SIZE];

	int threads_per_blockN;
	int blocksN;
	int threadsN;
	int n;
	int m;

	int j, v, k;
	int N;				//number of moves per neighbourhood
	int N_k;					// number of moves per this thread
	int rem_N;					//reminder
	int div_N;					//division
	int index_start_move_k;		//the index of a start move of this thread, e.g., (0,0) is 0, (0,1) is 1, etc. (0,n-1) is n-1, (1,0) is n
	int job_j;
	int job_v;

	int gindex;
	int lindex;	//local index for each block				
	int block_index;

	threads_per_blockN = *d_threads_per_block;
	blocksN = *d_blocksN;
	threadsN = (*d_threads_per_block)*(*d_blocksN);

	n = *d_n;
	m = *d_m;

	lindex = threadIdx.x;	//local index for each block																								
	block_index = blockIdx.x;
	gindex = threadIdx.x + blockIdx.x*threads_per_blockN;

	//gindex = blockIdx.x * blockDim.x + threadIdx.x;

	
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
		for (i = 0; i<listN * 2; i++)
		{
			listTabu[i] = d_listTabu[i];
		}

		//solutions
		for (j = 0; j < n; j++)
		{
			shared_pi_neigh_best[j] = d_pi[j];
		}
	}

	firstNeigh = 1;

	// wait for all threads to reach the barrier
	__syncthreads();
	
	N = n*n;
	rem_N = N % threadsN;
	div_N = (int)(N / threadsN);
	N_k = div_N + (gindex < rem_N);
	index_start_move_k = gindex*div_N + rem_N*(gindex >= rem_N) + gindex*(gindex < rem_N); //gindex >= rem_N->rem_n otherwise gindex


																						   // okreslone tylko liczy dany watek
	//for (iter = 0; iter < iterN; iter++)
	{
		best_job_j = 0;	//means that nothing has changed in reference to the previouse solution (po iteracji 0,0 oznacza, ze bez zmian)
		best_job_v = 0;

		firstNeigh = 0;	//moze nie byc potrzebny

		value_neigh_best = BIG_NUMBER;

		//pi = pi_neigh_best
		//for (l = 0; l < n; l++)
		//{
		//	pi[l] = shared_pi_neigh_best[l];
		//}


		//-------- search neighbourhood ----------//
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
				//at first time always copied for each thread
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

		/*
		shared_value[lindex] = value_neigh_best;
		shared_best_j[lindex] = best_job_j;
		shared_best_v[lindex] = best_job_v;

		// wait for all threads to reach the barrier		
		__syncthreads();
		//<-------------------- barrier -------//


		if (lindex == 0)
		{
			index_best = 0;
			value_neigh_best = shared_value[index_best];

			for (i = 1; i < threads_per_blockN; i++)	//to the number of threads, check if the currecn thread is the best
			{
				if (shared_value[i] < value_neigh_best)
				{
					value_neigh_best = shared_value[i];
					index_best = i;
				}
			}

			d_best_value_neigh[block_index] = shared_value[index_best];
			d_best_job_j[block_index] = shared_best_j[index_best];
			d_best_job_v[block_index] = shared_best_v[index_best];		
		}

		*/

		shared_value[lindex] = value_neigh_best;

		// wait for all threads to reach the barrier		
		__syncthreads();
		//<-------------------- barrier -------//


		//using index best - requires less memory per block than using lindex ==0 (i.e., job_j, job_v for each thred is not necessary, which give 2x4KB less memory

		index_best = lindex;		//if value_i = value_k, then solution i is chosen instead of solution k (i<k)
		for (i = 0; i < threads_per_blockN; i++)	//to the number of threads, check if the currecn thread is the best
		{
			if ((shared_value[i] <  value_neigh_best) || (shared_value[i] == value_neigh_best) && (i < lindex)) //this thread (lindex) is not the best
			{
				index_best = -1;
				break;
			}
		}

		if (lindex == index_best)
		{
			d_best_value_neigh[block_index] = value_neigh_best;
			d_best_job_j[block_index] = best_job_j;
			d_best_job_v[block_index] = best_job_v;
		}

	}
	__syncthreads();
}



//------------ GPU AlgTS_BaT -------------//
// Full Utilization of Blocks and Threads
// one block TS
void GPU_AlgTS_FU_BaT(problem *p, solution &result, const solution sUB, int listN, int iterN, int threadsN, int blocksN)	//threads per block, blocks
{
	int i, j, v;

	int *d_pi, *d_n, *d_m;
	float *d_aj, *d_alphaj;
	int *d_parametersTS;

	int *d_listTabu;
	float *d_best_value_neigh;
	int *d_best_job_j;
	int *d_best_job_v;


	int *d_threadsN;
	int *d_blocksN;

	float *d_err_note;

	clock_t t1, t2;


	//??????!!!!!!!!!!!!!!!!!
	//blocksN = 4;
	//threadsN = 1;
	//??????!!!!!!!!!!!!!!!!!


	result = sUB;

	int n, m;
	n = p->n;
	m = p->m;
	
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
	blocksN = min(blocksN, (int)((n*n) / threadsN));		// blocksN*threadsN cannot be greater than neighourhood size, N= n*n
	blocksN = max(blocksN, 1);
	//--------------------//


	//------------- local variables ------//
	int *listTabu;
	listTabu = new int[2 * listN];
	for (i = 0; i < 2 * listN; i++)
	{
		listTabu[i] = 0;
	}
	int listIdx;
	listIdx = 0;

	float *best_value_neigh;
	int *best_job_j;
	int *best_job_v;
	best_value_neigh = new float[blocksN];
	best_job_j = new int[blocksN];
	best_job_v = new int[blocksN];	
	for (i = 0; i < blocksN; i++)
	{
		best_value_neigh[i] = 0;
		best_job_j[i] = 0;
		best_job_v[i] = 0;
	}
	
	float *aj, *alphaj;	
	aj = new float[n];
	alphaj = new float[n];
	for (j = 0; j < n; j++)
	{
		aj[j] = p->jobs[0][j].aj;
		alphaj[j] = p->jobs[0][j].alphaj;		
	}

	int *parametersTS;
	parametersTS = new int[2];
	parametersTS[0] = listN;
	parametersTS[1] = iterN;

	int *pi;
	int *pi_best;	//only locally important
	pi = new int[n];
	pi_best = new int[n];
	for (j = 0; j < n; j++)
	{
		pi[j] = result.pi[j];
		pi_best[j] = result.pi[j];
	}
	float value_best;
	value_best = result.value;
	
	cout << "===== TS GPU FU BaT =====" << endl;
	cout << "blocksN " << blocksN << endl;
	cout << "threadsN " << threadsN << endl;
	cout << "listN " << parametersTS[0] << endl;
	cout << "iterN " << parametersTS[1] << endl;
	cout << "value " << value_best << endl;


	t1 = clock();

	cudaMalloc(&d_blocksN, sizeof(int));
	cudaMalloc(&d_threadsN, sizeof(int));
	cudaMalloc(&d_pi, n * sizeof(int));
	cudaMalloc(&d_aj, n * sizeof(float));
	cudaMalloc(&d_alphaj, n * sizeof(float));
	cudaMalloc(&d_n, sizeof(int));
	cudaMalloc(&d_m, sizeof(int));
	cudaMalloc(&d_parametersTS, 2 * sizeof(int));
	cudaMalloc(&d_err_note, sizeof(float));

	cudaMalloc(&d_listTabu, 2*listN * sizeof(int));
	cudaMalloc(&d_best_value_neigh, blocksN * sizeof(float));
	cudaMalloc(&d_best_job_j, blocksN * sizeof(int));
	cudaMalloc(&d_best_job_v, blocksN * sizeof(int));


	t2 = clock();

	cout << "cuda alloc " << t2 - t1 << endl;

	t1 = clock();

	float err_note = -1;

	
	cudaMemcpy(d_blocksN, &blocksN, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_threadsN, &threadsN, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pi, pi, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_aj, aj, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_alphaj, alphaj, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_m, &m, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_parametersTS, parametersTS, 2 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_err_note, &err_note, sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(d_listTabu, listTabu, 2*listN * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_best_value_neigh, best_value_neigh, blocksN * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_best_job_j, best_job_j, blocksN * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_best_job_v, best_job_v, blocksN * sizeof(int), cudaMemcpyHostToDevice);

	t2 = clock();

	cout << "cuda mem copy " << t2 - t1 << endl;

	//system("PAUSE");

	t1 = clock();


	// !!!??? zobaczyc jak jest w innych programach ustalane threads/blocks, etc.


	
	//(int *d_blocksN, int *d_threads_per_block, int *d_n, int *d_m, int *d_pi, float *d_aj, float *d_alphaj, int *d_parametersTS, int *d_listTabu, float *d_best_value_neigh, int *d_best_job_j, int *d_best_job_v, unsigned int seed, int *d_err_note)

	int iter;
	int job_j;
	int job_v;
	float value_neigh;
	int best_block_index;
	//permutation tmp_pi(n);

	for (iter = 0; iter < iterN; iter++)
	{
		//for (j = 0; j < n; j++)
		//{		
		//	tmp_pi.Set(j, pi[j]);
		//}
		//cout << " pi in " << tmp_pi << endl;

		//cudaDeviceSynchronize();

		KernelTS_FU_BaT <<< blocksN, threadsN >> > (d_blocksN, d_threadsN, d_n, d_m, d_pi, d_aj, d_alphaj, d_parametersTS, d_listTabu, d_best_value_neigh, d_best_job_j, d_best_job_v, time(NULL), d_err_note);
	
		cudaDeviceSynchronize();

		//Device -> Host
		cudaMemcpy(best_value_neigh, d_best_value_neigh, blocksN * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(best_job_j, d_best_job_j, blocksN * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(best_job_v, d_best_job_v, blocksN * sizeof(int), cudaMemcpyDeviceToHost);

		//cout << endl;

		best_block_index = 0;
		value_neigh = best_value_neigh[best_block_index];
		//cout << "C : " << best_value_neigh[0] << " (" << best_job_j[0] << " , " << best_job_v[0] << ")   |  ";
		for (i = 1; i < blocksN; i++)
		{
			//cout << "C : "<< best_value_neigh[i] << " ("<<best_job_j[i]<<" , "<<best_job_v[i]<<")   |  ";
			if (best_value_neigh[i] < value_neigh)
			{
				best_block_index = i;
				value_neigh = best_value_neigh[i];
			}
		}
		//cout << endl;

		job_j = best_job_j[best_block_index];
		job_v = best_job_v[best_block_index];

		HostInsertTable(pi, job_j, job_v);

		//cout << " pi out " << tmp_pi << endl;

		//cudaMemcpy(&err_note, d_err_note, sizeof(float), cudaMemcpyDeviceToHost);
		//cout << "err --------------- note : " << err_note << endl;

		//system("PAUSE");


		//---------- update the best ---------//
		
		if (value_best > value_neigh)
		{
			value_best = value_neigh;
			for (i = 0; i < n; i++)
			{
				pi_best[i] = pi[i];
			}
		}
		//------------------------------------//

		//------ add to tabu list -------//
		listTabu[0 * listN + listIdx] = job_j;
		listTabu[1 * listN + listIdx] = job_v;
		listIdx = (listIdx + 1) % listN;


		//for (i = 0; i < listN; i++)
		//{
		//	cout <<"("<<listTabu[0 * listN + i] << " " << listTabu[1 * listN + i] << ")  ";
		//}
		//cout << endl;
		//cout << "TL : " << job_j << " " << job_v << endl;
		//-------------------------------//
			
		cudaMemcpy(d_pi, pi, n * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_listTabu, listTabu, 2*listN * sizeof(int), cudaMemcpyHostToDevice);
		//cudaMemcpy(d_err_note, &err_note, sizeof(float), cudaMemcpyHostToDevice);
	}
	t2 = clock();

	cout << "cuda run " << t2 - t1 << endl;

	t1 = clock();

	//cudaMemcpy(pi, d_pi, n * sizeof(int), cudaMemcpyDeviceToHost);

	//for tests only
	cudaMemcpy(&err_note, d_err_note, sizeof(float), cudaMemcpyDeviceToHost);


	t2 = clock();
	//cout << "err --------------- note : " << err_note << endl;
	//cout << "cuda mem to host " << t2 - t1 << endl;

	//cudaMemcpy(pi, d_pi, n * sizeof(int), cudaMemcpyDeviceToHost);

	for (j = 0; j < n; j++)
	{
		//cout << pi[j] << " ";
		result.pi.Set(j, pi_best[j]);
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

	cudaFree(d_listTabu);
	cudaFree(d_best_value_neigh);
	cudaFree(d_best_job_j);
	cudaFree(d_best_job_v);

	t2 = clock();

	cout << "cuda free " << t2 - t1 << endl << endl;


	delete[]parametersTS;
	delete[]aj;
	delete[]alphaj;
	delete[]pi;

	delete[]listTabu;
	delete[]best_value_neigh;
	delete[]best_job_j;
	delete[]best_job_v;

	delete[]pi_best;
}




//------------ Kernel TS BaT -------------//
// Full Utilization of Blocks and Threads
// Revised notation and names of variables 
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
__global__ void KernelTS_FFU_BaT(int *d_blocksN, int *d_threads_per_block, int *d_n, int *d_m, int *d_pi, float *d_aj, float *d_alphaj, int *d_parametersTS, int *d_listTabu, float *d_best_value_neigh, int *d_best_job_j, int *d_best_job_v, unsigned int seed, float *d_err_note)
{	
	float CP[MAX_MACHINE_SIZE];				//for criterion value

	float value_C;						//the temporary crterion value	
	float value_neigh_best;				//the best found value for the given thread
	int best_job_j, best_job_v;

	int pi[MAX_PROBLEM_SIZE];				// temporary pi for analysing neighbourhood

	__shared__ float shared_aj[MAX_PROBLEM_SIZE];			//much faster than local float aj_shared[MAX_PROBLEM_SIZE];
	__shared__ float shared_alphaj[MAX_PROBLEM_SIZE];

	int isInTabu;
	int tabuListSize;
	__shared__ int shared_tabuList[2 * MAX_TABU_LIST];		//[1,1; 2,2; ...]

	__shared__ int shared_pi_neigh_best[MAX_PROBLEM_SIZE];	//pi_neigh the best for the block
	__shared__ float shared_value_neigh_best[MAX_THREADS];	//the best solution value for each thread


	int threads_per_block;
	int blocks_per_grid;
	int number_all_threads;
	int n;
	int m;

	int j, v, k, i;
	int neighborhoodSizeN;		//number of moves per neighbourhood
	int N_k;					// number of moves per this thread
	int rem_N;					//reminder
	int div_N;					//division
	int index_start_move_k;		//the index of a start move of this thread, e.g., (0,0) is 0, (0,1) is 1, etc. (0,n-1) is n-1, (1,0) is n
	int job_j;
	int job_v;

	int gindex;
	int lindex;	//local index for each block				
	int block_index;
	int index_thread_best;

	n = *d_n;
	m = *d_m;

	threads_per_block = *d_threads_per_block;
	blocks_per_grid = *d_blocksN;
	number_all_threads = (*d_threads_per_block)*(*d_blocksN);

	lindex = threadIdx.x;	//local index for each block																								
	block_index = blockIdx.x;
	gindex = threadIdx.x + blockIdx.x*threads_per_block;

	//gindex = blockIdx.x * blockDim.x + threadIdx.x;


	tabuListSize = d_parametersTS[0];
	
	//initialize shared parameters
	if (lindex == 0)
	{
		//problem parameters
		for (j = 0; j<n; j++)
		{
			shared_aj[j] = d_aj[j];
			shared_alphaj[j] = d_alphaj[j];
		}

		//tabu list
		for (j = 0; j<tabuListSize * 2; j++)
		{
			shared_tabuList[j] = d_listTabu[j];
		}

		//solutions
		for (j = 0; j < n; j++)
		{
			shared_pi_neigh_best[j] = d_pi[j];
		}
	}

	// wait for all threads to reach the barrier
	__syncthreads();

	neighborhoodSizeN = n*n;
	rem_N = neighborhoodSizeN % number_all_threads;
	div_N = (int)(neighborhoodSizeN / number_all_threads);
	N_k = div_N + (gindex < rem_N);
	index_start_move_k = gindex*div_N + rem_N*(gindex >= rem_N) + gindex*(gindex < rem_N); //gindex >= rem_N->rem_n otherwise gindex
																						 

	value_neigh_best = BIG_NUMBER;

	//-------- search neighbourhood ----------//
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
			//at first time always copied for each thread
			//pi = pi_neigh_best
			for (i = 0; i < n; i++)
			{
				pi[i] = shared_pi_neigh_best[i];
			}

			InsertTable(pi, j, v);
			job_j = j;
			job_v = v;
		}
		//----------------------------//

		if (j != v)
		{
			value_C = Criterion2(&n, &m, pi, shared_aj, shared_alphaj, CP);

			//---------- check Tabu List -------------//
			if ((job_j != job_v) && (value_C < value_neigh_best))
			{
				isInTabu = 0;
				for (i = 0; i < tabuListSize; i++)
				{
					if ((job_j == shared_tabuList[0 * tabuListSize + i]) && (job_v == shared_tabuList[1 * tabuListSize + i])) //[0][l], [1][l]
					{
						isInTabu = 1;
						break;
					}
				}

				if (!isInTabu)
				{
					value_neigh_best = value_C;
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

	//d_err_note[0] = 8 + 10*best_job_j + 100 * best_job_v + 8*1000;

	shared_value_neigh_best[lindex] = value_neigh_best;

	// wait for all threads to reach the barrier		
	__syncthreads();
	//<-------------------- barrier -------//

	//using index best - requires less memory per block than using lindex ==0 (i.e., job_j, job_v for each thred is not necessary, which give 2x4KB less memory

	index_thread_best = lindex;		//if value_i = value_k, then solution i is chosen instead of solution k (i<k)
	for (i = 0; i < threads_per_block; i++)	//to the number of threads, check if the currecn thread is the best
	{
		if ((shared_value_neigh_best[i] <  value_neigh_best) || (shared_value_neigh_best[i] == value_neigh_best) && (i < lindex)) //this thread (lindex) is not the best
		{
			index_thread_best = -1;
			break;
		}
	}

	if (lindex == index_thread_best)
	{
		d_best_value_neigh[block_index] = value_neigh_best;
		d_best_job_j[block_index] = best_job_j;
		d_best_job_v[block_index] = best_job_v;
	}

	//__syncthreads();
}



//------------ GPU AlgTS_BaT -------------//
// Full Utilization of Blocks and Threads
// Revised notation and names of variables 
void GPU_AlgTS_FFU_BaT(problem *p, solution &result, const solution sUB, int _tabuListSize, int iterN, int _threads_per_block, int _blocks_per_grid)	//threads per block, blocks
{
	int *d_pi, *d_n, *d_m;
	float *d_aj, *d_alphaj;
	int *d_parametersTS;

	int *d_tabuList;
	float *d_best_value_neigh;
	int *d_best_job_j;
	int *d_best_job_v;

	int *d_threads_per_block;
	int *d_blocks_per_grid;

	int threads_per_block;
	int blocks_per_grid;

	float *d_err_note;

	int i;
	int iter;
	float value_neigh;
	int index_block_best;

	result = sUB;

	int n, m;
	n = p->n;
	m = p->m;
	
	threads_per_block = _threads_per_block;
	blocks_per_grid = _blocks_per_grid;
	
	//--------------------//
	if (threads_per_block <= 0)
	{
		threads_per_block = MAX_THREADS;
	}
	threads_per_block = min(threads_per_block, MAX_THREADS);
	threads_per_block = min(threads_per_block, n*n);		// cannot be greater than neighourhood size, N= n*n


	if (blocks_per_grid <= 0)
	{
		blocks_per_grid = MAX_BLOCKS;
	}
	blocks_per_grid = min(blocks_per_grid, MAX_BLOCKS);
	blocks_per_grid = min(blocks_per_grid, (int)((n*n) / blocks_per_grid));		// blocksN*threadsN cannot be greater than neighourhood size, N= n*n
	blocks_per_grid = max(blocks_per_grid, 1);
	//--------------------//


	//------------- local variables ------//
	int tabuListSize;
	int *tabuList;
	tabuListSize = _tabuListSize;
	tabuList = new int[2 * tabuListSize];
	for (i = 0; i < 2 * tabuListSize; i++)
	{
		tabuList[i] = 0;
	}
	int tabuListIdx;
	tabuListIdx = 0;

	float *best_value_neigh;
	int *best_job_j;
	int *best_job_v;
	best_value_neigh = new float[blocks_per_grid];
	best_job_j = new int[blocks_per_grid];
	best_job_v = new int[blocks_per_grid];
	for (i = 0; i < blocks_per_grid; i++)
	{
		best_value_neigh[i] = 0;
		best_job_j[i] = 0;
		best_job_v[i] = 0;
	}

	float *aj, *alphaj;
	aj = new float[n];
	alphaj = new float[n];
	for (i = 0; i < n; i++)
	{
		aj[i] = p->jobs[0][i].aj;
		alphaj[i] = p->jobs[0][i].alphaj;
	}

	int *parametersTS;
	parametersTS = new int[2];
	parametersTS[0] = tabuListSize;
	parametersTS[1] = iterN;

	int *pi;
	int *pi_best;	//only locally important
	pi = new int[n];
	pi_best = new int[n];
	for (i = 0; i < n; i++)
	{
		pi[i] = result.pi[i];
		pi_best[i] = result.pi[i];
	}
	float value_best;
	value_best = result.value;

	float err_note = -1;

	cout << "===== TS GPU FFU BaT =====" << endl;
	cout << "blocksN " << blocks_per_grid << endl;
	cout << "threadsN " << threads_per_block << endl;
	cout << "listN " << parametersTS[0] << endl;
	cout << "iterN " << parametersTS[1] << endl;
	cout << "value " << value_best << endl;


	cudaMalloc(&d_blocks_per_grid, sizeof(int));
	cudaMalloc(&d_threads_per_block, sizeof(int));
	cudaMalloc(&d_pi, n * sizeof(int));
	cudaMalloc(&d_aj, n * sizeof(float));
	cudaMalloc(&d_alphaj, n * sizeof(float));
	cudaMalloc(&d_n, sizeof(int));
	cudaMalloc(&d_m, sizeof(int));
	cudaMalloc(&d_parametersTS, 2 * sizeof(int));
	cudaMalloc(&d_err_note, sizeof(float));
	cudaMalloc(&d_tabuList, 2 * tabuListSize * sizeof(int));
	cudaMalloc(&d_best_value_neigh, blocks_per_grid * sizeof(float));
	cudaMalloc(&d_best_job_j, blocks_per_grid * sizeof(int));
	cudaMalloc(&d_best_job_v, blocks_per_grid * sizeof(int));

		
	cudaMemcpy(d_blocks_per_grid, &blocks_per_grid, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_threads_per_block, &threads_per_block, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pi, pi, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_aj, aj, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_alphaj, alphaj, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_m, &m, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_parametersTS, parametersTS, 2 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_err_note, &err_note, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tabuList, tabuList, 2 * tabuListSize * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_best_value_neigh, best_value_neigh, blocks_per_grid * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_best_job_j, best_job_j, blocks_per_grid * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_best_job_v, best_job_v, blocks_per_grid * sizeof(int), cudaMemcpyHostToDevice);

	for (iter = 0; iter < iterN; iter++)
	{

		//input data <<<x=(N+255)/256, y=256>>>, i.e., sufficient that xy coveres N

		KernelTS_FFU_BaT <<< blocks_per_grid, threads_per_block >>> (d_blocks_per_grid, d_threads_per_block, d_n, d_m, d_pi, d_aj, d_alphaj, d_parametersTS, d_tabuList, d_best_value_neigh, d_best_job_j, d_best_job_v, time(NULL), d_err_note);

		//synchronize blocks
		cudaDeviceSynchronize();

		//Device -> Host
		cudaMemcpy(best_value_neigh, d_best_value_neigh, blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(best_job_j, d_best_job_j, blocks_per_grid * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(best_job_v, d_best_job_v, blocks_per_grid * sizeof(int), cudaMemcpyDeviceToHost);

		index_block_best = 0;
		value_neigh = best_value_neigh[index_block_best];
		
		for (i = 1; i < blocks_per_grid; i++)
		{		
			if (best_value_neigh[i] < value_neigh)
			{
				index_block_best= i;
				value_neigh = best_value_neigh[i];
			}
		}

		HostInsertTable(pi, best_job_j[index_block_best], best_job_v[index_block_best]);

		//---------- update the best ---------//
		if (value_best > value_neigh)
		{
			value_best = value_neigh;
			for (i = 0; i < n; i++)
			{
				pi_best[i] = pi[i];
			}
		}
		//------------------------------------//

		//------ add to tabu list -------//
		tabuList[0 * tabuListSize + tabuListIdx] = best_job_j[index_block_best];
		tabuList[1 * tabuListSize + tabuListIdx] = best_job_v[index_block_best];
		tabuListIdx = (tabuListIdx + 1) % tabuListSize;
		//-------------------------------//

		cudaMemcpy(d_pi, pi, n * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_tabuList, tabuList, 2 * tabuListSize * sizeof(int), cudaMemcpyHostToDevice);
	}

	//for tests only
	cudaMemcpy(&err_note, d_err_note, sizeof(float), cudaMemcpyDeviceToHost);
	
	for (i = 0; i < n; i++)
	{
		result.pi.Set(i, pi_best[i]);
	}

	result.value = p->Criterion(result.pi);

	cudaFree(d_blocks_per_grid);
	cudaFree(d_threads_per_block);
	cudaFree(d_pi);
	cudaFree(d_aj);
	cudaFree(d_alphaj);
	cudaFree(d_n);
	cudaFree(d_m);
	cudaFree(d_parametersTS);
	cudaFree(d_err_note);

	cudaFree(d_tabuList);
	cudaFree(d_best_value_neigh);
	cudaFree(d_best_job_j);
	cudaFree(d_best_job_v);

	delete[]parametersTS;
	delete[]aj;
	delete[]alphaj;
	delete[]pi;

	delete[]tabuList;
	delete[]best_value_neigh;
	delete[]best_job_j;
	delete[]best_job_v;

	delete[]pi_best;
}



//------------ Kernel TS BaT -------------//
// Full Utilization of Blocks and Threads
// Revised notation and names of variables 
//
// blocks are synchronized by global function, - host does not calculate any values, blocks communicate by global memory
// gindex == 0 manages d_tabuList, d_pi_best
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
__global__ void KernelTS_FFU_BaT_block_sync(int *d_blocksN, int *d_threads_per_block, int *d_n, int *d_m, int *d_pi, int *d_pi_best, float *d_value_best, float *d_aj, float *d_alphaj, int *d_parametersTS, int *d_tabuList, int *d_tabuListIdx, float *d_best_value_neigh, int *d_best_job_j, int *d_best_job_v, unsigned int seed, float *d_err_note)
{
	float CP[MAX_MACHINE_SIZE];				//for criterion value //cannot be shared, since all threads will use the same space, otherwise MAX_MACHINE_SIZE * MAX_THREADS

	float value_C;						//the temporary crterion value	
	float value_neigh_best;				//the best found value for the given thread
	int best_job_j, best_job_v;

	int pi[MAX_PROBLEM_SIZE];				// temporary pi for analysing neighbourhood

	__shared__ float shared_aj[MAX_PROBLEM_SIZE];			//much faster than local float aj_shared[MAX_PROBLEM_SIZE];
	__shared__ float shared_alphaj[MAX_PROBLEM_SIZE];

	int isInTabu;
	int tabuListSize;
	int tabuListIdx;		//only for faster update by gindex == 0
	__shared__ int shared_tabuList[2 * MAX_TABU_LIST];		//[1,1; 2,2; ...]

	__shared__ int shared_pi_neigh_best[MAX_PROBLEM_SIZE];	//pi_neigh the best for the block
	__shared__ float shared_value_neigh_best[MAX_THREADS];	//the best solution value for each thread

	int index_block_best;


	int threads_per_block;
	int blocks_per_grid;
	int number_all_threads;
	int n;
	int m;

	int j, v, k, i;
	int neighborhoodSizeN;		//number of moves per neighbourhood
	int N_k;					// number of moves per this thread
	int rem_N;					//reminder
	int div_N;					//division
	int index_start_move_k;		//the index of a start move of this thread, e.g., (0,0) is 0, (0,1) is 1, etc. (0,n-1) is n-1, (1,0) is n
	int job_j;
	int job_v;

	int gindex;
	int lindex;	//local index for each block				
	int block_index;
	int index_thread_best;

	n = *d_n;
	m = *d_m;

	threads_per_block = *d_threads_per_block;
	blocks_per_grid = *d_blocksN;
	number_all_threads = (*d_threads_per_block)*(*d_blocksN);

	lindex = threadIdx.x;	//local index for each block																								
	block_index = blockIdx.x;
	gindex = threadIdx.x + blockIdx.x*threads_per_block;

	//gindex = blockIdx.x * blockDim.x + threadIdx.x;


	tabuListSize = d_parametersTS[0];

	//initialize shared parameters
	if (lindex == 0)
	{
		//problem parameters
		for (j = 0; j<n; j++)
		{
			shared_aj[j] = d_aj[j];
			shared_alphaj[j] = d_alphaj[j];
		}

		//tabu list
		tabuListIdx = *d_tabuListIdx;		//to updata tabu list locally (globaly for gindex == 0)
		for (j = 0; j<tabuListSize * 2; j++)
		{
			shared_tabuList[j] = d_tabuList[j];
		}

		//solutions
		for (j = 0; j < n; j++)
		{
			shared_pi_neigh_best[j] = d_pi[j];
		}
	
		// initialize parameters from the previous iteration
		// if it is the first iteration (iter==1)
		// then d_best_value_neigh[...] = BIG_NUMBER, whereas d_value_best = result.value (initial)
		// thus no update (for d_pi_best and d_value_best)
		// and d_best_job_j/v [...] = 0, thus no insertion, and (0,0) added to tabuList

		index_block_best = 0;
		value_neigh_best = d_best_value_neigh[index_block_best];
		for (i = 1; i < blocks_per_grid; i++)
		{
			value_C = d_best_value_neigh[i];		//for faster analysis
			if (value_C < value_neigh_best)
			{
				index_block_best = i;
				value_neigh_best = value_C;
			}
		}

		
		
		best_job_j = d_best_job_j[index_block_best];
		best_job_v = d_best_job_v[index_block_best];		
		
		//d_err_note[0] = 8 + 10 * best_job_j + 100 * best_job_v + 8 * 1000;
		//d_err_note[0] = shared_tabuList[1*tabuListSize + 1];

		//if (!((best_job_j == 0) && (best_job_v == 0)))		//mozna ale nie trzeba, przy pierwszej iteracji pominie wpis
		{
			InsertTable(shared_pi_neigh_best, best_job_j, best_job_v);

			//------ add to tabu list -------//
			shared_tabuList[0 * tabuListSize + tabuListIdx] = best_job_j;
			shared_tabuList[1 * tabuListSize + tabuListIdx] = best_job_v;
			tabuListIdx = (tabuListIdx + 1) % tabuListSize;					//imortant only for gindex == 0
			//-------------------------------//

			//update global parameters (input for all blocks in the next iteration
			if (gindex == 0)
			{
				//------ update global neigh (update from previous iteration) -------//
				for (i = 0; i < n; i++)
				{
					d_pi[i] = shared_pi_neigh_best[i];
				}
				//--------------------------

				//---------- update global the best ---------//			
				if (value_neigh_best < *d_value_best)
				{
					*d_value_best = value_neigh_best;
					for (i = 0; i < n; i++)
					{
						d_pi_best[i] = shared_pi_neigh_best[i];
					}
				}
				//------------------------------------//

				//updata tabuList
				*d_tabuListIdx = tabuListIdx;
				for (i = 0; i < tabuListSize * 2; i++)
				{
					d_tabuList[i] = shared_tabuList[i];
				}
			}
		}
	}

	// wait for all threads to reach the barrier
	// all threads have the same tabuList, shared_pi_neigh_best
	__syncthreads();

	neighborhoodSizeN = n*n;
	rem_N = neighborhoodSizeN % number_all_threads;
	div_N = (int)(neighborhoodSizeN / number_all_threads);
	N_k = div_N + (gindex < rem_N);
	index_start_move_k = gindex*div_N + rem_N*(gindex >= rem_N) + gindex*(gindex < rem_N); //gindex >= rem_N->rem_n otherwise gindex


	value_neigh_best = BIG_NUMBER;

	//-------- search neighbourhood ----------//
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
			//at first time always copied for each thread
			//pi = pi_neigh_best
			for (i = 0; i < n; i++)
			{
				pi[i] = shared_pi_neigh_best[i];
			}

			InsertTable(pi, j, v);
			job_j = j;
			job_v = v;
		}
		//----------------------------//

		if (j != v)
		{
			value_C = Criterion2(&n, &m, pi, shared_aj, shared_alphaj, CP);

			//---------- check Tabu List -------------//
			if ((job_j != job_v) && (value_C < value_neigh_best))
			{
				isInTabu = 0;
				for (i = 0; i < tabuListSize; i++)
				{
					if ((job_j == shared_tabuList[0 * tabuListSize + i]) && (job_v == shared_tabuList[1 * tabuListSize + i])) //[0][l], [1][l]
					{
						isInTabu = 1;
						break;
					}
				}

				if (!isInTabu)
				{
					value_neigh_best = value_C;
					best_job_j = job_j;
					best_job_v = job_v;
				}
			}
			//----------------------------------------//
		}
		if ((job_j == 2) && (job_v == 3))
		{
		//	d_err_note[0] = value_C;
		//	d_err_note[0] = isInTabu;
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

	//d_err_note[0] = 8 + 10*best_job_j + 100 * best_job_v + 8*1000;
	//d_err_note[0] = value_neigh_best;

	d_err_note[0] = 8 + 10 * best_job_j + 100 * best_job_v + 8 * 1000;

	shared_value_neigh_best[lindex] = value_neigh_best;

	// wait for all threads to reach the barrier		
	__syncthreads();
	//<-------------------- barrier -------//

	//using index best - requires less memory per block than using lindex ==0 (i.e., job_j, job_v for each thred is not necessary, which give 2x4KB less memory

	index_thread_best = lindex;		//if value_i = value_k, then solution i is chosen instead of solution k (i<k)
	for (i = 0; i < threads_per_block; i++)	//to the number of threads, check if the currecn thread is the best
	{
		if ((shared_value_neigh_best[i] <  value_neigh_best) || (shared_value_neigh_best[i] == value_neigh_best) && (i < lindex)) //this thread (lindex) is not the best
		{
			index_thread_best = -1;
			break;
		}
	}

	if (lindex == index_thread_best)
	{
		d_best_value_neigh[block_index] = value_neigh_best;
		d_best_job_j[block_index] = best_job_j;
		d_best_job_v[block_index] = best_job_v;
	}

	//__syncthreads();
}


//------------ GPU AlgTS_BaT -------------//
// Full Utilization of Blocks and Threads
// Revised notation and names of variables 
void GPU_AlgTS_FFU_BaT_block_sync(problem *p, solution &result, const solution sUB, int _tabuListSize, int iterN, int _threads_per_block, int _blocks_per_grid)	//threads per block, blocks
{
	int *d_pi, *d_n, *d_m;
	int *d_pi_best;
	float *d_aj, *d_alphaj;
	int *d_parametersTS;

	int *d_tabuListIdx;
	int *d_tabuList;
	float *d_best_value_neigh;
	float *d_value_best;
	int *d_best_job_j;
	int *d_best_job_v;

	int *d_threads_per_block;
	int *d_blocks_per_grid;

	int threads_per_block;
	int blocks_per_grid;

	float *d_err_note;

	int i;
	int iter;
	float value_neigh;
	int index_block_best;

	result = sUB;

	int n, m;
	n = p->n;
	m = p->m;

	threads_per_block = _threads_per_block;
	blocks_per_grid = _blocks_per_grid;

	//--------------------//
	if (threads_per_block <= 0)
	{
		threads_per_block = MAX_THREADS;
	}
	threads_per_block = min(threads_per_block, MAX_THREADS);
	threads_per_block = min(threads_per_block, n*n);		// cannot be greater than neighourhood size, N= n*n


	if (blocks_per_grid <= 0)
	{
		blocks_per_grid = MAX_BLOCKS;
	}
	blocks_per_grid = min(blocks_per_grid, MAX_BLOCKS);
	blocks_per_grid = min(blocks_per_grid, (int)((n*n) / blocks_per_grid));		// blocksN*threadsN cannot be greater than neighourhood size, N= n*n
	blocks_per_grid = max(blocks_per_grid, 1);
	//--------------------//


	//------------- local variables ------//
	int tabuListSize;
	int *tabuList;
	tabuListSize = _tabuListSize;
	tabuList = new int[2 * tabuListSize];
	for (i = 0; i < 2 * tabuListSize; i++)
	{
		tabuList[i] = 0;
	}
	int tabuListIdx;
	tabuListIdx = 0;

	float *best_value_neigh;
	int *best_job_j;
	int *best_job_v;
	best_value_neigh = new float[blocks_per_grid];
	best_job_j = new int[blocks_per_grid];
	best_job_v = new int[blocks_per_grid];
	for (i = 0; i < blocks_per_grid; i++)
	{
		best_value_neigh[i] = BIG_NUMBER;
		best_job_j[i] = 0;
		best_job_v[i] = 0;
	}

	float *aj, *alphaj;
	aj = new float[n];
	alphaj = new float[n];
	for (i = 0; i < n; i++)
	{
		aj[i] = p->jobs[0][i].aj;
		alphaj[i] = p->jobs[0][i].alphaj;
	}

	int *parametersTS;
	parametersTS = new int[2];
	parametersTS[0] = tabuListSize;
	parametersTS[1] = iterN;

	int *pi;
	int *pi_best;	//only locally important
	pi = new int[n];
	pi_best = new int[n];
	for (i = 0; i < n; i++)
	{
		pi[i] = result.pi[i];
		pi_best[i] = result.pi[i];
	}
	float value_best;
	value_best = result.value;

	float err_note = -1;

	cout << "===== TS GPU FFU BaT block sync =====" << endl;
	cout << "blocksN " << blocks_per_grid << endl;
	cout << "threadsN " << threads_per_block << endl;
	cout << "listN " << parametersTS[0] << endl;
	cout << "iterN " << parametersTS[1] << endl;
	cout << "value " << value_best << endl;


	cudaMalloc(&d_blocks_per_grid, sizeof(int));
	cudaMalloc(&d_threads_per_block, sizeof(int));
	cudaMalloc(&d_n, sizeof(int));
	cudaMalloc(&d_m, sizeof(int));
	cudaMalloc(&d_pi, n * sizeof(int));
	cudaMalloc(&d_pi_best, n * sizeof(int));
	cudaMalloc(&d_value_best, sizeof(float));
	cudaMalloc(&d_aj, n * sizeof(float));
	cudaMalloc(&d_alphaj, n * sizeof(float));
	cudaMalloc(&d_parametersTS, 2 * sizeof(int));
	cudaMalloc(&d_tabuList, 2 * tabuListSize * sizeof(int));
	cudaMalloc(&d_tabuListIdx, sizeof(int));
	cudaMalloc(&d_best_value_neigh, blocks_per_grid * sizeof(float));
	cudaMalloc(&d_best_job_j, blocks_per_grid * sizeof(int));
	cudaMalloc(&d_best_job_v, blocks_per_grid * sizeof(int));
	cudaMalloc(&d_err_note, sizeof(float));

	cudaMemcpy(d_blocks_per_grid, &blocks_per_grid, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_threads_per_block, &threads_per_block, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_m, &m, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pi, pi, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pi_best, pi, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_value_best, &value_best, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_aj, aj, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_alphaj, alphaj, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_parametersTS, parametersTS, 2 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tabuList, tabuList, 2 * tabuListSize * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tabuListIdx, &tabuListIdx, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_best_value_neigh, best_value_neigh, blocks_per_grid * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_best_job_j, best_job_j, blocks_per_grid * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_best_job_v, best_job_v, blocks_per_grid * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_err_note, &err_note, sizeof(float), cudaMemcpyHostToDevice);


	for (iter = 0; iter < iterN; iter++)
	{
		//input data <<<x=(N+255)/256, y=256>>>, i.e., sufficient that xy coveres N //przyklad przeliczania
		KernelTS_FFU_BaT_block_sync <<< blocks_per_grid, threads_per_block >>> (d_blocks_per_grid, d_threads_per_block, d_n, d_m, d_pi, d_pi_best, d_value_best, d_aj, d_alphaj, d_parametersTS, d_tabuList, d_tabuListIdx, d_best_value_neigh, d_best_job_j, d_best_job_v, time(NULL), d_err_note);

		//synchronize blocks
		cudaDeviceSynchronize();		
	}
	//final update (from the last iteration //
	//Device -> Host
	cudaMemcpy(pi, d_pi, n * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(pi_best, d_pi_best, n * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&value_best, d_value_best, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(best_value_neigh, d_best_value_neigh, blocks_per_grid * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(best_job_j, d_best_job_j, blocks_per_grid * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(best_job_v, d_best_job_v, blocks_per_grid * sizeof(int), cudaMemcpyDeviceToHost);

	index_block_best = 0;
	value_neigh = best_value_neigh[index_block_best];
	for (i = 1; i < blocks_per_grid; i++)
	{
		if (best_value_neigh[i] < value_neigh)
		{
			index_block_best = i;
			value_neigh = best_value_neigh[i];
		}
	}
	
	//---------- update the best ---------//
	if (value_best > value_neigh)
	{		
		HostInsertTable(pi, best_job_j[index_block_best], best_job_v[index_block_best]);		//do not copy if the last neigh is not the best
		value_best = value_neigh;
		for (i = 0; i < n; i++)
		{
			pi_best[i] = pi[i];
		}
	}
	//------------------------------------//

	
	//for tests only	
	cudaMemcpy(&err_note, d_err_note, sizeof(float), cudaMemcpyDeviceToHost);

	for (i = 0; i < n; i++)
	{
		result.pi.Set(i, pi_best[i]);
	}
	
	result.value = p->Criterion(result.pi);

	cudaFree(d_blocks_per_grid);
	cudaFree(d_threads_per_block);
	cudaFree(d_n);
	cudaFree(d_m);
	cudaFree(d_pi);
	cudaFree(d_pi_best);
	cudaFree(d_value_best);
	cudaFree(d_aj);
	cudaFree(d_alphaj);
	cudaFree(d_tabuList);
	cudaFree(d_tabuListIdx);
	cudaFree(d_best_value_neigh);
	cudaFree(d_best_job_j);
	cudaFree(d_best_job_v);
	cudaFree(d_parametersTS);
	cudaFree(d_err_note);


	
	delete[]pi;
	delete[]pi_best;
	delete[]tabuList;
	delete[]aj;
	delete[]alphaj;
	delete[]parametersTS;
	delete[]best_value_neigh;
	delete[]best_job_j;
	delete[]best_job_v;


}

