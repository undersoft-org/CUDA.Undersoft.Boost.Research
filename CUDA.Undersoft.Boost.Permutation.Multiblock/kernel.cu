

#include"stdafx.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include"problem.h"
#include"GAlgDef.h"


#define CLK_PER_MS CLK_TCK*1000

//using namespace std;




int main()
{


	int i, j, n = 5, m = 1, ni = 1, namax = 50, na = 50, ina = 0; //n, m, ni, na - number of jobs, machines, instances and algorithms
	problem p;
	solution sBest, sA;
	clock_t t1, t2;
	time_t t;
	float errtmp;
	float Param[12];
	char fbuff;
	int firstif = 1;
	ifstream inStr;
	ofstream outStr, outStrTime, outStrMem, outStrLatex;


	//----------------------------------//
	AlgExp *algTab;
	algTab = new AlgExp[namax];

	//--------------------------//

	algTab[ina].name = "GPU TS FU BaT 1  ";
	algTab[ina].fun = gGPU_AlgTSonGPU_FU_BaT_1;
	algTab[ina].active = 0;
	ina++;

	algTab[ina].name = "GPU TS FU BaT 2  ";
	algTab[ina].fun = gGPU_AlgTSonGPU_FU_BaT_2;
	algTab[ina].active = 0;
	ina++;

	algTab[ina].name = "GPU TS FU BaT 4  ";
	algTab[ina].fun = gGPU_AlgTSonGPU_FU_BaT_4;
	algTab[ina].active = 0;
	ina++;

	algTab[ina].name = "GPU TS FU BaT 8  ";
	algTab[ina].fun = gGPU_AlgTSonGPU_FU_BaT_8;
	algTab[ina].active = 0;
	ina++;

	algTab[ina].name = "GPU TS FU BaT 16 ";
	algTab[ina].fun = gGPU_AlgTSonGPU_FU_BaT_16;
	algTab[ina].active = 0;
	ina++;


	algTab[ina].name = "GPU TS FU BaT    ";
	algTab[ina].fun = gGPU_AlgTSonGPU_FU_BaT;
	algTab[ina].active = 0;//
	ina++;

	algTab[ina].name = "GPU TS FFU BaT   ";
	algTab[ina].fun = gGPU_AlgTSonGPU_FFU_BaT;
	algTab[ina].active = 3;
	ina++;

	algTab[ina].name = "GPU TS FFU BaT bs ";
	algTab[ina].fun = gGPU_AlgTSonGPU_FFU_BaT_block_sync;
	algTab[ina].active = 3;
	ina++;

	na = ina;


	//system("PAUSE");
	//--------------------------//

	//------ init files ------//
	inStr.open("test.txt");
	if (inStr.fail())
	{
		cout << "blad otwierania pliku " << endl;
		exit(1);
	}

	outStr.open("fileout.txt");
	if (outStr.fail())
	{
		cout << "blad otwierania pliku " << endl;
		exit(1);
	}

	outStrTime.open("fileout_time.txt");
	if (outStrTime.fail())
	{
		cout << "blad otwierania pliku " << endl;
		exit(1);
	}

	outStrMem.open("fileout_mem.txt");
	if (outStrMem.fail())
	{
		cout << "blad otwierania pliku " << endl;
		exit(1);
	}

	outStrLatex.open("fileout_latex.txt");
	if (outStrMem.fail())
	{
		cout << "blad otwierania pliku " << endl;
		exit(1);
	}
	//------------------------//        

	inStr.ignore(1000, ';');
	if (inStr.eof()) cout << "Plik wejsciowy pusty, badz niewlasciwy format pliku..." << endl;

	while (!inStr.eof() && !inStr.bad())
	{
		inStr >> n >> m >> ni;

		if (inStr.eof()) break;          // in case of empty lines at the enf of file

		for (i = 0; i<12; i++)
			inStr >> Param[i];

		srand((unsigned)time(&t));

		p.SetProblem(m, n);

		sA.pi.InitPerm(n);
		sBest.pi.InitPerm(n);

		for (i = 0; i<na; i++)
			algTab[i].Clear();

		cout << "n = " << n << "\t" << "m = " << m << "\t" << "ni = " << ni << endl;
		cout << "bj = [" << Param[0] * Param[2] << ", " << Param[1] * Param[2] << "]\t" <<
			"aphaj = [" << Param[3] * Param[5] << ", " << Param[4] * Param[5] << "]\t" <<
			"gj = [" << Param[6] * Param[8] << ", " << Param[7] * Param[8] << "]\t" << endl;
		cout << "aj = [" << Param[9] * Param[11] << ", " << Param[10] * Param[11] << "]" << endl;
		cout << "======================================" << endl << endl;

		cout << "============" << endl;
		for (i = 0; i<ni; i++)
		{

			// bj, alphaj, gj, aj
			p.DrawJobs(Param[0], Param[1], Param[2],
				Param[3], Param[4], Param[5],
				Param[6], Param[7], Param[8],
				Param[9], Param[10], Param[11]);

			//p.PrintJobs();

			firstif = 1;
			for (j = 0; j<na; j++)
				if (algTab[j].active != 0)
				{
					sA.param = 0;

					t1 = clock();
					algTab[j].fun(&p, sA);
					t2 = clock();

					algTab[j].value = sA.value;

					if ((algTab[j].tmin > t2 - t1) || (i == 0)) algTab[j].tmin = t2 - t1;
					if ((algTab[j].tmax < t2 - t1) || (i == 0)) algTab[j].tmax = t2 - t1;
					algTab[j].t += (t2 - t1);

					if (((algTab[j].kmin > sA.param) || (i == 0)) && (algTab[j].active == 1)) //number of visited nodes
						algTab[j].kmin = sA.param;
					if (((algTab[j].kmax < sA.param) || (i == 0)) && (algTab[j].active == 1)) //number of visited nodes
						algTab[j].kmax = sA.param;
					if (algTab[j].active == 1)
						algTab[j].k = algTab[j].k - (algTab[j].k - sA.param) / (i + 1);//(algTab[j].k/(i+1))*i + sA.param/(i+1);

					if (((algTab[j].k2min > sA.param2) || (i == 0)) && (algTab[j].active == 1)) //number of visited nodes
						algTab[j].k2min = sA.param2;
					if (((algTab[j].k2max < sA.param2) || (i == 0)) && (algTab[j].active == 1)) //number of visited nodes
						algTab[j].k2max = sA.param2;
					if (algTab[j].active == 1)
						algTab[j].k2 = algTab[j].k2 - (algTab[j].k2 - sA.param2) / (i + 1);//(algTab[j].k/(i+1))*i + sA.param/(i+1);

																						   //------------ time alloc, calc, dell ----------//	
					if (((algTab[j].tmin_alloc > sA.t_alloc) || (i == 0)) && (algTab[j].active == 1))
						algTab[j].tmin_alloc = sA.t_alloc;
					if (((algTab[j].tmax_alloc < sA.t_alloc) || (i == 0)) && (algTab[j].active == 1))
						algTab[j].tmax_alloc = sA.t_alloc;
					if (algTab[j].active == 1)
						algTab[j].t_alloc = algTab[j].t_alloc - (algTab[j].t_alloc - sA.t_alloc) / (i + 1);

					if (((algTab[j].tmin_calc > sA.t_calc) || (i == 0)) && (algTab[j].active == 1))
						algTab[j].tmin_calc = sA.t_calc;
					if (((algTab[j].tmax_calc < sA.t_calc) || (i == 0)) && (algTab[j].active == 1))
						algTab[j].tmax_calc = sA.t_calc;
					if (algTab[j].active == 1)
						algTab[j].t_calc = algTab[j].t_calc - (algTab[j].t_calc - sA.t_calc) / (i + 1);

					if (((algTab[j].tmin_del > sA.t_del) || (i == 0)) && (algTab[j].active == 1))
						algTab[j].tmin_del = sA.t_del;
					if (((algTab[j].tmax_del < sA.t_del) || (i == 0)) && (algTab[j].active == 1))
						algTab[j].tmax_del = sA.t_del;
					if (algTab[j].active == 1)
						algTab[j].t_del = algTab[j].t_del - (algTab[j].t_del - sA.t_del) / (i + 1);
					//----------------------------------------------//


					//------------ memory alloc total diff ----------//	
					if (((algTab[j].mem_alloc_min > sA.mem_alloc) || (i == 0)) && (algTab[j].active == 1))
						algTab[j].mem_alloc_min = sA.mem_alloc;
					if (((algTab[j].mem_alloc_max < sA.mem_alloc) || (i == 0)) && (algTab[j].active == 1))
						algTab[j].mem_alloc_max = sA.mem_alloc;
					if (algTab[j].active == 1)
						algTab[j].mem_alloc = (SIZE_T)((long long int) algTab[j].mem_alloc - ((long long int)algTab[j].mem_alloc - (long long int)sA.mem_alloc) / (i + 1));

					if (((algTab[j].mem_total_min > sA.mem_total) || (i == 0)) && (algTab[j].active == 1))
						algTab[j].mem_total_min = sA.mem_total;
					if (((algTab[j].mem_total_max < sA.mem_total) || (i == 0)) && (algTab[j].active == 1))
						algTab[j].mem_total_max = sA.mem_total;
					if (algTab[j].active == 1)
						algTab[j].mem_total = (SIZE_T)((long long int)algTab[j].mem_total - ((long long int)algTab[j].mem_total - (long long int)sA.mem_total) / (i + 1));

					if (((algTab[j].mem_diff_min > sA.mem_diff) || (i == 0)) && (algTab[j].active == 1))
						algTab[j].mem_diff_min = sA.mem_diff;
					if (((algTab[j].mem_diff_max < sA.mem_diff) || (i == 0)) && (algTab[j].active == 1))
						algTab[j].mem_diff_max = sA.mem_diff;
					if (algTab[j].active == 1)
						algTab[j].mem_diff = (SIZE_T)((long long int)algTab[j].mem_diff - ((long long int)algTab[j].mem_diff - (long long int)sA.mem_diff) / (i + 1));

					//----------------------------------------------//



					if (((sBest.value > sA.value) || (firstif)) && (algTab[j].active != 0)) {
						sBest.pi = sA.pi;
						sBest.value = sA.value;
						firstif = 0;
					}
				}

			//------ calculate errors ------//
			for (j = 0; j<na; j++)
				if (algTab[j].active != 0)
				{
					errtmp = (algTab[j].value - sBest.value) * 100 / sBest.value;
					//if (fabs(errtmp)<0.00005) errtmp = 0.0;

					if ((algTab[j].errmin > errtmp) || (i == 0))
					{
						algTab[j].errmin = errtmp;
						if (algTab[j].active == 2)
							algTab[j].kmin = sBest.value;   //not for B&B

					}
					if ((algTab[j].errmax < errtmp) || (i == 0))
					{
						algTab[j].errmax = errtmp;
						if (algTab[j].active == 2)
							algTab[j].kmax = sBest.value; //not for B&B
					}

					algTab[j].err += errtmp;
					if (algTab[j].active == 2)
						algTab[j].k += sBest.value; //not for B&B

													//------ number iterations when it is the best ------//
					if (errtmp == 0)
						if (algTab[j].active == 3)
							algTab[j].k++;
				}
			//------------------------------//

			if (i % 10 == 0) cout << i << endl;
		}
		//----------------------------------//

		//------ Screen ------//
		cout << "============" << endl;
		for (j = 0; j<na; j++)
			if (algTab[j].active != 0)
				cout << algTab[j].name << "\t\t"
				<< algTab[j].err / ni << "\t\t" << algTab[j].errmin << "\t\t" << algTab[j].errmax << endl;
		cout << "======================================" << endl << endl;
		//--------------------//


		//------ write file ------//
		outStr << "n = " << n << "\t" << "m = " << m << "\t" << "ni = " << ni << endl;
		outStr << "bj = [" << Param[0] * Param[2] << ", " << Param[1] * Param[2] << "]\t" <<
			"alphaj = [" << Param[3] * Param[5] << ", " << Param[4] * Param[5] << "]\t" <<
			"gj = [" << Param[6] * Param[8] << ", " << Param[7] * Param[8] << "]\t" <<
			"aj = [" << Param[9] * Param[11] << ", " << Param[10] * Param[11] << "]" << endl;
		outStr << "======================================" << endl << endl;
		for (j = 0; j<na; j++)
			if (algTab[j].active != 0)
				outStr << algTab[j].name << "\t" << algTab[j].t / ni << " &\t" << algTab[j].tmin << " &\t" << algTab[j].tmax << " &&\t"
				<< algTab[j].err / ni << " &\t" << algTab[j].errmin << " &\t" << algTab[j].errmax << " &&\t"
				//
				<< algTab[j].k << " &\t" << algTab[j].kmin << " &\t" << algTab[j].kmax << " &&\t"
				<< algTab[j].k2 << " &\t" << algTab[j].k2min << " &\t" << algTab[j].k2max << " &&\t"
				//
				//<<algTab[j].t_alloc<<" &\t"<<algTab[j].tmin_alloc<<" &\t"<<algTab[j].tmax_alloc<<" &&\t"
				//<<algTab[j].t_calc<<" &\t"<<algTab[j].tmin_calc<<" &\t"<<algTab[j].tmax_calc<<" &&\t"
				//<<algTab[j].t_del<<" &\t"<<algTab[j].tmin_del<<" &\t"<<algTab[j].tmax_del
				//
				<< endl;

		outStr << "======================================" << endl << endl;
		//------------------------//


		//------ write file memory ------//
		outStrMem << "n = " << n << "\t" << "m = " << m << "\t" << "ni = " << ni << endl;
		outStrMem << "bj = [" << Param[0] * Param[2] << ", " << Param[1] * Param[2] << "]\t" <<
			"alphaj = [" << Param[3] * Param[5] << ", " << Param[4] * Param[5] << "]\t" <<
			"gj = [" << Param[6] * Param[8] << ", " << Param[7] * Param[8] << "]\t" <<
			"aj = [" << Param[9] * Param[11] << ", " << Param[10] * Param[11] << "]" << endl;
		outStrMem << "======================================" << endl << endl;
		for (j = 0; j<na; j++)
			if (algTab[j].active != 0)
				outStrMem << algTab[j].name << " &&\t"
				//
				//<<algTab[j].k<<" &\t"<<algTab[j].kmin<<" &\t"<<algTab[j].kmax<<" &&\t"
				//<<algTab[j].k2<<" &\t"<<algTab[j].k2min<<" &\t"<<algTab[j].k2max<<" &&\t"            
				//
				<< algTab[j].mem_alloc << " &\t" << algTab[j].mem_alloc_min << " &\t" << algTab[j].mem_alloc_max << " &&\t"
				<< algTab[j].mem_total << " &\t" << algTab[j].mem_total_min << " &\t" << algTab[j].mem_total_max << " &&\t"
				<< algTab[j].mem_diff << " &\t" << algTab[j].mem_diff_min << " &\t" << algTab[j].mem_diff_max
				//
				<< endl;

		outStrMem << "======================================" << endl << endl;
		//------------------------//

		//------ write file time ------//
		outStrTime << "n = " << n << "\t" << "m = " << m << "\t" << "ni = " << ni << endl;
		outStrTime << "bj = [" << Param[0] * Param[2] << ", " << Param[1] * Param[2] << "]\t" <<
			"alphaj = [" << Param[3] * Param[5] << ", " << Param[4] * Param[5] << "]\t" <<
			"gj = [" << Param[6] * Param[8] << ", " << Param[7] * Param[8] << "]\t" <<
			"aj = [" << Param[9] * Param[11] << ", " << Param[10] * Param[11] << "]" << endl;
		outStrTime << "======================================" << endl << endl;
		for (j = 0; j<na; j++)
			if (algTab[j].active != 0)
				outStrTime << algTab[j].name << " &&\t"
				//
				//<<algTab[j].k<<" &\t"<<algTab[j].kmin<<" &\t"<<algTab[j].kmax<<" &&\t"
				//<<algTab[j].k2<<" &\t"<<algTab[j].k2min<<" &\t"<<algTab[j].k2max<<" &&\t"            
				//
				<< algTab[j].t_alloc << " &\t" << algTab[j].tmin_alloc << " &\t" << algTab[j].tmax_alloc << " &&\t"
				<< algTab[j].t_calc << " &\t" << algTab[j].tmin_calc << " &\t" << algTab[j].tmax_calc << " &&\t"
				<< algTab[j].t_del << " &\t" << algTab[j].tmin_del << " &\t" << algTab[j].tmax_del
				//
				<< endl;

		outStrTime << "======================================" << endl << endl;
		//------------------------//



		//------ write file time ------//
		outStrLatex << "n = " << n << "\t" << "m = " << m << "\t" << "ni = " << ni << endl;
		outStrLatex << "bj = [" << Param[0] * Param[2] << ", " << Param[1] * Param[2] << "]\t" <<
			"alphaj = [" << Param[3] * Param[5] << ", " << Param[4] * Param[5] << "]\t" <<
			"gj = [" << Param[6] * Param[8] << ", " << Param[7] * Param[8] << "]\t" <<
			"aj = [" << Param[9] * Param[11] << ", " << Param[10] * Param[11] << "]" << endl;
		outStrLatex << "======================================" << endl << endl;
		for (j = 0; j<na; j++)
			if (algTab[j].active != 0)
				outStrLatex
				//
				<< n << " &\t"
				//
				<< Param[0] * Param[2] << " &\t"
				//
				<< "[" << Param[9] * Param[11] << ", " << Param[10] * Param[11] << "] &&\t"
				//
				<< algTab[j].name << " &&\t"
				//
				<< algTab[j].t / ni << " &\t"
				<< algTab[j].t_calc << " &\t"
				<< algTab[j].t_alloc << " &\t"
				<< algTab[j].t_del << " &&\t"
				//
				<< algTab[j].mem_total << " &\t"
				<< algTab[j].mem_alloc << " &\t"
				<< algTab[j].mem_diff << " &&\t"
				//
				<< algTab[j].k2 << " \t \\\\"
				//
				<< endl;

		outStrLatex << "======================================" << endl << endl;
		//------------------------//
	}
	inStr.close();
	outStr.close();
	outStrTime.close();
	outStrMem.close();
	outStrLatex.close();

	p.ClearProblem();
	delete algTab;














	system("PAUSE");
	/*
	MyAlgorithm *mya;

	mya = new MyAlgorithm;
	
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	int *q;

	q = new int;

	cudaGetDeviceCount(q);

	

	cout << *q <<endl;

	system("PAUSE");
	*/

    return 0;
}

/*

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
*/