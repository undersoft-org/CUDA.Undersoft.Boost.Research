//!!! czy algorytmy powinny wyliczac sobie UB.value w oparciu o permutacje
// czy tez pobierac ja jako argument
// - jezeli jako arg to przyspiesze
// - jezeli liczy przez pi to unikanie bledu
//


#include"stdafx.h"

#include"problem.h"
#include"GSortAlg.h"

#include "MyKernel.h"


//------------------------------------------//
//----------------- problem ----------------//
//------------------------------------------//

//---------------- problem -----------------//
problem::problem()
{
	n = 0;
	m = 0;
	jobs = NULL;
	Cj = NULL;

	K_KB_MB = 1024 * 1024;
}

//---------------- ~problem ----------------//
problem::~problem()
{
	ClearProblem();
}


//--------------- SetProblem ---------------//
void problem::SetProblem(int m, int n)
{

	int i, j;

	//delete earlier settings
	ClearProblem();

	/*if(jobs!=NULL)
	{
	for(int i=0; i<this->m; i++)
	if(jobs[i]!=NULL) delete jobs[i];
	delete jobs;
	}
	if(Cj!=NULL) delete Cj;
	*/


	//new values
	this->m = m;
	this->n = n;

	jobs = new job*[m];

	for (i = 0; i<m; i++)
	{
		jobs[i] = new job[n];
		for (j = 0; j<n; j++)
			jobs[i][j].pj = new float[n];
	}

	Cj = new float[m];			//criteria for machines 0, ..., m-1
}

//-------------- ClearProblem --------------//
void problem::ClearProblem()
{
	if (jobs != NULL)
	{
		for (int i = 0; i<m; i++)
		{
			for (int j = 0; j<n; j++)
			{
				if (jobs[i][j].pj != NULL) delete jobs[i][j].pj;
				//if(jobs[i][j].gji!=NULL) delete jobs[i][j].gji;
				//if(jobs[i][j].aji!=NULL) delete jobs[i][j].aji;
			}
			if (jobs[i] != NULL) delete jobs[i];
		}
		delete jobs;
	}

	if (Cj != NULL) delete Cj;  //criteria for all machines

	jobs = NULL;
	Cj = NULL;
	n = 0;
	m = 0;
}

//--------------- criterion ----------------//
float problem::Criterion(permutation pi, int h) //Pm||Cmax
{
	int i, j, v;
	float C;
	float *CP;		//completion times on P

	h = min(h, n - 1); // to do not exceed number of jobs

	CP = new float[m];

	for (i = 0; i<m; i++) CP[i] = 0;

	v = 0;
	i = 0;
	for (j = 0; j<n; j++)
	{
		if (pi[j] >= n - m + 1)		//[1, 2, n-2, n-3, 4, 5, n-1] m=4 --> n-m+1 = n-3 -> P0(1, 2), P1(), P2(4, 5), P3()
		{
			i++;
			v = 0;			
		}
		else
		{
			CP[i] += p_j(0, pi[j], v);
			v += jobs[0][pi[j]].aj;
		}
	}

	C = CP[0];
	for (i = 1; i<m; i++)
	{
		if (C<CP[i]) C = CP[i];
	}
	delete[]CP;
	return(C);
}



//----------------- AlgTS_neigh_iter ------------------//
// dziala lepiej niz TS
// iteruje wszystkie n^2 ruchy w sasiedztwie
// parallel moze dawac inny wynik,
// gdyz to samo kryterium moze byc dla roznych j,v, wowczas inne j,v trafiaja na liste tabu (w innej kolejnosci) 
void problem::AlgTS_neigh_iter(solution &result, const solution sUB, int listN, int iterN)
{
	bool isInTabu, firstNeigh;
	int i, iter, listIndx, **listTabu, best_jobv1, best_jobv2;
	solution s_tmp, s_neigh, s_neigh_best, *listTab;
	float value_best_neigh;

	int N;
	int j, v, k;
	int job_j;
	int job_v; //? czy potrzebne?
	int intN;
	int restN;
	
	listTabu = new int*[2];
	for (i = 0; i<2; i++)
	{
		listTabu[i] = new int[listN];
		for (j = 0; j < listN; j++)
		{
			listTabu[i][j] = 0;
		}
	}

	s_tmp = sUB;
	s_tmp.value = Criterion(s_tmp.pi);
	s_neigh_best = s_tmp;
	result = s_tmp;
	listIndx = 0;

	N = n*n;

	value_best_neigh = s_neigh_best.value;

	for (iter = 0; iter<iterN; iter++)
	{
		best_jobv1 = 0;
		best_jobv2 = 0;

		// ??? firstNeigh = 1;
		firstNeigh = 0;

		value_best_neigh = 100000000;

		s_tmp = s_neigh_best;

		j = 0;
		v = 0;
		job_j = -1;

		for (k = 0; k < N; k++)
		{							
			//------- obtain move --------//
			if (job_j == j)
			{
				s_tmp.pi.Swap(v - 1, v);	//here v is > 0 - always
				job_j = j;
				job_v = v;
			}
			else
			{
				s_tmp = s_neigh_best;
				s_tmp.pi.Insert(j, v);
				job_j = j;
				job_v = v;
			}
			//----------------------------//
			
			if (j != v)		//swap nie zrobi, gdy j==v, a nie powinien tylko wylcizaæ
			{
				s_tmp.value = Criterion(s_tmp.pi);

				//---------- check Tabu List -------------//
				if ((job_j != job_v) && (firstNeigh == 1 || s_tmp.value < value_best_neigh))
				{
					isInTabu = 0;
					for (int l = 0; l < listN; l++)
					{
						if ((job_j == listTabu[0][l]) && (job_v == listTabu[1][l]))
						{
							isInTabu = 1;
							break;
						}
					}

					if (!isInTabu)
					{
						value_best_neigh = s_tmp.value;
						best_jobv1 = job_j;
						best_jobv2 = job_v;
						firstNeigh = 0;
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

		if ((best_jobv1 == 0) && (best_jobv2 == 0))
		{
			cout << "000000000000000" << endl;
		}

		s_neigh_best.pi.Insert(best_jobv1, best_jobv2);
		s_neigh_best.value = value_best_neigh;

		//------ add to tabu list ------//
		listTabu[0][listIndx] = best_jobv1;
		listTabu[1][listIndx] = best_jobv2;

		listIndx = (listIndx + 1) % listN;
		//------------------------------//

		if (result.value > s_neigh_best.value)
		{
			result = s_neigh_best;		
		}
	}


	//    cout<<"TS - "<<result.value<<endl;

	//cout << "TS iter " << result.value << " : " << result.pi << endl;

	//cout<<"TS<---------------"<<endl;
	for (k = 0; k<2; k++)
		delete listTabu[k];

	delete listTabu;

	//  cout<<"result = " <<result.value<<endl;
	//  cout<<"########################"<<endl;
	//  system("PAUSE");

}

//----------------- AlgTS2 as GPU TS2 ------------------//
// dziala lepiej niz TS
// iteruje wszystkie n^2-n ruchy w sasiedztwie
// ale wtedy dla pominietych nie robi "swap" i nie przechodzi do wlasiwej pozycji wstawienia
// nie bedzie dzialal jak tabu search - z zasady pomija rozwiazania
// rozwiazanie zastosowane to: wprowadzic swap-swap (x2) w takich przypadkach tj. (v == j+1)
// podejscie z tablica ma sens przy rozpraszaniu, ale zysk nie bedzie zbyt wielki (teoretycznie -1 ruch mniej dla watku)
//
void problem::AlgTS_neigh_iter_tab(solution &result, const solution sUB, int listN, int iterN)
{
	bool isInTabu, firstNeigh;
	int i, l, iter, listIndx, **listTabu, best_jobv1, best_jobv2;
	solution s_tmp, s_neigh, s_neigh_best, *listTab;
	float value_best_neigh;

	int N;
	int j, v, k;
	int job_j;
	int job_v; //? czy potrzebne?
	int intN;
	int restN;
	int **neigh_tab;

	listTabu = new int*[2];
	for (i = 0; i<2; i++)
	{
		listTabu[i] = new int[listN];
		for (j = 0; j < listN; j++)
		{
			listTabu[i][j] = 0;
		}
	}

	s_tmp = sUB;
	s_tmp.value = Criterion(s_tmp.pi);
	s_neigh_best = s_tmp;
	result = s_tmp;
	listIndx = 0;


	//------- initialize neighour move iteration table ---------//
	
	N = n*n;
	neigh_tab = new int*[2];

	for (i = 0; i<2; i++)
	{
		neigh_tab[i] = new int[N-n];
	}

	v = 0;
	j = 0;
	k = 0;
	for (l = 0; l < N; l++)
	{
		if (j != v)
		{
			neigh_tab[0][k] = j;
			neigh_tab[1][k] = v;
			k++;
		}

		v++;
		if (v >= n)
		{
			v = 0;
			j++;
		}
	}
	
	//----------------------------------------------------------//



	value_best_neigh = s_neigh_best.value;

	for (iter = 0; iter<iterN; iter++)
	{
		best_jobv1 = 0;
		best_jobv2 = 0;

		firstNeigh = 1;

		s_tmp = s_neigh_best;

		j = 0;
		v = 0;
		job_j = -1;

		for (k = 0; k < N-n; k++)
		{
			j = neigh_tab[0][k];
			v = neigh_tab[1][k];

			//jezeli bylo (2, 1)->(2,2)-(2,3) to (2,2) nie pojawi sie, ale swap zrobi (2,2) i pomienite zostanie jednen ruch (2,n)

			//------- obtain move --------//
			if (job_j == j)
			{
				if (v == j+1)		//(2,3), czyli przejscie (przeskok byl) z (2,1) -> (2,3)
				{
					s_tmp.pi.Swap(v - 2, v-1);
				}

				s_tmp.pi.Swap(v - 1, v);	//here v is > 0 - always

				job_j = j;
				job_v = v;
			}
			else
			{
				s_tmp = s_neigh_best;
				s_tmp.pi.Insert(j, v);
				job_j = j;
				job_v = v;
			}
			//----------------------------//

			//if (j != v)		//swap nie zrobi, gdy j==v, a nie powinien tylko wylcizaæ
			{
				s_tmp.value = Criterion(s_tmp.pi);

				//---------- check Tabu List -------------//				
				if ((job_j != job_v) && (firstNeigh == 1 || s_tmp.value < value_best_neigh))
				{
					isInTabu = 0;
					for (int l = 0; l < listN; l++)
					{
						if ((job_j == listTabu[0][l]) && (job_v == listTabu[1][l]))
						{
							isInTabu = 1;
							break;
						}
					}

					if (!isInTabu)
					{
						value_best_neigh = s_tmp.value;
						best_jobv1 = job_j;
						best_jobv2 = job_v;
						firstNeigh = 0;
					}
				}
				//----------------------------------------//												
			}
		}

		s_neigh_best.pi.Insert(best_jobv1, best_jobv2);
		s_neigh_best.value = value_best_neigh;

		//------ add to tabu list ------//
		listTabu[0][listIndx] = best_jobv1;
		listTabu[1][listIndx] = best_jobv2;

		listIndx = (listIndx + 1) % listN;
		//------------------------------//

		if (result.value > s_neigh_best.value)
		{
			result = s_neigh_best;
		}
	}


	//    cout<<"TS - "<<result.value<<endl;

	//cout<<"TS<---------------"<<endl;
	for (l = 0; l<2; l++)
	{
		delete listTabu[l];	
	}
	delete []listTabu;

	for (l = 0; l < 2; l++)
	{
		delete neigh_tab[l];
	}
	delete []neigh_tab;

	

	//  cout<<"result = " <<result.value<<endl;
	//  cout<<"########################"<<endl;
	//  system("PAUSE");

}



// bj, alphaj, gj, aj, dj
//---------------- DrawJobs ----------------//
void problem::DrawJobs(int bj_min, int bj_max, float bj_s, int alphaj_min, int alphaj_max, float alphaj_s,
	int gj_min, int gj_max, float gj_s, int aj_min, int aj_max, float aj_s)
{


	//----- LE power -----//

	int i, j, v, minpj;
	float maxbj;

	for (j = 0; j<n; j++)
	{
		//jobs[0][j].bj = (bj_min + rand() % (bj_max - bj_min + 1))*bj_s;
		//jobs[0][j].alphaj = (alphaj_min + rand()%(alphaj_max-alphaj_min+1) )*alphaj_s;

		jobs[0][j].alphaj = alphaj_max*alphaj_s;

		jobs[0][j].gj = (int)((gj_min + rand() % (gj_max - gj_min + 1))*gj_s*n);
		jobs[0][j].aj = (aj_min + rand() % (aj_max - aj_min + 1))*aj_s;
	}

	minpj = jobs[0][0].aj;
	maxbj = jobs[0][0].bj;
	for (j = 1; j<n; j++)
	{
		if (jobs[0][j].aj<minpj) minpj = jobs[0][j].aj;
		if (jobs[0][j].bj<maxbj) maxbj = jobs[0][j].bj;
	}

	jobs[0][0].gj = int((minpj - 1) / jobs[0][0].bj);		//g<= pmin/b

															/*
															for(j=0; j<n; j++)
															{
															for(v=0; v<n; v++)
															{
															cout<<p_j(0,j,v)<<" ";
															}
															cout<<endl;
															}
															*/

															// nie dziala DP jezeli nie sa brane pod uwage nadpisywanie wartosci np. 1\3 i 3\1 zamiast 3\1 i 1\3

															/*
															jobs[0][0].aj = 1;
															jobs[0][1].aj = 6;
															jobs[0][2].aj = 7;
															jobs[0][3].aj = 8;
															jobs[0][4].aj = 10;
															*/



															//nie dziala DP
															/*
															jobs[0][0].aj = 1;
															jobs[0][1].aj = 1;
															jobs[0][2].aj = 5;
															jobs[0][3].aj = 6;
															jobs[0][4].aj = 8;
															jobs[0][5].aj = 9;
															jobs[0][6].aj = 10;
															*/


															/*
															jobs[0][0].aj = 4;
															jobs[1][0].aj = 8;

															jobs[0][1].aj = 3;
															jobs[1][1].aj = 3;

															jobs[0][2].aj = 3;
															jobs[1][2].aj = 4;

															jobs[0][3].aj = 1;
															jobs[1][3].aj = 4;

															jobs[0][4].aj = 8;
															jobs[1][4].aj = 7;
															*/


															//----- AE stepwise ------//

															/*
															int i, j, k, l, v, *q_tmp, sumq, sumg;
															float *aji_tmp;


															for(i=0; i<m; i++)
															for(j=0; j<n; j++)
															{
															k = (2 + rand()%(n-2+1) ); //2..n (min + rand()%(max-min+1) )

															jobs[i][j].k = k;
															jobs[i][j].gji = new int[k];
															jobs[i][j].aji = new float[k];

															q_tmp = new int[k];

															//losowanie pomocnicze
															sumq = 0;
															for(l=0; l<k; l++)
															{

															jobs[i][j].aji[l] = (aj_min + rand()%(aj_max-aj_min+1) );
															q_tmp[l] = (1 + rand()%(n-1+1) ); //2..n (min + rand()%(max-min+1) );	 //1..n
															sumq += q_tmp[l];
															//	cout<<"q    "<<q_tmp[l]<<endl;
															}

															//cout<<"sum q "<<sumq<<endl;



															//------------ sortowanie aji i wyliczenie gji w q rosnace ------//
															// skalowanie moze powodowac przekroczenie n i odciecie schodkow koncowych
															permutation pi_tmp(k);
															aji_tmp = new float[k];
															for(l=0; l<k; l++)
															{
															q_tmp[l] = max(1, int(((float)q_tmp[l]*(float)n)/(float)sumq) );   //step length //skalowanie
															aji_tmp[l] =  jobs[i][j].aji[l];
															}
															sort_up(aji_tmp, pi_tmp,0,k-1);

															for(l=0; l<k; l++)
															{
															jobs[i][j].gji[l] = q_tmp[pi_tmp[l]];									//gji is step length
															jobs[i][j].aji[l] = aji_tmp[pi_tmp[l]];
															//	cout<<"gl    "<<l<<"   "<<jobs[j].gji[l]<<"   aj   "<<jobs[j].aji[l]<<endl;
															}

															//cout<<"--- skalowanie ---"<<endl;


															sumg = 1;
															for(l=0; l<k-1; l++)
															{
															//cout<<"q/s*n  "<<(float)q_tmp[l]/(float)sumq*(float)n<<endl;
															//cout<<"q*n/s  "<<((float)q_tmp[l]*(float)n)/(float)sumq<<endl;
															//cout<<"aaaa  int "<<int(((float)q_tmp[l]*(float)n)/(float)sumq)<<endl;
															//cout<<"dodaj "<<max(1, int(((float)q_tmp[l]*(float)n)/(float)sumq) )<<endl;

															sumg += jobs[i][j].gji[l];		//gij is threshold, cannot be greater than n
															sumg = min(n, sumg);		// powinno rozwiazac problem obcinania schodkow przez skalowanie (za duze wartosci gji)
															jobs[i][j].gji[l] = sumg;		//gji = 1,...,n
															//	cout<<"gl    "<<l<<"   "<<jobs[i][j].gjl[l]<<"   aj   "<<jobs[i][j].aji[l]<<endl;
															}
															jobs[i][j].gji[k-1] = n;

															//cout<<"gl    "<<l<<"   "<<jobs[i][j].gji[k-1]<<"   aj   "<<jobs[i][j].aji[k-1]<<endl;
															//cout<<"sumg   "<<sumg<<endl;
															//		cout<<jobs[i][j].k<<endl;



															delete q_tmp;
															delete aji_tmp;

															l=0;
															for(v=0; v<n; v++)
															{
															if((v+1==jobs[i][j].gji[l])&&(l<k-1)) l++; //pozycja v=1,...,n, v+1 zeby nie bylo od 0
															jobs[i][j].pj[v]=jobs[i][j].aji[l];
															//cout<<"p^"<<i<<"_"<<j<<"("<<v+1<<")= "<<jobs[i][j].pj[v]<<endl;
															}
															}

															//can be delated since pij(v) is calculated
															for(i=0; i<m; i++)
															for(j=0; j<n; j++)
															{
															delete jobs[i][j].gji;
															jobs[i][j].gji = NULL;
															delete jobs[i][j].aji;
															jobs[i][j].aji = NULL;
															}

															*/

															/*

															jobs[0][0].aj =     1;
															jobs[0][1].aj =     6;
															jobs[0][2].aj =     10;
															jobs[0][3].aj =     5;
															jobs[0][4].aj =     9;
															jobs[0][5].aj =     7;
															jobs[0][6].aj =     8;
															jobs[0][7].aj =     8;
															jobs[0][8].aj =     3;
															jobs[0][9].aj =     4;
															jobs[0][10].aj =     3;
															jobs[0][11].aj =     7;
															jobs[0][12].aj =     2;
															jobs[0][13].aj =     3;
															jobs[0][14].aj =     6;
															jobs[0][15].aj =     10;
															jobs[0][16].aj =     5;
															jobs[0][17].aj =     10;
															jobs[0][18].aj =     7;

															*/


															/*
															jobs[0][0].aj = 6;
															jobs[0][1].aj = 3;
															jobs[0][2].aj = 5;
															jobs[0][3].aj = 2;
															jobs[0][4].aj = 5;
															jobs[0][5].aj = 5;
															jobs[0][6].aj = 9;
															jobs[0][7].aj = 0;
															*/


/*
jobs[0][0].aj = 9;
jobs[0][1].aj = 3;
jobs[0][2].aj = 14;
jobs[0][3].aj = 11;
jobs[0][4].aj = 1;
jobs[0][5].aj = 19;
jobs[0][6].aj = 14;
*/


//n=16, m=4, iter=50, listtabu 10, threads 4
/*
jobs[0][0].aj = 89;
jobs[0][1].aj = 33;
jobs[0][2].aj = 98;
jobs[0][3].aj = 56;
jobs[0][4].aj = 6;
jobs[0][5].aj = 49;
jobs[0][6].aj = 91;
jobs[0][7].aj = 100;
jobs[0][8].aj = 78;
jobs[0][9].aj = 28;
jobs[0][10].aj = 20;
jobs[0][11].aj = 84;
jobs[0][12].aj = 7;
jobs[0][13].aj = 20;
jobs[0][14].aj = 36;
*/

//n=16, m=4, iter=10, tabu=10, threads = 4

/*
jobs[0][0].aj = 52;
jobs[0][1].aj = 58;
jobs[0][2].aj = 55;
jobs[0][3].aj = 81;
jobs[0][4].aj = 64;
jobs[0][5].aj = 90;
jobs[0][6].aj = 62;
jobs[0][7].aj = 36;
jobs[0][8].aj = 24;
jobs[0][9].aj = 93;
jobs[0][10].aj = 48;
jobs[0][11].aj = 59;
jobs[0][12].aj = 15;
//jobs[0][13].aj = 0;
//jobs[0][14].aj = 0;
//jobs[0][15].aj = 0;

jobs[0][13].aj = 83;
jobs[0][14].aj = 55;
*/

}

//---------------- PrintJobs ---------------//
void problem::PrintJobs()
{
	int i, j;

	int sumpj;

	//cout<<jobs[0][0].aj<<endl;
	sumpj = 0;

	cout << n << " " << m << endl;

	for (j = 0; j<n; j++)
	{
		cout << "===============================" << endl;
		cout << "==========" << j << "===============" << endl;
		cout << "===============================" << endl;

		i = 0;
		//for(i=0; i<m; i++)
		{
			//cout<<"=========="<< i <<"==============="<<endl;

			jobs[i][n - 1].aj = 0;
			jobs[i][n - 1].bj = 0;
			jobs[i][n - 1].alphaj = 0;
			jobs[i][n - 1].gj = 0;;

			cout << "aj =     " << jobs[i][j].aj << endl;
			cout << "bj =     " << jobs[i][j].bj << endl;
			cout << "alphaj = " << jobs[i][j].alphaj << endl;
			cout << "gj =     " << jobs[i][j].gj << endl;
			//cout<<"================================="<<endl;



		}
		sumpj += jobs[0][j].aj;
	}



	cout << "sumpj  " << sumpj << endl;
}


void problem::AlgTSonGPU_neigh_iter(solution &result, const solution sUB, int listN, int iterN, int threadsN)
{
	//run on GPU in MyKernal.cu
	GPU_AlgTS_neigh_iter(this, result, sUB, listN, iterN, threadsN);

	//cout << "ItR " << result.value << " : " << result.pi << endl;
}


void problem::AlgTSonGPU_BaT(solution &result, const solution sUB, int listN, int iterN, int threadsN, int blocksN)
{
	//run on GPU in MyKernal.cu
	GPU_AlgTS_BaT(this, result, sUB, listN, iterN, threadsN, blocksN);
	
	//cout<<"BaT "<<result.value<<" : "<< result.pi << endl;
}

