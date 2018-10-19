//standard includes
#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <omp.h>

/*
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <curand.h>
#include <curand_kernel.h>
*/

#include "CycleTimer.h"

#define BLOCKSIZE  1024
#define SCAN_BLOCK_DIM  BLOCKSIZE
//#include "exclusiveScan.cu_inl"

//include definition file
#include "neuralNetwork.h"

using namespace std;


/*
__global__ void
forward_prop_kernel(double *device_output, double *input, double *weights, int num_first, int num_second) {
	int linearThreadIndex = threadIdx.x;
	int unit = blockIdx.x;

    __shared__ double prefixSumInput[BLOCKSIZE];
    __shared__ double prefixSumOutput[BLOCKSIZE];
    __shared__ double prefixSumScratch[2 * BLOCKSIZE];

    if (linearThreadIndex < num_first) {
    	prefixSumInput[linearThreadIndex] = input[linearThreadIndex] * weights[linearThreadIndex*num_second + unit];
    }

    __syncthreads();

    sharedMemExclusiveScan(linearThreadIndex, prefixSumInput, prefixSumOutput, 
                            prefixSumScratch, BLOCKSIZE);

    __syncthreads();

    if (linearThreadIndex == 0 && unit < num_second) {
    	device_output[unit] = 1/(1+exp(-1*prefixSumOutput[num_first]));
    }
}


// first, second -> input=input+1, nhidden
// first, second -> hidden=hidden+1, noutput
__global__ void
forward_prop_kernel_batch(double *device_output, double *input, double *weights, int num_first, int num_second, int batchSize) {
	int linearThreadIndex = threadIdx.x;
	// PRINT LINEAR THREAD INDEX TO DEBUG 
	int unit = blockIdx.x%num_second;
	int batch = blockIdx.x/num_second;

    __shared__ double prefixSumInput[BLOCKSIZE];
    __shared__ double prefixSumOutput[BLOCKSIZE];
    __shared__ double prefixSumScratch[2 * BLOCKSIZE];

    if (linearThreadIndex < num_first) {
    	prefixSumInput[linearThreadIndex] = input[batch*linearThreadIndex] * weights[linearThreadIndex*num_second + unit];
    }

    __syncthreads();

    sharedMemExclusiveScan(linearThreadIndex, prefixSumInput, prefixSumOutput, 
                            prefixSumScratch, BLOCKSIZE);

    __syncthreads();

    if (linearThreadIndex == 0 && unit < num_second) {
    	device_output[batch*unit] = 1/(1+exp(-1*prefixSumOutput[num_first]));
    }
}

*/

/*******************************************************************
* Constructor
********************************************************************/
neuralNetwork::neuralNetwork(int nI, int nH, int nO, int bS) : nInput(nI), nHidden(nH), nOutput(nO), batchSize(bS)
{				
	//create neuron lists
	//--------------------------------------------------------------------------------------------------------
	inputNeurons = new( double[batchSize*(nInput + 1)] );
        for (int b= 0; b<batchSize; b++) {
            for (int i=0; i<nInput+1; i++) {
                if (i==nInput) {
                    inputNeurons[(b+1)*(nInput)] = -1;
                }
                else {
                    inputNeurons[b*(nInput+1) + i] = 0;
                } 
            }
        }

	//create input bias neuron
	// inputNeurons[nInput] = -1;

	hiddenNeurons = new( double[batchSize*(nHidden + 1)] );
        for (int b=0; b<batchSize; b++) {
            for (int i=0; i<nHidden+1; i++) {
                if (i==nHidden) {
                    hiddenNeurons[(b+1)*(nHidden)] = -1;
                }
                else {
                    hiddenNeurons[b*(nHidden+1) + i] = 0; 
                }
            }
        }
	// for ( int i=0; i < nHidden; i++ ) hiddenNeurons[i] = 0;

	//create hidden bias neuron
	// hiddenNeurons[nHidden] = -1;

	// outputNeurons = new( double[nOutput] );
	outputNeurons = new( double[batchSize*(nOutput + 1)] );
	for ( int i=0; i < batchSize*(nOutput+1); i++ ) {
		outputNeurons[i] = 0;
	}

	// for ( int i=0; i < nOutput; i++ ) outputNeurons[i] = 0;

	//create weight lists (include bias neuron weights)
	//--------------------------------------------------------------------------------------------------------
	wInputHidden = new( double*[nInput + 1] );
	wInputHidden[0] = new (double[(nInput + 1)*nHidden]);
	for ( int i=1; i <= nInput; i++ ) {
		wInputHidden[i] = wInputHidden[i-1] + nHidden;
	}
	for ( int i=0; i <= nInput; i++ ) 
	{
		for ( int j=0; j < nHidden; j++ ) wInputHidden[i][j] = 0;		
	}

	wHiddenOutput = new( double*[nHidden + 1] );
	wHiddenOutput[0] = new (double[(nHidden + 1)*nOutput]);
	for ( int i=1; i <= nHidden; i++ ) {
		wHiddenOutput[i] = wHiddenOutput[i-1] + nOutput;
	}
	for ( int i=0; i <= nHidden; i++ ) 
	{
		for ( int j=0; j < nOutput; j++ ) wHiddenOutput[i][j] = 0;		
	}
	
	//initialize weights
	//--------------------------------------------------------------------------------------------------------
	initializeWeights();		
}

/*******************************************************************
* Destructor
********************************************************************/
neuralNetwork::~neuralNetwork()
{
	//delete neurons
	delete[] inputNeurons;
	delete[] hiddenNeurons;
	delete[] outputNeurons;

	//delete weight storage
	for (int i=0; i <= nInput; i++) delete[] wInputHidden[i];
	delete[] wInputHidden;

	for (int j=0; j <= nHidden; j++) delete[] wHiddenOutput[j];
	delete[] wHiddenOutput;

	/*
	cudaFree(device_output1);
	cudaFree(input);
	cudaFree(w1);

	cudaFree(device_output2);
	cudaFree(hidden);
	cudaFree(w2);
	*/
	
}

/*******************************************************************
* Save Neuron Weights
*******************************************************************/
bool neuralNetwork::saveWeights(char* filename)
{
	//open file for reading
	fstream outputFile;
	outputFile.open(filename, ios::out);

	if ( outputFile.is_open() )
	{
		outputFile.precision(50);		

		//output weights
		for ( int i=0; i <= nInput; i++ ) 
		{
			for ( int j=0; j < nHidden; j++ ) 
			{
				outputFile << wInputHidden[i][j] << ",";				
			}
		}
		
		for ( int i=0; i <= nHidden; i++ ) 
		{		
			for ( int j=0; j < nOutput; j++ ) 
			{
				outputFile << wHiddenOutput[i][j];					
				if ( i * nOutput + j + 1 != (nHidden + 1) * nOutput ) outputFile << ",";
			}
		}

		//print success
		cout << endl << "Neuron weights saved to '" << filename << "'" << endl;

		//close file
		outputFile.close();
		
		return true;
	}
	else 
	{
		cout << endl << "Error - Weight output file '" << filename << "' could not be created: " << endl;
		return false;
	}
}

/*******************************************************************
* Return the NN accuracy on the set
********************************************************************/
double neuralNetwork::getSetAccuracy( std::vector<dataEntry*>& set )
{
	double incorrectResults = 0;
		
	//for every training input array
	for ( int tp = 0; tp < (int) set.size(); tp++)
	{						
		//feed inputs through network and backpropagate errors
		feedForward( set[tp]->pattern );

		int predicted = distance(outputNeurons, max_element(outputNeurons, outputNeurons + nOutput));
		int expected = distance(set[tp]->target, max_element(set[tp]->target, set[tp]->target + nOutput));
		
		if (predicted != expected) incorrectResults++;	
		
	}//end for
	
	//calculate error and return as percentage
	return 100 - (incorrectResults/set.size() * 100);
}

/*******************************************************************
* Initialize Neuron Weights
********************************************************************/
void neuralNetwork::initializeWeights()
{
	double startTime = CycleTimer::currentSeconds();

	/*
	cudaMalloc(&device_output1, sizeof(double) * batchSize*nHidden);
    cudaMalloc(&input, sizeof(double) * batchSize*(nInput+1));
    cudaMalloc(&w1, sizeof(double) * (nInput+1)*nHidden);

    cudaMalloc(&device_output2, sizeof(double) * batchSize*nOutput);
    cudaMalloc(&hidden, sizeof(double) * batchSize*(nHidden+1));
    cudaMalloc(&w2, sizeof(double) * (nHidden+1)*nOutput);
    */
    

	//set weights between input and hidden 		
	//--------------------------------------------------------------------------------------------------------
	for(int i = 0; i <= nInput; i++)
	{		
		for(int j = 0; j < nHidden; j++) 
		{
			//set weights to random values
			wInputHidden[i][j] = ( (( (double)(rand()%1000)+1)/1000)/10 - 0.05);
		}
	}
	
	//set weights between input and hidden
	//--------------------------------------------------------------------------------------------------------
	for(int i = 0; i <= nHidden; i++)
	{		
		for(int j = 0; j < nOutput; j++) 
		{
			//set weights to random values
			wHiddenOutput[i][j] = ( (( (double)(rand()%1000)+1)/1000)/10 - 0.05);
		}
	}
	double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;

    printf("Time Taken Seq:%f\n", overallDuration);
}
/*******************************************************************
* Activation Function
********************************************************************/
inline double neuralNetwork::activationFunction( double x )
{
	//sigmoid function
	return 1/(1+exp(-x));
}	

void neuralNetwork::feedForwardBatch(vector<double*> patternVector) {

	for (int b = 0; b<batchSize; b++) {
	    for(int i = 0; i < nInput+1; i++) { 
            if (i!=nInput) {
                inputNeurons[b*(nInput+1) + i] = patternVector[b][i];
            }
        }
	}

	//#pragma omp parallel 
	//{
		// for (int b = 0; b<batchSize; b++) {
		//     for(int i = 0; i < nInput+1; i++) { 
  //               if (i!=nInput) {
  //                   inputNeurons[b*(nInput+1) + i] = patternVector[b][i];
  //               }
  //           }
		// }

		//#pragma omp for
		for (int b=0; b<batchSize; b++) {


		//double temp = 0.0;
		
		//#pragma omp for //schedule(static, 16)
			for(int j=0; j < nHidden; j++)
			{
				double temp = 0.0;
				//clear value
				//hiddenNeurons[b*nHidden+j] = 0;	
				//get weighted sum of pattern and bias neuron
			 	// #pragma omp parallel for reduction(+ : temp)
				for( int i=0; i <= nInput; i++ ) {
					temp += inputNeurons[b*(nInput+1) +i] * wInputHidden[i][j];
				}
				// cout << "temp: " << hiddenNeurons[j] << endl;
				//set to result of sigmoid
				hiddenNeurons[b*nHidden +j] = activationFunction( temp );			
				// cout << "output: " << hiddenNeurons[j] << endl;
			}
			
		
			// double endTime1 = CycleTimer::currentSeconds();
			// printf("Time:%f\n", endTime1 - startTime);
			//Calculating Output Layer values - include bias neuron
			//--------------------------------------------------------------------------------------------------------
			//#pragma omp for //schedule(static, 16)//reduction(+ : temp)
			for(int k=0; k < nOutput; k++)
			{
				double temp = 0.0;
				//clear value
				//outputNeurons[k*b] = 0;			
						
				//get weighted sum of pattern and bias neuron
				// #pragma omp for //reduction(+ : temp)
				for( int j=0; j <= nHidden; j++ ) {
					temp += hiddenNeurons[b*(nHidden+1)+j] * wHiddenOutput[j][k];
				}
				//set to result of sigmoid
				outputNeurons[b*nOutput +k] = activationFunction( temp );
			}
		}
	//}

/*
	dim3 blockDim(1024, 1);
    dim3 gridDim(1024);//((1024*1024) + blockDim.x - 1) / blockDim.x);
    cudaMemcpy(input, inputNeurons, sizeof(double) * batchSize*(nInput+1), cudaMemcpyHostToDevice);
    cudaMemcpy(w1, wInputHidden[0], (nInput+1)*nHidden*sizeof(double), cudaMemcpyHostToDevice);
    forward_prop_kernel_batch<<<gridDim, blockDim>>>(device_output1, input, w1, nInput+1, nHidden, batchSize);
    cudaThreadSynchronize();
    cudaMemcpy(hiddenNeurons, device_output1, batchSize*nHidden*sizeof(double), cudaMemcpyDeviceToHost);

    //w2 part
    dim3 gridDim2(nOutput*batchSize);//((1024*1024) + blockDim.x - 1) / blockDim.x);
	cudaMemcpy(hidden, hiddenNeurons, sizeof(double) * batchSize*(nHidden+1), cudaMemcpyHostToDevice);
	cudaMemcpy(w2, wHiddenOutput[0], (nHidden+1)*nOutput*sizeof(double), cudaMemcpyHostToDevice);
	forward_prop_kernel_batch<<<gridDim2, blockDim>>>(device_output2, hidden, w2, nHidden+1, nOutput,batchSize);
	cudaThreadSynchronize();
	cudaMemcpy(outputNeurons, device_output2, batchSize*nOutput*sizeof(double), cudaMemcpyDeviceToHost);
	*/

    //    dim3 gridDim2(nOutput);//((1024*1024) + blockDim.x - 1) / blockDim.x);
	
 //    cudaMemcpy(hidden, hiddenNeurons, sizeof(double) * (nHidden+1), cudaMemcpyHostToDevice);
 //    // double endTime1 = CycleTimer::currentSeconds();
    
 //    cudaMemcpy(w2, wHiddenOutput[0], (nHidden+1)*nOutput*sizeof(double), cudaMemcpyHostToDevice);
 //    // double endTime2 = CycleTimer::currentSeconds();

	// forward_prop_kernel<<<gridDim2, blockDim>>>(device_output2, hidden, w2, nHidden+1, nOutput);

	// cudaThreadSynchronize();
	// // double endTime3 = CycleTimer::currentSeconds();

	// cudaMemcpy(outputNeurons, device_output2, nOutput*sizeof(double), cudaMemcpyDeviceToHost);

}

/*******************************************************************
* Feed Forward Operation
********************************************************************/
void neuralNetwork::feedForward(double* pattern)
{
	//set input neurons to input values
	for(int i = 0; i < nInput; i++) {
		inputNeurons[i] = pattern[i];
	}

	double startTime = CycleTimer::currentSeconds();

	/*	
	dim3 blockDim(1024, 1);
    dim3 gridDim(nHidden);//((1024*1024) + blockDim.x - 1) / blockDim.x);
	
    cudaMemcpy(input, inputNeurons, sizeof(double) * (nInput+1), cudaMemcpyHostToDevice);
    // double endTime1 = CycleTimer::currentSeconds();
    
    cudaMemcpy(w1, wInputHidden[0], (nInput+1)*nHidden*sizeof(double), cudaMemcpyHostToDevice);
    // double endTime2 = CycleTimer::currentSeconds();

	forward_prop_kernel<<<gridDim, blockDim>>>(device_output1, input, w1, nInput+1, nHidden);

	cudaThreadSynchronize();
	// double endTime3 = CycleTimer::currentSeconds();

	cudaMemcpy(hiddenNeurons, device_output1, nHidden*sizeof(double), cudaMemcpyDeviceToHost);
	// double endTime4 = CycleTimer::currentSeconds();
	*/
	
	
	// double time1 = endTime1 - startTime;
	// double time2 = endTime2 - endTime1;
	// double time3 = endTime3 - endTime2;
	// double time4 = endTime4 - endTime3;

 //    printf("Time 1:%f\n", time1);
 //    printf("Time 2:%f\n", time2);
 //    printf("Time 3:%f\n", time3);
 //    printf("Time 4:%f\n", time4);


    //Calculate Hidden Layer values - include bias neuron
	//--------------------------------------------------------------------------------------------------------

	#pragma omp parallel 
	{
		double temp = 0.0;
		
		#pragma omp for //schedule(static, 16)
		for(int j=0; j < nHidden; j++)
		{
			temp = 0.0;
			//clear value
			hiddenNeurons[j] = 0;	
			//get weighted sum of pattern and bias neuron
		 	// #pragma omp parallel for reduction(+ : temp)
			for( int i=0; i <= nInput; i++ ) {
				temp += inputNeurons[i] * wInputHidden[i][j];
			}
			// cout << "temp: " << hiddenNeurons[j] << endl;
			//set to result of sigmoid
			hiddenNeurons[j] = activationFunction( temp );			
			// cout << "output: " << hiddenNeurons[j] << endl;
		}
		
	
		// double endTime1 = CycleTimer::currentSeconds();
		// printf("Time:%f\n", endTime1 - startTime);
		//Calculating Output Layer values - include bias neuron
		//--------------------------------------------------------------------------------------------------------
		#pragma omp for //schedule(static, 16)//reduction(+ : temp)
		for(int k=0; k < nOutput; k++)
		{
			temp = 0.0;
			//clear value
			outputNeurons[k] = 0;			
					
			//get weighted sum of pattern and bias neuron
			// #pragma omp for //reduction(+ : temp)
			for( int j=0; j <= nHidden; j++ ) {
				temp += hiddenNeurons[j] * wHiddenOutput[j][k];
			}
			//set to result of sigmoid
			outputNeurons[k] = activationFunction( temp );
		}
	}
	
	
	/*
	
    dim3 gridDim2(nOutput);//((1024*1024) + blockDim.x - 1) / blockDim.x);
	
    cudaMemcpy(hidden, hiddenNeurons, sizeof(double) * (nHidden+1), cudaMemcpyHostToDevice);
    // double endTime1 = CycleTimer::currentSeconds();
    
    cudaMemcpy(w2, wHiddenOutput[0], (nHidden+1)*nOutput*sizeof(double), cudaMemcpyHostToDevice);
    // double endTime2 = CycleTimer::currentSeconds();

	forward_prop_kernel<<<gridDim2, blockDim>>>(device_output2, hidden, w2, nHidden+1, nOutput);

	cudaThreadSynchronize();
	// double endTime3 = CycleTimer::currentSeconds();

	cudaMemcpy(outputNeurons, device_output2, nOutput*sizeof(double), cudaMemcpyDeviceToHost);
	// double endTime4 = CycleTimer::currentSeconds();
	*/

	double endTime3 = CycleTimer::currentSeconds();

	double time = endTime3 - startTime;

	// cout << "Forward = " << time << endl;
	
	
}

/*
void neuralNetwork::printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
*/
