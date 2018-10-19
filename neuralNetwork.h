/*******************************************************************
* Basic Feed Forward Neural Network Class
* ------------------------------------------------------------------
********************************************************************/

#ifndef NNetwork
#define NNetwork

#include "dataReader.h"
// #include <cublas_v2.h>

class neuralNetworkTrainer;

class neuralNetwork
{
	//class members
	//--------------------------------------------------------------------------------------------
private:

	//number of neurons
	int nInput, nHidden, nOutput, batchSize;
	
	//neurons
	float* inputNeurons;
	float* hiddenNeurons;
	float* outputNeurons;

	//weights
	float** wInputHidden;
	float** wHiddenOutput;

	float *device_output1;
	float *input;
	float *w1;

	float *device_output2;
	float *hidden;
	float *w2;

	// cublasHandle_t handle;

	//Friends
	//--------------------------------------------------------------------------------------------
	friend neuralNetworkTrainer;
	
	//public methods
	//--------------------------------------------------------------------------------------------

public:

	//constructor & destructor
	neuralNetwork(int numInput, int numHidden, int numOutput, int batchSize);
	~neuralNetwork();

	//weight operations
	bool saveWeights(char* outputFilename);
	double getSetAccuracy( std::vector<dataEntry*>& set );

	void printCudaInfo();

	//private methods
	//--------------------------------------------------------------------------------------------

private: 

	void initializeWeights();
	inline float activationFunction( float x );
	void feedForward( float* pattern );
	void feedForwardBatch(std::vector<float*> patternVector);
	
};

#endif
