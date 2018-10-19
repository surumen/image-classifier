//standard includes
#include <iostream>
#include <fstream>
#include <math.h>
#include <algorithm>
#include <omp.h>

/*
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
*/

//include definition file
#include "neuralNetworkTrainer.h"

#include "CycleTimer.h"

using namespace std;

/*******************************************************************
* constructor
********************************************************************/
neuralNetworkTrainer::neuralNetworkTrainer( neuralNetwork *nn )	:	NN(nn),
																	epoch(0),
																	learningRate(LEARNING_RATE),
																	maxEpochs(MAX_EPOCHS),
																	desiredAccuracy(DESIRED_ACCURACY),																	
																	useBatch(false),
																	trainingSetAccuracy(0),
																	validationSetAccuracy(0),
																	generalizationSetAccuracy(0)																	
{
	//create delta lists
	//--------------------------------------------------------------------------------------------------------
	deltaInputHidden = new( double*[NN->nInput + 1] );
	for ( int i=0; i <= NN->nInput; i++ ) 
	{
		deltaInputHidden[i] = new (double[NN->nHidden]);
		for ( int j=0; j < NN->nHidden; j++ ) deltaInputHidden[i][j] = 0;		
	}

	deltaHiddenOutput = new( double*[NN->nHidden + 1] );
	for ( int i=0; i <= NN->nHidden; i++ ) 
	{
		deltaHiddenOutput[i] = new (double[NN->nOutput]);			
		for ( int j=0; j < NN->nOutput; j++ ) deltaHiddenOutput[i][j] = 0;		
	}

	//create error gradient storage
	//--------------------------------------------------------------------------------------------------------
	hiddenErrorGradients = new( double[(NN->batchSize)*(NN->nHidden + 1)] );
	for (int b = 0; b<NN->batchSize; b++) {
	    for(int i = 0; i < NN->nHidden+1; i++) { 
            hiddenErrorGradients[b*(NN->nHidden+1) + i] = 0;
        }
	}
	
	outputErrorGradients = new( double[(NN->batchSize)*(NN->nOutput + 1)] );
	for (int b = 0; b<NN->batchSize; b++) {
	    for(int i = 0; i < NN->nOutput+1; i++) { 
            outputErrorGradients[b*(NN->nOutput+1) + i] = 0;
        }
	}

	// hiddenErrorGradients = new( double[NN->nHidden + 1] );
	// for ( int i=0; i <= NN->nHidden; i++ ) hiddenErrorGradients[i] = 0;
	
	// outputErrorGradients = new( double[NN->nOutput + 1] );
	// for ( int i=0; i <= NN->nOutput; i++ ) outputErrorGradients[i] = 0;
}


/*******************************************************************
* Set training parameters
********************************************************************/
void neuralNetworkTrainer::setTrainingParameters( double lR, bool batch )
{
	learningRate = lR;
	useBatch = batch;
}
/*******************************************************************
* Set stopping parameters
********************************************************************/
void neuralNetworkTrainer::setStoppingConditions( int mEpochs, double dAccuracy )
{
	maxEpochs = mEpochs;
	desiredAccuracy = dAccuracy;	
}
/*******************************************************************
* Enable training logging
********************************************************************/
void neuralNetworkTrainer::enableLogging(const char* filename, int resolution = 1)
{
	//create log file 
	if ( ! logFile.is_open() )
	{
		logFile.open(filename, ios::out);

		if ( logFile.is_open() )
		{
			//write log file header
			logFile << "Epoch,Training Set Accuracy, Generalization Set Accuracy,Training Set MSE, Generalization Set MSE" << endl;
			
			//enable logging
			loggingEnabled = true;
			
			//resolution setting;
			logResolution = resolution;
			lastEpochLogged = -resolution;
		}
	}
}
/*******************************************************************
* calculate output error gradient
********************************************************************/
inline double neuralNetworkTrainer::getOutputErrorGradient( double desiredValue, double outputValue)
{
	//return error gradient
	return outputValue * ( 1 - outputValue ) * ( desiredValue - outputValue );
}

/*******************************************************************
* calculate input error gradient
********************************************************************/
double neuralNetworkTrainer::getHiddenErrorGradient( int j )
{
	//get sum of hidden->output weights * output error gradients
	double weightedSum = 0;
	for( int k = 0; k < NN->nOutput; k++ ) {
		weightedSum += NN->wHiddenOutput[j][k] * outputErrorGradients[k];
	}

	return NN->hiddenNeurons[j] * ( 1 - NN->hiddenNeurons[j] ) * weightedSum;
}

double neuralNetworkTrainer::getHiddenErrorGradientBatch( int j, int b )
{
	//get sum of hidden->output weights * output error gradients
	double weightedSum = 0;
	for( int k = 0; k < NN->nOutput; k++ ) {
		weightedSum += NN->wHiddenOutput[j][k] * outputErrorGradients[b*k];
	}

	return NN->hiddenNeurons[j] * ( 1 - NN->hiddenNeurons[j] ) * weightedSum;
}
/*******************************************************************
* Train the NN using gradient descent
********************************************************************/
void neuralNetworkTrainer::trainNetwork( trainingDataSet* tSet )
{
	cout	<< endl << " Neural Network Training Starting: " << endl
			<< "==========================================================================" << endl
			<< " LR: " << learningRate << ", Max Epochs: " << maxEpochs << ", Batch: " << useBatch*NN->batchSize << endl
			<< " " << NN->nInput << " Input Neurons, " << NN->nHidden << " Hidden Neurons, " << NN->nOutput << " Output Neurons" << endl
			<< "==========================================================================" << endl << endl;

	//reset epoch and log counters
	epoch = 0;
	lastEpochLogged = -logResolution;
		
	//train network using training dataset for training and generalization dataset for testing
	//--------------------------------------------------------------------------------------------------------
	while (	( trainingSetAccuracy < desiredAccuracy || generalizationSetAccuracy < desiredAccuracy ) && epoch < maxEpochs )				
	{			
		//store previous accuracy
		double previousTAccuracy = trainingSetAccuracy;
		double previousGAccuracy = generalizationSetAccuracy;

		//use training set to train network
		runTrainingEpoch( tSet->trainingSet );

		// trainingSetAccuracy = NN->getSetAccuracy( tSet->trainingSet );
		//get generalization set accuracy
		generalizationSetAccuracy = NN->getSetAccuracy( tSet->generalizationSet );

		//Log Training results
		if ( loggingEnabled && logFile.is_open() && ( epoch - lastEpochLogged == logResolution ) ) 
		{
			logFile << epoch << "," << trainingSetAccuracy << "," << generalizationSetAccuracy << endl;
			lastEpochLogged = epoch;
		}
		
		//print out change in training /generalization accuracy (only if a change is greater than a percent)
		if ( ceil(previousTAccuracy) != ceil(trainingSetAccuracy) || ceil(previousGAccuracy) != ceil(generalizationSetAccuracy) ) 
		{	
			cout << "Epoch :" << epoch;
			cout << " TSet Acc:" << trainingSetAccuracy << "%" ;
			cout << " GSet Acc:" << generalizationSetAccuracy << "%" << endl;
		}
		
		//once training set is complete increment epoch
		epoch++;

	}//end while

	//get validation set accuracy
	validationSetAccuracy = NN->getSetAccuracy(tSet->validationSet);

	//log end
	logFile << epoch << "," << trainingSetAccuracy << "," << generalizationSetAccuracy << endl << endl;
	logFile << "Training Complete!!! - > Elapsed Epochs: " << epoch << " Validation Set Accuracy: " << validationSetAccuracy << endl;
			
	//out validation accuracy
	cout << endl << "Training Complete!!! - > Elapsed Epochs: " << epoch << endl;
	cout << " Validation Set Accuracy: " << validationSetAccuracy << endl << endl;
}
/*******************************************************************
* Run a single training epoch
********************************************************************/
void neuralNetworkTrainer::runTrainingEpoch( vector<dataEntry*> trainingSet )
{
	double startIter = CycleTimer::currentSeconds();
	//incorrect patterns
	double incorrectPatterns = 0;
	
	vector<double*>largePattern;
	vector<double*>largeTarget; 

	//for every training pattern
	for ( int tp = 0; tp < (int) trainingSet.size(); tp++)
	{
		largePattern.push_back(trainingSet[tp]->pattern);
		largeTarget.push_back(trainingSet[tp]->target);
		//feed inputs through network and backpropagate errors
		// double startForward = CycleTimer::currentSeconds();
		if (useBatch && ((tp == (int) trainingSet.size()-1) || (largePattern.size() == NN->batchSize))) {
			NN->feedForwardBatch( largePattern );
			backpropagateBatch( largeTarget );
			updateWeights();
			largePattern.clear();
		} else {
			NN->feedForward( trainingSet[tp]->pattern );
			backpropagate( trainingSet[tp]->target );
		}

		// double endForward = CycleTimer::currentSeconds();
	 //    double timeForward = endForward - startForward;

	 //    printf("Forward: %f\n", timeForward);

	    // double startBack = CycleTimer::currentSeconds();

		// double endBack = CycleTimer::currentSeconds();
	 //    double timeBack = endBack - startBack;

	 //    printf("Backprop: %f\n", timeBack);


		int predicted = distance(NN->outputNeurons, max_element(NN->outputNeurons, NN->outputNeurons + NN->nOutput));
		int expected = distance(trainingSet[tp]->target, max_element(trainingSet[tp]->target, trainingSet[tp]->target + NN->nOutput));
		
		if (predicted != expected) incorrectPatterns++;
			
		
	}//end for

	//if using batch learning - update the weights
	// if ( useBatch ) updateWeights();
	
	//update training accuracy
	trainingSetAccuracy = 100 - (incorrectPatterns/trainingSet.size() * 100);

	double endIter = CycleTimer::currentSeconds();
    double timeIter = endIter - startIter;

    printf("Iteration: %f\n", timeIter);

}

/*******************************************************************
* Propagate errors back through NN and calculate delta values in batches
********************************************************************/
void neuralNetworkTrainer::backpropagateBatch(vector<double*> desiredOutputsVector) {
	for (int b=0; b<NN->batchSize; b++) {
		// #pragma omp parallel
		// {
			// #pragma omp for
			for (int k = 0; k < (NN->nOutput); k++)
			{
				// cout << "TNUM " << omp_get_thread_num() << endl;
				//get error gradient for every output node
				//outputErrorGradients[k] = getOutputErrorGradient( desiredOutputsVector[b][k], NN->outputNeurons[b*k] );
				outputErrorGradients[b*k] = getOutputErrorGradient( desiredOutputsVector[b][k], NN->outputNeurons[b*k] );

				//for all nodes in hidden layer and bias neuron
				// #pragma omp for
				for (int j = 0; j <= NN->nHidden; j++) 
				{
					// if (omp_get_thread_num()) {
					// 	cout << "TNUM " << omp_get_thread_num() << endl;
					// }
					//calculate change in weight
					// #pragma omp atomic
					deltaHiddenOutput[j][k] += learningRate * NN->hiddenNeurons[b*j] * outputErrorGradients[b*k];
				}
			}
			// #pragma omp for
			for (int j = 0; j < NN->nHidden; j++)
			{
				//get error gradient for every hidden node
				hiddenErrorGradients[b*j] = getHiddenErrorGradientBatch( j, b );

				//for all nodes in input layer and bias neuron
				// #pragma omp for
				for (int i = 0; i <= NN->nInput; i++)
				{
					//calculate change in weight 
					// #pragma omp atomic
					deltaInputHidden[i][j] += learningRate * NN->inputNeurons[b*i] * hiddenErrorGradients[b*j]; 

				}
			}
		// }
	}

}

/*******************************************************************
* Propagate errors back through NN and calculate delta values
********************************************************************/
void neuralNetworkTrainer::backpropagate( double* desiredOutputs )
{
	double startBack = CycleTimer::currentSeconds();
	#pragma omp parallel
	{
		//modify deltas between hidden and output layers
		//--------------------------------------------------------------------------------------------------------
		#pragma omp for
		for (int k = 0; k < NN->nOutput; k++)
		{
			// cout << "TNUM " << omp_get_thread_num() << endl;
			//get error gradient for every output node
			outputErrorGradients[k] = getOutputErrorGradient( desiredOutputs[k], NN->outputNeurons[k] );
			
			//for all nodes in hidden layer and bias neuron
			// #pragma omp for
			for (int j = 0; j <= NN->nHidden; j++) 
			{
				// if (omp_get_thread_num()) {
				// 	cout << "TNUM " << omp_get_thread_num() << endl;
				// }
				//calculate change in weight
				if ( !useBatch ) deltaHiddenOutput[j][k] = learningRate * NN->hiddenNeurons[j] * outputErrorGradients[k];
				else deltaHiddenOutput[j][k] += learningRate * NN->hiddenNeurons[j] * outputErrorGradients[k];
			}
		}
		//modify deltas between input and hidden layers
		//--------------------------------------------------------------------------------------------------------
		#pragma omp for
		for (int j = 0; j < NN->nHidden; j++)
		{
			//get error gradient for every hidden node
			hiddenErrorGradients[j] = getHiddenErrorGradient( j );

			//for all nodes in input layer and bias neuron
			// #pragma omp for
			for (int i = 0; i <= NN->nInput; i++)
			{
				//calculate change in weight 
				if ( !useBatch ) deltaInputHidden[i][j] = learningRate * NN->inputNeurons[i] * hiddenErrorGradients[j];
				else deltaInputHidden[i][j] += learningRate * NN->inputNeurons[i] * hiddenErrorGradients[j]; 

			}
		}
	}
	

	
	
	//if using stochastic learning update the weights immediately
	if ( !useBatch ) updateWeights();

	double endBack = CycleTimer::currentSeconds();

	double timeBack = endBack - startBack;

	// cout << "Back = " << timeBack << endl;
}
/*******************************************************************
* Update weights using delta values
********************************************************************/
void neuralNetworkTrainer::updateWeights()
{
	//input -> hidden weights
	//--------------------------------------------------------------------------------------------------------
	for (int i = 0; i <= NN->nInput; i++)
	{
		for (int j = 0; j < NN->nHidden; j++) 
		{
			//update weight
			NN->wInputHidden[i][j] += deltaInputHidden[i][j];	
			
			//clear delta only if using batch (previous delta is needed for momentum
			if (useBatch) deltaInputHidden[i][j] = 0;				
		}
	}
	
	//hidden -> output weights
	//--------------------------------------------------------------------------------------------------------
	for (int j = 0; j <= NN->nHidden; j++)
	{
		for (int k = 0; k < NN->nOutput; k++) 
		{					
			//update weight
			NN->wHiddenOutput[j][k] += deltaHiddenOutput[j][k];
			
			//clear delta only if using batch (previous delta is needed for momentum)
			if (useBatch)deltaHiddenOutput[j][k] = 0;
		}
	}
}
