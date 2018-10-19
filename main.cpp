/*******************************************************************
* Neural Network Training Example
* ------------------------------------------------------------------
* Bobby Anguelov - takinginitiative.wordpress.com (2008)
* MSN & email: banguelov@cs.up.ac.za
*********************************************************************/

//standard libraries
#include <iostream>
#include <ctime>
#include <stdlib.h>

//custom includes
#include "neuralNetwork.h"
#include "neuralNetworkTrainer.h"
// #include <cublas_v2.h>

#include "CycleTimer.h"

//use standard namespace
using namespace std;

int main()
{	
	double startTime = CycleTimer::currentSeconds();
	//seed random number generator
	srand( (unsigned int) time(0) );
	
	//create data set reader and load data file
	dataReader d;
	d.loadDataFile("mnist_train.csv",784,10);
	d.setNumSets(1);
	// d.setCreationApproach( STATIC, 10 );	

	//create neural network
	neuralNetwork nn(784,200,10, 1);
	nn.printCudaInfo();

	//create neural network trainer
	neuralNetworkTrainer nT( &nn );
	nT.setTrainingParameters(0.5, false);
	nT.setStoppingConditions(100, 100);
	nT.enableLogging("log.csv", 5);
	
	//train neural network on data sets
	for (int i=0; i < d.getNumTrainingSets(); i++ )
	{
		nT.trainNetwork( d.getTrainingDataSet() );
	}

	//save the weights
	nn.saveWeights("weights.csv");

	double endTime = CycleTimer::currentSeconds();
    double totalTime = endTime - startTime;

    cout << "TOTAL TIME: " << totalTime << endl;
		
	cout << endl << endl << "-- END OF PROGRAM --" << endl;
	char c; cin >> c;
	return 0;
}
