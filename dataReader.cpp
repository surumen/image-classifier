//include definition file
#include "dataReader.h"

#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <algorithm>
#include <string.h>

using namespace std;

/*******************************************************************
* Destructor
********************************************************************/
dataReader::~dataReader()
{
	//clear data
	for (int i=0; i < (int) data.size(); i++ ) delete data[i];		
	data.clear();		 
}
/*******************************************************************
* Loads a csv file of input data
********************************************************************/
bool dataReader::loadDataFile( const char* filename, int nI, int nT )
{
	//clear any previous data		

	for (int i=0; i < (int) data.size(); i++ ) delete data[i];		
	data.clear();
	tSet.clear();
	
	//set number of inputs and outputs
	nInputs = nI;
	nTargets = nT;

	//open file for reading
	fstream inputFile;
	inputFile.open(filename, ios::in);	

	if ( inputFile.is_open() )
	{
		string line = "";
		
		//read data
		while ( !inputFile.eof() )
		{
			getline(inputFile, line);				
			
			//process line
			if (line.length() > 2 ) processLine(line);
		}		
		
		//shuffle data
		random_shuffle(data.begin(), data.end());

		//split data set
		trainingDataEndIndex = (int) ( 0.85 * data.size() );
		int gSize = (int) ( ceil(0.05 * data.size()) );
		int vSize = (int) ( data.size() - trainingDataEndIndex - gSize );
							
		//generalization set
		for ( int i = trainingDataEndIndex; i < trainingDataEndIndex + gSize; i++ ) tSet.generalizationSet.push_back( data[i] );
				
		//validation set
		for ( int i = trainingDataEndIndex + gSize; i < (int) data.size(); i++ ) tSet.validationSet.push_back( data[i] );
		
		//print success
		cout << "Input File: " << filename << "\nRead Complete: " << data.size() << " Patterns Loaded"  << endl;

		//close file
		inputFile.close();
		
		return true;
	}
	else 
	{
		cout << "Error Opening Input File: " << filename << endl;
		return false;
	}
}

/*******************************************************************
* Processes a single line from the data file
********************************************************************/
void dataReader::processLine( string &line )
{
	//create new pattern and target
	float* pattern = new float[nInputs];
	float* target = new float[nTargets];
	
	//store inputs		
	char* cstr = new char[line.size()+1];
	char* t;
	strncpy(cstr, line.c_str(), line.size() + 1);

	//tokenise
	int i = 0;
    char* nextToken = NULL;
	t=strtok(cstr, ",");
	
	while ( t!=NULL && i < (nInputs + 1))
	{	
		if ( i == 0 ) {
			target[atoi(t)] = 1.0f;
			for (int j = 0; j<nTargets; j++) {
				if (j!=atoi(t)) {
					target[j] = 0.0f;
				}
			}
		}
		else pattern[i-1] = atof(t) / 256.0f;

		//move token onwards
		t = strtok(NULL,",");
		i++;			
	}


	//add to records
	data.push_back( new dataEntry( pattern, target ) );
}
/*******************************************************************
* Selects the data set creation approach
********************************************************************/
void dataReader::setNumSets( int numSets)//int approach, float param1, float param2 )
{
	//only 1 data set
	numTrainingSets = numSets;

}

/*******************************************************************
* Returns number of data sets created by creation approach
********************************************************************/
int dataReader::getNumTrainingSets()
{
	return numTrainingSets;
}
/*******************************************************************
* Get data set created by creation approach
********************************************************************/
trainingDataSet* dataReader::getTrainingDataSet()
{		
	createDataSet();
	return &tSet;
}
/*******************************************************************
* Get all data entries loaded
********************************************************************/
vector<dataEntry*>& dataReader::getAllDataEntries()
{
	return data;
}

/*******************************************************************
* Create a static data set (all the entries)
********************************************************************/
void dataReader::createDataSet()
{
	//training set
	for ( int i = 0; i < trainingDataEndIndex; i++ ) tSet.trainingSet.push_back( data[i] );		
}

