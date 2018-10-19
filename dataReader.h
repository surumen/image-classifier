/*******************************************************************
* CSV Data File Reader and Training Set Creator
* ------------------------------------------------------------------
********************************************************************/

#ifndef _DATAREADER
#define _DATAREADER

//include standard header files
#include <vector>
#include <string>

/*******************************************************************
* stores a data item
********************************************************************/
class dataEntry
{
public:	
	
	float* pattern;	//input patterns
	float* target;		//target result

public:	

	dataEntry(float* p, float* t): pattern(p), target(t) {}
		
	~dataEntry()
	{				
		delete[] pattern;
		delete[] target;
	}

};

/*******************************************************************
* Training Sets Storage - stores shortcuts to data items
********************************************************************/
class trainingDataSet
{
public:

	std::vector<dataEntry*> trainingSet;
	std::vector<dataEntry*> generalizationSet;
	std::vector<dataEntry*> validationSet;

	trainingDataSet(){}
	
	void clear()
	{
		trainingSet.clear();
		generalizationSet.clear();
		validationSet.clear();
	}
};

//data reader class
class dataReader
{
	
//private members
//----------------------------------------------------------------------------------------------------------------
private:

	//data storage
	std::vector<dataEntry*> data;
	int nInputs;
	int nTargets;

	//current data set
	trainingDataSet tSet;

	//total number of dataSets
	int numTrainingSets;
	int trainingDataEndIndex;
	
//public methods
//----------------------------------------------------------------------------------------------------------------
public:

	dataReader(): numTrainingSets(-1) {}
	~dataReader();
	
	bool loadDataFile( const char* filename, int nI, int nT );
	void setNumSets(int numSets);
	int getNumTrainingSets();	
	
	trainingDataSet* getTrainingDataSet();
	std::vector<dataEntry*>& getAllDataEntries();

//private methods
//----------------------------------------------------------------------------------------------------------------
private:
	
	void createDataSet();	
	void processLine( std::string &line );
};

#endif
