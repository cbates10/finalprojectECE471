#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>
#include "Matrix.h"
#include "Pr.h"
#include <string>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <chrono>


/* Time calculations made using the chrono library from c++ 14 */

using namespace std;
using namespace std::chrono;

/* Index class used to keep index values grouped together with distance values */
class index{
  public:
	double get_dist() { return distsqr; }
	int get_line() { return line; }
	bool operator<(const index &rhs) const{ return distsqr < rhs.distsqr; }
	index(double dist, int ind) { distsqr = dist; line = ind; }
  private: 
	double distsqr;
	int line;
};

/* Initial normalization of training data */
void normalizeinit(Matrix &values, Matrix &means, vector<double> &std){
  double stdDev = 0;
  means = mean(values, values.getCol());
  for(int j = 0; j < values.getCol(); j++){
    for(int i = 0; i < values.getRow(); i++){
      stdDev += pow(values(i,j) - means(j,0), 2);
    }
	stdDev /= (values.getRow());
    stdDev = sqrt(stdDev);
	std.push_back(stdDev);
    for(int i2 = 0; i2 < values.getRow(); i2++){
      values(i2,j) = (values(i2,j) - means(j,0))/stdDev;
	}
	stdDev = 0;
  }
}

/* Normalization of testing data */
void normalize(Matrix &values, Matrix &means, vector<double> &std){
  for(int j = 0; j < values.getCol(); j++){
    for(int i = 0; i < values.getRow(); i++){
      values(i,j) = (values(i,j) - means(j,0))/std[j];
	}
  }
}
/* kNN classifier implementation. Arguments are k (the number of nearest neighbors to check), test (A matrix of testing values,
 * train (A matrix of testing values), outcomete (a vector of class values that correspond to the indices of the testing set),
 * outcometr (a vector of class values that correspond to the indices of the training set), numclass (the number of different classes),
 * distance (the distance degree metric for minkowski distance, by default it is set to 2) */
void kNN(int k, Matrix test, Matrix train, vector<int> outcomete, vector<int> outcometr, int numclass, int distance = 2){
  vector<index> distsqr;
  double tmpd;
  int size = 0, outcome;
  int TP = 0, FP = 0, TN = 0, FN = 0;
  int kclass, tmpi = 0, rowdex = 0;
  int hitc1 = 0, hitc2 = 0;
  vector<int> hits;
  int correct = 0, incorrect = 0;
  double accuracy;
  bool change;
  for(int z = 0; z < test.getRow(); z++){
	distsqr.clear();
	size = 0;
    for(int i = 0; i < train.getRow(); i++){
	  tmpd = 0;
	  change = false;
      if(size < k){
		// Push back the distances for the first k samples
        for(int j = 0; j < train.getCol(); j++){
          tmpd += abs(pow(test(z,j) - train(i,j), distance));
	    }
	    distsqr.push_back(index(abs(tmpd), i));
	    size++;
	  } else {
		// Patrial distance. Only resort vector if the current largest distance is swapped 
	    for(int j = 0; j < train.getCol(); j++){
          tmpd += abs(pow(test(z,j) - train(i,j), distance));
		  if(tmpd > distsqr[k - 1].get_dist()){
		  	break;
		  }
		  if(j == train.getCol() - 1){
            distsqr[k - 1] = index(tmpd, i);
		    change = true;
		  } 
	    } 
	  }
	  // Sort vector after the first k items are calculated or the last item has been changed out 
      if(size == k - 1 || change){
	    sort(distsqr.begin(), distsqr.end());
	  }
    }
	// Calculate accuracy as well as the number of true positives, negative, false positives, and negatives
	hits.resize(numclass, 0);
	for(int i = 0; i < distsqr.size(); i++){
      hits[outcometr[distsqr[i].get_line()]]++;
	}
	for(int i = 0; i < numclass; i++){
      if(tmpi < hits[i]){
        tmpi = hits[i];
		kclass = i;
	  }
	}
	if(outcomete[rowdex++] == kclass){
	  if(kclass == 0)
		TP++;
	  else
		TN++;
	  correct++;
	}
	else{
	  incorrect++;
      if(kclass == 0)
		FP++;
	  else
		FN++;
	}
  }
  accuracy = (double)correct/(correct + incorrect);
  cout << "Accuracy " << accuracy << endl;
  cout << "TP/TN " << TP << " " << TN << endl;
  cout << "FP/FN " << FP << " " << FN << endl;
}

/* The initialization of 10 fold cross validation. A matrix of the training data is provided and a vector
 * of vectors detailing the 10 group indices Output contains the accuracy information for each validation set
 * for each k value between 1 and sqrt(n)*/
void cross_kNN(Matrix traininit, vector< vector<int> > cross_val){
  int sizerow = 0;
  int curindex = 0;
  vector<double> std;
  vector<int> outcometr;
  vector<int> outcomete;
  Matrix means;
  Matrix train = subMatrix(traininit,0,0,traininit.getRow()-1,traininit.getCol()-2);
  normalizeinit(train, means, std); 
  // 10 matrix groups
  Matrix x1(cross_val[0].size(), train.getCol());
  Matrix x2(cross_val[1].size(), train.getCol());
  Matrix x3(cross_val[2].size(), train.getCol());
  Matrix x4(cross_val[3].size(), train.getCol());
  Matrix x5(cross_val[4].size(), train.getCol());
  Matrix x6(cross_val[5].size(), train.getCol());
  Matrix x7(cross_val[6].size(), train.getCol());
  Matrix x8(cross_val[7].size(), train.getCol());
  Matrix x9(cross_val[8].size(), train.getCol());
  Matrix x10(cross_val[9].size(), train.getCol());

  //Assemble matrices by crossval index
  for(int i = 0; i < x1.getRow(); i++){
    for(int j = 0; j < x1.getCol(); j++){
      x1(i,j) = train(cross_val[0][i] - 1,j);
	}
  }
  for(int i = 0; i < x2.getRow(); i++){
    for(int j = 0; j < x2.getCol(); j++){
      x2(i,j) = train(cross_val[1][i] - 1,j);
	}
  } 
  for(int i = 0; i < x3.getRow(); i++){
    for(int j = 0; j < x3.getCol(); j++){
      x3(i,j) = train(cross_val[2][i] - 1,j);
	}
  }
  for(int i = 0; i < x4.getRow(); i++){
    for(int j = 0; j < x4.getCol(); j++){
      x4(i,j) = train(cross_val[3][i] - 1,j);
	}
  }
  for(int i = 0; i < x5.getRow(); i++){
    for(int j = 0; j < x5.getCol(); j++){
      x5(i,j) = train(cross_val[4][i] - 1,j);
	}
  }
  for(int i = 0; i < x6.getRow(); i++){
    for(int j = 0; j < x6.getCol(); j++){
      x6(i,j) = train(cross_val[5][i] - 1,j);
	}
  }
  for(int i = 0; i < x7.getRow(); i++){
    for(int j = 0; j < x7.getCol(); j++){
      x7(i,j) = train(cross_val[6][i] - 1,j);
	}
  }
  for(int i = 0; i < x8.getRow(); i++){
    for(int j = 0; j < x8.getCol(); j++){
      x8(i,j) = train(cross_val[7][i] - 1,j);
	}
  }
  for(int i = 0; i < x9.getRow(); i++){
    for(int j = 0; j < x9.getCol(); j++){
      x9(i,j) = train(cross_val[8][i] - 1,j);
	}
  }
  for(int i = 0; i < x10.getRow(); i++){
    for(int j = 0; j < x10.getCol(); j++){
      x10(i,j) = train(cross_val[9][i] - 1,j);
	}
  }
  // Assemble the submatrix groups into a vector for easy indexing
  vector<Matrix> sections;
  sections.push_back(x1);
  sections.push_back(x2);
  sections.push_back(x3);
  sections.push_back(x4);
  sections.push_back(x5);
  sections.push_back(x6);
  sections.push_back(x7);
  sections.push_back(x8);
  sections.push_back(x9);
  sections.push_back(x10);
 
  for(int i = 0; i < 10; i++){
	sizerow = 0;
	// Skip over the testing matrix 
    for(int j = 0; j < 10; j++){
	  if(j == i)
		continue;
      sizerow += cross_val[j].size();
	}
	Matrix trainfold(sizerow,x1.getCol());
	curindex = 0;
	// Assemble all the values to be put into the training matrix
	for(int j = 0; j < 10; j++){
	  if(j == i) 
		continue;
      for(int c = 0; c < sections[j].getRow(); c++){
		for(int b = 0; b < sections[j].getCol(); b++){
          trainfold(curindex,b) = sections[j](c,b);
		}
		outcometr.push_back(traininit(cross_val[j][c], traininit.getCol()-1));
		curindex++;
	  }
	}
	// Assemble the testing matrix
	Matrix testfold(cross_val[i].size(),x1.getCol());
	curindex = 0;
	for(int c = 0; c < sections[i].getRow(); c++){
	  for(int b = 0; b < sections[i].getCol(); b++){
        testfold(curindex,b) = sections[i](c,b);
	  }
      outcomete.push_back(traininit(cross_val[i][c], traininit.getCol()-1));
	  curindex++;
	}
	cout << "Fglass group as validation set " << i << endl;
	for(int j = 1; j < sqrt(traininit.getRow()); j++){
      cout << j << " ";
	  kNN(j, testfold, trainfold, outcomete, outcometr, 7, 2);
	}
	cout << endl;
  }
}

/* Case I Euclidean min discriminant function */
int Euclidean_min(Matrix values, Matrix mean1, Matrix mean2, double prior1, double prior2, double var){
  double result;
  double class1, class2;
  Matrix mat_1 = transpose((1/var)*mean1)->*transpose(values)+(-1/(2*var))*transpose(mean1)->*mean1 + log(prior1);
  class1 = mat_1(0,0);
  Matrix mat_2 = transpose((1/var)*mean2)->*transpose(values)+(-1/(2*var))*transpose(mean2)->*mean2 + log(prior2);
  class2 = mat_2(0,0);
  if(class1 > class2)
	result = 0;
  else 
	result = 1;
  return result;
}

/* Case II Mahalanobis discriminant function */
int Mahalanobis_min(Matrix values, Matrix mean1, Matrix mean2, Matrix covariance, double prior1, double prior2){
  double result;
  double class1, class2;
  Matrix mat_1 = transpose(covariance->*mean1)->*transpose(values) + (-0.5)*transpose(mean1)->*covariance->*(mean1) + log(prior1);
  class1 = mat_1(0,0);
  Matrix mat_2 = transpose(covariance->*mean2)->*transpose(values) + (-0.5)*transpose(mean2)->*covariance->*(mean2) + log(prior2);
  class2 = mat_2(0,0);
  if(class1 > class2)
	result = 0;
  else
	result = 1;
  return result;
}

/* Case III Mahalanobis discriminant function, no assumptions on covariance values */
int Hyper_Quadric(Matrix values, Matrix mean1, Matrix mean2, Matrix icovariance1, Matrix icovariance2, double prior1, double prior2, double det1, double det2){
  double result;
  double class1, class2;
  Matrix mat_1 = values->*((-0.5)*icovariance1)->*transpose(values) + transpose(icovariance1->*mean1)->*transpose(values) + (-0.5)*transpose(mean1)->*icovariance1->*mean1 -(0.5)*log(det1) + log(prior1);
  class1 = mat_1(0,0);
  Matrix mat_2 = values->*((-0.5)*icovariance2)->*transpose(values) + transpose(icovariance2->*mean2)->*transpose(values) + (-0.5)*transpose(mean2)->*icovariance2->*mean2 -(0.5)*log(det2) + log(prior2);
  class2 = mat_2(0,0);
  if(class1 > class2)
    result = 0;
  else 
	result = 1;
  return result;
}

/* Solves for classication accuracy for each discriminant function. Arguments are the means, class matrices, and testing data set */
void solve_accuracy(Matrix &values, Matrix &mat_c1, Matrix &mat_c2, Matrix &mean_c1, Matrix &mean_c2, int c1_size1, int c2_size1, double probc1, double probc2){
  Matrix line;
  Matrix Sigma1 = cov(mat_c1, mat_c1.getCol());
  Matrix Sigma2 = cov(mat_c2, mat_c2.getCol());

  int first = 0;

  int correct = 0, incorrect = 0, outcome = 0;
  int TP = 0, TN = 0, FP = 0, FN = 0;
  double accuracy = 0;
  high_resolution_clock::time_point start = high_resolution_clock::now();
  for(int i = 0; i < c1_size1 + c2_size1; i++){
    line = subMatrix(values, i, 0, i, values.getCol() - 1);
	outcome = Euclidean_min(line, mean_c1, mean_c2, probc1, probc2, Sigma2(0,0) * Sigma2(0,0)); 
	if(outcome == 0 && i < c1_size1){
	  correct++;
	  TP++;
	}
	else if(outcome == 1 && i >= c1_size1){
	  correct++;
	  TN++;
	}
	else{
	  incorrect++;
	  if(i < c1_size1)
		FN++;
	  else
		FP++; 
	}
  }
  high_resolution_clock::time_point stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop -start).count();
  // Accuracy, sensitivity, specificity, precision and recall calculations 
  accuracy = (double) correct/(correct+incorrect);
  cout << "Duration " << duration << endl;
  cout << "Euclidean accuracy " <<  accuracy << endl;
  cout << "Sensitivity and specificty " << setprecision(7) << fixed << (double)TP/(TP+FN) << " " << (double)TN/(TN+FP) << endl;
  cout << "Precision and Recall " << (double)TP/(TP+FP) << " " << (double)TP/(TP+FN) << endl;
  cout << "TP " << TP << endl;
  cout << "FN " << FN << endl;
  cout << "TN " << TN << endl;
  cout << "FP " << FP << endl;
  // Reset values to 0
  correct = 0;
  incorrect = 0;
  TP = 0;
  TN = 0;
  FP = 0;
  FN = 0;
  
  Matrix inverseSigma1 = inverse(cov(mat_c1,mat_c1.getCol()));
  start = high_resolution_clock::now();
  for(int i = 0; i < c1_size1 + c2_size1; i++){
    line = subMatrix(values, i, 0, i, values.getCol() - 1);
	outcome = Mahalanobis_min(line, mean_c1, mean_c2, inverseSigma1, probc1, probc2);
	if(outcome == 0 && i < c1_size1){
	  correct++;
      TP++;
	}
	else if(outcome == 1 && i >= c1_size1){
	  correct++;
      TN++;
	}
	else{
	  incorrect++;
      if(i < c1_size1)
		FN++;
	  else
		FP++;
	}
  }
  stop = high_resolution_clock::now();
  duration = duration_cast<microseconds>(stop - start).count();
  accuracy = (double) correct/(correct+incorrect);
  cout << "Mahalanobis accuracy " << accuracy << endl;
  cout << "Duration " << duration << endl;
  cout << "Sensitivity and specificity " << setprecision(7) << fixed << (double)TP/(TP+FN) << "   " << (double)TN/(TN+FP) << endl;
  cout << "Precision and Recall " << setprecision(7) << fixed << (double)TP/(TP+FP) << "   " << (double)TP/(TP+FN) << endl;
  cout << "TP " << TP << endl;
  cout << "FN " << FN << endl;
  cout << "TN " << TN << endl;
  cout << "FP " << FP << endl;
  correct = 0;
  incorrect = 0;
  TP = 0;
  TN = 0;
  FP = 0;
  FN = 0;
  double detSigma1 = det(cov(mat_c1, mat_c1.getCol()));
  double detSigma2 = det(cov(mat_c2, mat_c2.getCol()));
  Matrix inverseSigma2 = inverse(cov(mat_c2, mat_c2.getCol()));
  start = high_resolution_clock::now();
  for(int i = 0; i < c1_size1 + c2_size1; i++){
    line = subMatrix(values, i, 0, i, values.getCol() - 1);
	outcome = Hyper_Quadric(line, mean_c1, mean_c2, inverseSigma1, inverseSigma2, probc1, probc2, detSigma1, detSigma2);
	if(outcome == 0 && i < c1_size1){
	  correct++;
      TP++;
	}
	else if(outcome == 1 && i >= c1_size1){
	  correct++;
      TN++;
	}
	else{
	  incorrect++;
      if(i < c1_size1)
		FN++;
	  else
		FP++;
	}
  }
  stop = high_resolution_clock::now();
  duration = duration_cast<microseconds>(stop-start).count();
  accuracy = (double) correct/(correct+incorrect);
  cout << "Accuracy hyper quadric " << accuracy << endl;
  cout << "Duration " << duration << endl;
  cout << "Sensitivity and specificity " << setprecision(7) << fixed << (double)TP/(TP+FN) << " " << (double)TN/(TN+FP) << endl;
  cout << "Precision and Recall " << setprecision(7) << fixed << (double)TP/(TP+FP) << "   " << (double)TP/(TP+FN) << endl;
  cout << "TP " << TP << endl;
  cout << "FN " << FN << endl;
  cout << "TN " << TN << endl;
  cout << "FP " << FP << endl; 
  return;
}

int main(int argc, char **argv){
  struct stat buf;
  struct dirent *de, *subde;
  DIR *d, *subd;
  char cbuf[1000];
  string sbuf, line;
  ifstream fin;
  float tmpf;
  vector< double > std;
  vector< float > feat;
  vector< vector<float> > wordfeat;
  vector< vector< vector<float> > > features;
  vector<Matrix> traindat;
  vector<Matrix> train_means;
  Matrix traindat_total;
  Matrix means;
  int count = 0;

  d = opendir(argv[1]);
  for(de = readdir(d); de != NULL; de = readdir(d)){
	sbuf = string(de->d_name);
    if(sbuf != "." && sbuf != ".."){
	  sprintf(cbuf, "%s/%s", argv[1], de->d_name);
	  cout << "Processing " << cbuf << endl;
	  subd = opendir(cbuf);
	  for(subde = readdir(subd); subde != NULL; subde = readdir(subd)){
		sprintf(cbuf, "%s/%s/%s", argv[1], de->d_name, subde->d_name);
		fin.open(cbuf);
		if(!fin.is_open()){
          perror(cbuf);
		  exit(1);
		}
		while(getline(fin, line)){
          stringstream ss(line);
		  while(ss >> tmpf){
            feat.push_back(tmpf);
		  }
		}
		if(!feat.empty())
		  wordfeat.push_back(feat);
		feat.clear();
		fin.close();
	  }
	  closedir(subd);
	  features.push_back(wordfeat);
	  wordfeat.clear();
    }
  }
  traindat_total = Matrix(count, features[0][0].size());
  count = 0;
  for(int i = 0; i < features.size(); i++){
    for(int j = 0; j < features[i].size(); j++){
	  for(int z = 0; z < features[i][i].size(); z++){
		traindat_total(count,z) = features[i][j][z];
	  }
	  count++;
	}
  }
  // TODO read in testing data 
  normalizeinit(traindat_total, means, std);
  // TODO normalize testing data
  count = 0;
  for(int i = 0; i < features.size(); i++){
    traindat.push_back(subMatrix(traindat_total, count, 0, count + features[i].size() - 1, traindat_total.getCol() - 1));
	count += features[i].size();
  }
  for(int i = 0; i < traindat.size(); i++){
    train_means.push_back(mean(traindat[i],traindat[i].getCol()));
  }

  /* PRINCIPAL COMPONENT ANALYSIS */

  Matrix Sigma = cov(traindat_total, traindat_total.getCol());

  Matrix eigvec(traindat_total.getCol(), traindat_total.getCol());
  Matrix eigval(traindat_total.getCol(), 1);

  jacobi(Sigma, eigval, eigvec);

  cout << "Eigenvalues : " << endl << eigval << endl << "Eigenvectors : " << endl << eigvec << endl;

  int j = Sigma.getRow() -1; //Index value of row
  int eigsum = 0, eigsum2 = 0;
  for(int i = 0; i < Sigma.getRow(); i++){
    eigsum += eigval(i,0);
  }
  
  vector<int> eigremoved;
  double tmpval, error_val = 0;
  while(error_val < 0.8){
    tmpval = error_val;
	eigsum2 += eigval(j, 0);
    error_val = eigsum2/eigsum;
	eigremoved.push_back(j--);
  }

  eigremoved.pop_back();
  error_val = tmpval;
  cout << "eigenvalue error tolerance" << endl << error_val << endl;
  cout << "Eigenvalues removed " << endl;
  for(int i = 0; i < eigremoved.size(); i++){
    cout << eigval(eigremoved[i],0) << endl;
  }

  Matrix PCA_eigvec = subMatrix(eigvec, 0, 0, eigvec.getRow() -1, eigremoved.back() - 1);
  cout << "PCA eigenvalue matrix " << endl << PCA_eigvec << endl;

  /* PCA VALUES USED FOR CALCULATIONS */
  Matrix tX = transpose(transpose(PCA_eigvec)->*transpose(traindat_total));
  // TODO PCA on testing data
  vector<Matrix> tmats;
  for(int i = 0; i < traindat.size(); i++){
    tmats.push_back(transpose(transpose(PCA_eigvec)->*transpose(traindat[i])));
  }
  vector<Matrix> tmeans;
  for(int i = 0; i < tmats.size(); i++){
    //tmeans.push_back(mean(tmats[i], tmats[i].getCol()));
  }
}
