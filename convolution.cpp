#include <iostream>
#include <climits>
#include <algorithm>
#include <vector>
#include <cstring>
#include <fstream>
#include <cmath>
#include "imagproc.h"
// #include <bits/stdc++.h>
using namespace std;

int main(int argc,char* argv[]){
            //cout<<argv[1]<<" "<<argv[2]<<" "<<argv[3]<<argv[4]<< argv[5]<<endl;
          if(!strcmp(argv[1],"convolution")){
              if(argc!=6)
              {
              	cout<<"Error: convolution requires 4 inputs namely input_filename input_rowsize output_filename output_rowsize"<<endl;;
              	return 0;
              }
              vector<vector<float> > input;
              vector<vector<float> > kernel;
              if(take_input(kernel,argv[4],argv[5])&&take_input(input,argv[2],argv[3])){
                  display(convolution_withoutpadding(input,kernel));
              }
          }
          else if(!strcmp(argv[1],"convolution_withpadding")){
              vector<vector<float> > input;
              int padsize = stoi(argv[2]);
              vector<vector<float> > kernel;
              if(argc!=7)
              {
              	cout<<"Error: convolution_withpadding requires 5 inputs namely padsize input_filename input_rowsize output_filename output_rowsize"<<endl;;
              	return 0;
              }
              if(take_input(input,argv[3],argv[4])&&take_input(kernel,argv[5],argv[6]))
                   display(convolution_withpadding(padsize,input,kernel));
              
          }
          else if(!strcmp(argv[1],"convolution_matrixmult")){
              vector<vector<float> > kernel;
              vector<vector<float> > input;
              if(argc!=6)
              {
              	cout<<"Error: convolution_withpadding requires 4 inputs namely input_filename input_rowsize output_filename output_rowsize"<<endl;;
              	return 0;
              }
              if(take_input(input,argv[2],argv[3])&&take_input(kernel,argv[4],argv[5]))
                  display(convolution_withoutpadding_matrixmult(input,kernel));
              
          }
          else if(!strcmp(argv[1],"convolution_withpadding_matrixmult")){
             vector<vector<float> > input;
              int padsize = stoi(argv[2]);
              vector<vector<float> > kernel;
              if(argc!=7)
              {
              	cout<<"Error: convolution_withpadding requires 5 inputs namely padsize input_filename input_rowsize output_filename output_rowsize"<<endl;;
              	return 0;
              }
              if(take_input(input,argv[3],argv[4])&&take_input(kernel,argv[5],argv[6]))
                     display(convolution_withpadding_matrixmult(padsize,input,kernel));
              
          }
          else if(!strcmp(argv[1],"softmax")){
            vector<float> input;
            ifstream file;
            if(argc!=2)
              {
              	cout<<"softmax requires a input namely input_filename"<<endl;;
              	return 0;
              }
	        file.open(argv[2]);
            if(file){
          	        string x;
          	        file>>x;
                      while (x!="\0"){
                      	input.push_back(stoi(x));
                      	file>>x;
                      }
                      disp(softmax(input));
                }
            else
                cout << "Given File with file name "<<argv[2] <<" not found"<< endl;
             
          }
          else if(!strcmp(argv[1],"sigmoid")){
          	 vector<float> input;
            ifstream file;
             if(argc!=2)
              {
              	cout<<"sigmoid requires a input namely input_filename"<<endl;;
              	return 0;
              }
	        file.open(argv[2]);
	          if(file){
                    string x;
        	          file >> x;
                    while (x!="\0"){
                    	input.push_back(stoi(x));
                    	file >> x;
                    }
                    disp(sigmoid(input));
                  }
            else{
                cout << "Given File with file name "<< argv[2] <<" not found"<< endl;

            }
          }
          else if(!strcmp(argv[1],"max_pooling")){
            vector<vector<float> > input;
             if(argc!=4)
              {
              	cout<<"max_pooling requires 2 input namely input_filename input_rowsize"<<endl;;
              	return 0;
              }
            if(take_input(input,argv[2],argv[3]))
            {
                display(max_pooling(input));
            }
          }
          else if(!strcmp(argv[1],"average_pooling")){
          	vector<vector<float> > input;
          	if(argc!=4)
              {
              	cout<<"average_pooling requires 2 input namely input_filename input_rowsize"<<endl;;
              	return 0;
              }
            if(take_input(input,argv[2],argv[3])){
                display(average_pooling(input));
            }
          }
          else if(!strcmp(argv[1],"relu_activation")){
              vector<vector<float> > inp;
              if(argc!=4)
              {
              	cout<<"relu_activation requires 2 input namely input_filename input_rowsize"<<endl;;
              	return 0;
              }
              if(take_input(inp,argv[2],argv[3]))
              {
                display(relu_activation(inp));
              }
          }
          else if(!strcmp(argv[1],"tanh_activation")){
              vector<vector<float> > inp;
               if(argc!=4)
              {
              	cout<<"tanh_activation requires 2 input namely input_filename input_rowsize"<<endl;;
              	return 0;
              }
              if(take_input(inp,argv[2],argv[3])){
                  display(tanh_activation(inp));
               }
          }
          else{
            cout << "Input type is not a given function. Please type from one of the functions from below" << endl;
            cout << "convolution input_filename input_rowsize output_filename output_rowsize" << endl;
            cout << "convolution_withpadding padsize input_filename input_rowsize output_filename output_rowsize" << endl;
            cout << "convolution_matrixmult input_filename input_rowsize output_filename output_rowsize" << endl;
            cout << "convolution_withpadding_matrixmult padsize input_filename input_rowsize output_filename output_rowsize" << endl;
            cout << "softmax input_filename" << endl;
            cout << "sigmoid input_filename" << endl;
            cout << "max_pooling input_filename input_rowsize" << endl;
            cout << "average_pooling input_filename input_rowsize" << endl;
            cout << "relu_activation input_filename input_rowsize" << endl;
            cout << "tanh_activation input_filename input_rowsize" << endl;
          }
    return 0;
}
