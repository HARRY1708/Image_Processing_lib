#include <iostream>
#include <climits>
#include <algorithm>
#include <vector>
#include <cstring>
#include <fstream>
#include <cmath>
// #include <bits/stdc++.h>
using namespace std;
void display(vector<vector<float> > input){
  for(int i=0;i<input.size();i++){
        for(int j=0;j<input[i].size();j++){
           cout<<input[i][j]<< " ";
        }
        cout << endl;
    }
}
vector<vector<float> > processed_input(vector<vector<float> > input,int kernel_size){

    vector<vector<float> > output;
    for(int i=0;i<kernel_size;i++){
        int y=0;
        for(int j=0;j<=(input.size()-kernel_size);j++){
            for(int k=0;k<=(input.size()-kernel_size);k++){
                if(i==0){
                    vector<float> v;
                    output.push_back(v);
                }
                for(int x=0;x<kernel_size;x++){
                    // cout << 
                    output[y].push_back(input[i+j][k+x]);
                }
                y++;
            }
        }
    }
    //display(output);
    return output;

} 
vector<vector<float> > processed_kernel(vector<vector<float> > kernel){

    vector<vector<float> > output;
    int k=0;
    for(int i=0;i<kernel.size();i++){
        
        for(int j=0;j<kernel[i].size();j++){
            vector<float> v;
            output.push_back(v);
            //cout << kernel[i][j]<< " ";
            output[k].push_back(kernel[i][j]);
            k++;
        }
     }
     //display(output);
    return output;
}

vector<vector<float> > matrix_mult(vector<vector<float> > input,vector<vector<float> > kernel){
    vector<vector<float> > output;
    float sum=0;
    for(int i=0;i<input.size();i++){
    	vector<float> v;
		  output.push_back(v);
        for(int j=0;j<kernel[0].size();j++){
            sum=0;
            for(int k=0;k<kernel.size();k++){
                sum+=input[i][k]*kernel[k][j];
            }
            output[i].push_back(sum);
        }
    }
    return output;
}
    
vector<vector<float> > inverse_input(vector<vector<float> > input){
    vector<vector<float> > output;
    int i= (int)sqrt(input.size());
    int x=0;
    for(int k=0;k<i;k++){
        vector<float> v;
        output.push_back(v);
        for(int j=0;j<i;j++){
            output[k].push_back(input[x][0]);
            x++;
        }
    }
    return output;
}

vector<vector<float> > convolution_withoutpadding(vector<vector<float> > input,vector<vector<float> > kernel){
             vector<vector<float> > output;
             for(int i=0;i<input.size()-kernel.size()+1;i++){
             	vector<float> v;
		        output.push_back(v);
             	for(int j=0;j<input.size()-kernel.size()+1;j++){
             		float sum=0;
             		for(int u=0;u<kernel.size();u++){
             	       for(int v=0;v<kernel.size();v++){
                            sum+=(input[i+u][j+v]*kernel[u][v]);
             	       }   
             		}
             	  output[i].push_back(sum);
             	}	
             }
  return output;
}

vector<vector<float> > convolution_withoutpadding_matrixmult(vector<vector<float> > input,vector<vector<float> > kernel){
      //display(processed_kernel(kernel));
      return inverse_input(matrix_mult(processed_input(input,kernel.size()),processed_kernel(kernel)));
}

vector<vector<float> > padding(int padsize,vector<vector<float> > input){

	vector<vector<float> > output;
	for(int i=0;i<input.size()+2*padsize;i++){
		  vector<float> v;
		  output.push_back(v);
        for(int j=0;j<input.size()+2*padsize;j++){
        	if(i<padsize||i>=input.size()+padsize||j<padsize||j>=input.size()+padsize)
             output[i].push_back(0.0);
            else
             output[i].push_back(input[i-padsize][j-padsize]); 	
        }
	}
	return output;
}

vector<vector<float> > convolution_withpadding(int padsize,vector<vector<float> > input,vector<vector<float> > kernel){
      
      return convolution_withoutpadding(padding(padsize,input),kernel);
}

vector<vector<float> > convolution_withpadding_matrixmult(int padsize,vector<vector<float> > input,vector<vector<float> > kernel){

      return convolution_withoutpadding_matrixmult(padding(padsize,input),kernel);
}

vector<vector<float> > relu_activation(vector<vector<float> > input){
	vector<vector<float> > output;
              //cout<<12345<<endl;
    for(int i=0;i<input.size();i++){
    	vector<float> v;
    	output.push_back(v);
    	 
     	for(int j=0;j<input[i].size();j++){
     		output[i].push_back(max(input[i][j],(float)0));
     	}
     }
    return output;
}

vector<vector<float> > tanh_activation(vector<vector<float> > input){
    vector<vector<float> > output;
    for(int i=0;i<input.size();i++){
    	vector<float> v;
    	output.push_back(v);
     	for(int j=0;j<input[i].size();j++){
     		output[i].push_back(tanh(input[i][j]));
     	}
     }
    return output;
}

// acts as a hidden layer in neural network by reducing the amount of data and taking relevant data by taking max of 2*2 sliding window
vector<vector<float> > max_pooling(vector<vector<float> > input){
    if(input.size()%2==1){
    	vector<float> extra;
    	for(int i=0;i<input.size();i++){
    		extra.push_back(0);
    	}
        input.push_back(extra);
        for(int i=0;i<input.size();i++){
        	input[i].push_back(0);
        }
    }
    vector<vector<float> > output;
    for(int i=0;i<input.size()/2;i++){
    	vector<float> v;
    	output.push_back(v);
    	for(int j=0;j<input.size()/2;j++){
    		output[i].push_back(max(input[2*i][2*j],max(input[2*i+1][2*j],max(input[2*i+1][2*j+1],input[2*i][2*j+1]))));
    	}
    }
    return output;
}

// acts as a hidden layer in neural network by reducing the amount of data and taking relevant data by taking average of 2*2 sliding window
vector<vector<float> > average_pooling(vector<vector<float> > input){
    if(input.size()%2==1){
    	vector<float> extra;
    for(int i=0;i<input.size();i++){
    		extra.push_back(0);
    	}
        input.push_back(extra);
    for(int i=0;i<input.size();i++){
        	input[i].push_back(0);
        }
    }
    vector<vector<float> > output;
    for(int i=0;i<input.size()/2;i++){
    	vector<float> v;
    	output.push_back(v);
    for(int j=0;j<input.size()/2;j++){
    		output[i].push_back((input[2*i][2*j]+input[2*i+1][2*j]+input[2*i+1][2*j+1]+input[2*i][2*j+1])/4);
    	}
    }
  return output;  
}
// maps the vector to [0,1] hence giving a measure of probability
vector<float> softmax(vector<float> input){
	float sum=0;
	vector<float> output;
    for(int j=0;j<input.size();j++){
     		sum+=exp(input[j]);
     	}
    for(int j=0;j<input.size();j++){
     		output.push_back(exp(input[j])/sum);
     	}
    return output;
}
// maps the vector to [0,1] hence giving a measure of probability
vector<float> sigmoid(vector<float> input){
	vector<float> output;
	for(int j=0;j<input.size();j++){
     		output.push_back(1/(1+exp(-1*input[j])));
     	}
    return output; 	
}

void disp(vector<float> input){
  for(int i=0;i<input.size();i++){        
           cout<<input[i]<<endl;
     }
}
bool take_input(vector<vector<float> > &input,string fil,string sizestr){
	     
           fstream file;
    	     file.open(fil);
           if(file){
          	     int size = stoi(sizestr);
          	     float x;
                 for(int i=0;i<size;i++){
          	      	for(int j=0;j<size;j++){
          	      		if(i==0){
          	      		     vector<float> v;
          	               input.push_back(v);
          	            }
          	            file>> x;
                        if(file){
                          input[j].push_back(x);

                        }
                        else{
                          cout << "Given File with file name "+fil+" doesnt have size "+sizestr<<endl;
                          return false;
                        }
          	            
          	      	}
          	    }
                return true;
          	    file.close();
            }
            else{
              cout << "Given File with file name "+fil+" not found"<< endl;
              return false;
            }
      //display(input);
}
int main(int argc,char* argv[]){
            //cout<<argv[1]<<" "<<argv[2]<<" "<<argv[3]<<argv[4]<< argv[5]<<endl;
          if(!strcmp(argv[1],"convolution")){

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
              if(take_input(input,argv[3],argv[4])&&take_input(kernel,argv[5],argv[6])){
                   display(convolution_withpadding(padsize,input,kernel));
              }
          }
          else if(!strcmp(argv[1],"convolution_matrixmult")){
              vector<vector<float> > kernel;
              vector<vector<float> > input;
              if(take_input(input,argv[2],argv[3])&&take_input(kernel,argv[4],argv[5])){
                  display(convolution_withoutpadding_matrixmult(input,kernel));
              }
          }
          else if(!strcmp(argv[1],"convolution_withpadding_matrixmult")){
             vector<vector<float> > input;
              int padsize = stoi(argv[2]);
              vector<vector<float> > kernel;
              if(take_input(input,argv[3],argv[4])&&take_input(kernel,argv[5],argv[6])){
                     display(convolution_withpadding_matrixmult(padsize,input,kernel));
              }
          }
          else if(!strcmp(argv[1],"softmax")){
            vector<float> input;
            ifstream file;
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
            else{
                cout << "Given File with file name "+argv[2]+" not found"<< endl;
             }
          }
          else if(!strcmp(argv[1],"sigmoid")){
          	 vector<float> input;
            ifstream file;
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
                cout << "Given File with file name "+argv[2]+" not found"<< endl;

            }
          }
          else if(!strcmp(argv[1],"max_pooling")){
            vector<vector<float> > input;
            if(take_input(input,argv[2],argv[3]))
            {
                display(max_pooling(input));
            }
          }
          else if(!strcmp(argv[1],"average_pooling")){
          	vector<vector<float> > input;
            if(take_input(input,argv[2],argv[3])){
                display(average_pooling(input));
            }
          }
          else if(!strcmp(argv[1],"relu_activation")){
              vector<vector<float> > inp;
              if(take_input(inp,argv[2],argv[3]))
              {
                display(relu_activation(inp));
              }
          }
          else if(!strcmp(argv[1],"tanh_activation")){
              vector<vector<float> > inp;
              if(take_input(inp,argv[2],argv[3])){
                  display(tanh_activation(inp));
               }
          }
          else{
            cout << "Input type is not a given function. Please type from one of the functions from below" << endl;
            cout << "convolution" << endl;
            cout << "convolution_withpadding" << endl;
            cout << "convolution_matrixmult" << endl;
            cout << "convolution_withpadding_matrixmult" << endl;
            cout << "softmax" << endl;
            cout << "sigmoid" << endl;
            cout << "max_pooling" << endl;
            cout << "average_pooling" << endl;
            cout << "relu_activation" << endl;
            cout << "tanh_activation" << endl;
          }
    return 0;
}
