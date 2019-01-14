#include <iostream>
#include <climits>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstring>
#include <bits/stdc++.h> 
//#include <boost/algorithm/string.hpp> 
using namespace std;
vector<vector<float> > processed_input(vector<vector<float> > input,int kernel_size){

    vector<vector<float> > output;
    for(int i=0;i<kernel_size;i++){
        for(int j=1;j<=(input.size()-kernel_size);j++){
            for(int k=1;k<=(input.size()-kernel_size);k++){
                if(i==0){
                    vector<float> v;
                    output.push_back(v);
                }
                for(int x=0;x<kernel_size;x++){
                    output[j*k-1].push_back(input[i+j-1][k+x-1]);
                }
            }
        }
    }
    return output;

} 
vector<vector<float> > processed_kernel(vector<vector<float> > kernel){

    vector<vector<float> > output;
    for(int i=0;i<kernel.size();i++){
        vector<float> v;
        output.push_back(v);
        for(int j=0;j<kernel[i].size();i++){
            output[i*3+j].push_back(kernel[i][j]);
        }
     }
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

    for(int k=1;k<=i;k++){
        vector<float> v;
        output.push_back(v);
        for(int j=1;j<=i;j++){
            output[k-1].push_back(input[k*j-1][0]);
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
    for(int i=0;i<input.size();i++){
     	for(int j=0;j<input[i].size();i++){
     		input[i][j]=max(input[i][j],(float)0);
     	}
     }
    return input;
}

vector<vector<float> > tanh_activation(vector<vector<float> > input){
    for(int i=0;i<input.size();i++){
     	for(int j=0;j<input[i].size();i++){
     		input[i][j]=tanh(input[i][j]);
     	}
     }
   return input;   
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
void display(vector<vector<float> > input){
	for(int i=0;i<input.size();i++){
        for(int j=0;j<input[i].size();j++){
            
            cout<<input[j][i]<<" ";
        }
        cout<<endl;
    }
}
void disp(vector<float> input){
  for(int i=0;i<input.size();i++){        
            cout<<input[i]<<" ";
     }
     cout << endl;
}
void take_input(vector<vector<float> > input,string fil,string sizestr){
	     ifstream file;
	     file.open(fil);
	     int size = stoi(sizestr);
	     float x;
       for(int i=0;i<size;i++){
	      	for(int j=0;j<size;j++){
	      		if(i==0){
	      		vector<float> v;
	            input.push_back(v);
	            }
	            file>> x;
	            input[j].push_back(x);
	      	}
	    }
}
void split(vector<string> result,string s){
	int i=0;
  String temp = "";
	while(i<s.size()){
        if(s[i]!=" "){
          temp=temp+s[i];
          i++;
        }
        else{
          result.push_back(temp);
          temp="";
          i++;
        }		
	}
}
int main(){
    string s;
     do{
          getline(cin,s);
          vector<string> result;
          result = strtok(s," ");
          if(result[0]=="convolution"){
              
              vector<vector<float> > input;
              take_input(input,result[1],result[2]);
              vector<vector<float> > kernel;
              take_input(kernel,result[3],result[4]);
              display(convolution_withoutpadding(input,kernel));
          }
          else if(result[0]=="convolution_withpadding"){
              vector<vector<float> > input;
              int padsize = stoi(result[1]);
              take_input(input,result[2],result[3]);
              vector<vector<float> > kernel;
              take_input(kernel,result[4],result[5]);
              display(convolution_withpadding(padsize,input,kernel));
          }
          else if(result[0]=="convolution_matrixmult"){
              vector<vector<float> > input;
              take_input(input,result[1],result[2]);
              vector<vector<float> > kernel;
              take_input(kernel,result[3],result[4]);
              display(convolution_withoutpadding_matrixmult(input,kernel));
          }
          else if(result[0]=="convolution_withpadding_matrixmult"){
              vector<vector<float> > input;
              int padsize = stoi(result[1]);
              take_input(input,result[2],result[3]);
              vector<vector<float> > kernel;
              take_input(kernel,result[4],result[5]);
              display(convolution_withpadding_matrixmult(padsize,input,kernel));
          }
          else if(result[0]=="softmax"){
            vector<float> input;
            ifstream file;
	        file.open(result[1]);
	        string x;
	        file> >x;
            while (x!="\0"){
            	input.push_back(stoi(x));
            	file> >x;
            }
            disp(softmax(input));
          }
          else if(result[0]=="sigmoid"){
          	 vector<float> input;
            ifstream file;
	        file.open(result[1]);
	        string x;
	        file >> x;
            while (x!="\0"){
            	input.push_back(stoi(x));
            	file >> x;
            }
           disp(sigmoid(input));
          }
          else if(result[0]=="max_pooling"){
            vector<vector<float> > input;
            take_input(input,result[1],result[2]);
            display(max_pooling(input));
          }
          else if(result[0]=="average_pooling"){
          	vector<vector<float> > input;
            take_input(input,result[1],result[2]);
            display(average_pooling(input));
          }
          else if(result[0]=="relu_activation"){
              vector<vector<float> > inp;
              take_input(inp,result[1],result[2]);
              display(relu_activation(inp));
          }
          else if(result[0]=="tanh_activation"){
              vector<vector<float> > inp;
              take_input(inp,result[1],result[2]);
              display(tanh_activation(inp));
          }
     }
     while(s!="\0");
    
}
