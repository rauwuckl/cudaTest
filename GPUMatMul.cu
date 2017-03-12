#include <iostream>
#include <chrono>
#include <functional>
#include <cmath>
#include <stdio.h>

using namespace std::chrono;

typedef std::function<void*() > func;

class Matrix;
__global__
void cudaMatMulClass(const int* A, int an, int am, const int* B, int bn, int bm, int* out);


class Matrix{
private:
   int m_dimX;
   int m_dimY;
   int* m_content;

   void deleteContent(){
	if(m_content != NULL){
	    cudaFree(m_content);
	    m_content = NULL;
	}	
   }

   class helper{
	public:
	int& operator[](int j){
	    return m_M->m_content[m_i*(m_M->dimY()) +j];
	}
	helper(const Matrix* m, int i): m_M(m), m_i(i){}
	private:
	    const Matrix* m_M;
	    int m_i;
	};	

//   class helperC{
//	public:
//	int operator[](int j){
//	    return m_M->m_content[m_i*j];
//	}
//	helper(const Matrix* m, int i): m_M(m), m_i(i){}
//	private:
//	    const Matrix* m_M;
//	    int m_i;
//	};	

public:
   Matrix():m_dimX(-1), m_dimY(-1), m_content(NULL){}

   Matrix(int dy, int dx, int val):
	m_dimX(dx), m_dimY(dy){

	    cudaMallocManaged(&m_content, m_dimX*m_dimY*sizeof(int));

	    for(int i=0; i<(m_dimY*m_dimX); i++){
		m_content[i] = val;
	    }
   }

   Matrix(int dy, int dx):
	m_dimX(dx), m_dimY(dy){
	    cudaMallocManaged(&m_content, m_dimX*m_dimY*sizeof(int));
   }

   int nElem(){return m_dimX*m_dimY;}
   int dimX() const{return m_dimX;} 
   int dimY() const{return m_dimY;} 

    Matrix& operator=(const Matrix& other){
	if(this != &other){
	    this->deleteContent();
	    this->m_content = other.m_content;
	    this->m_dimX = other.m_dimX;
	    this->m_dimY = other.m_dimY;
	}
	return *this;
    }

    void print(){
	//TODO
	    for(int i=0; i<m_dimY; i++){
		for(int j=0; j<m_dimX; j++){
		    std::cout << (*this)[i][j] << ",";
		}
		std::cout << std::endl;
	    }
	    std::cout << "("<<m_dimY<<","<<m_dimX<<")"<<std::endl;
    }

    helper operator[] (const int i) const{
	return helper(this, i);
    }


    Matrix matMul(const Matrix& other) const{
      if( this->m_dimX != other.m_dimY ){
          std::cout << "dimenstion don't fit " << std::endl;
          throw 0;
      }
      int blockSize = 256;
      int numBlocks = ((m_dimY * other.m_dimX) + blockSize - 1) / blockSize;

      Matrix ret(m_dimY, other.m_dimX);
      cudaMatMulClass<<<numBlocks,blockSize>>>(m_content, m_dimY, m_dimX, other.m_content, other.dimY(), other.dimX(), ret.m_content);
      //cudaMatMul<<<(2,2),(3,3)>>>(A, an, am, B, bn, bm, out);
      cudaDeviceSynchronize();
      return ret;
    }



};
__global__
void cudaMatMulClass(const int* A, int an, int am, const int* B, int bn, int bm, int* out){
    int i,j,ij, k;
    int ijS = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; //how many threads ber bock * how many blocks in the grid;

    //int jS = blockIdx.y * blockDim.y + threadIdx.y;
    //int jStride = blockDim.y * gridDim.y; //how many threads ber bock * how many blocks in the grid;
	if(blockIdx.x==0 && threadIdx.x==0){
	printf("blockInd.x= %d ,blockDim.x= %d, threadIdx.x= %d, GridDim.x= %d\n", blockIdx.x ,blockDim.x , threadIdx.x ,gridDim.x);
	printf("blockInd.y= %d ,blockDim.y= %d, threadIdx.y= %d, GridDim.y= %d\n", blockIdx.y ,blockDim.y , threadIdx.y ,gridDim.y);
	printf("ijS= %d ,stride= %d \n", ijS, stride);
}
    for(ij=ijS; ij < an*bm; ij+=stride){
	i = ij/bm;
	j = ij%bm;
	out[ij]=0;
	for(k=0; k<am; k++){
	    out[ij] += A[i*an + k]*B[k*bn +j];
	}
	//printf("(i:%d, j:%d)=%d \n", i,j, out[i][j]);
    }
}	

__global__
void cudaMatMul(int** A, int an, int am, int** B, int bn, int bm, int** out){
    int i,j,ij, k;
    int ijS = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; //how many threads ber bock * how many blocks in the grid;

    //int jS = blockIdx.y * blockDim.y + threadIdx.y;
    //int jStride = blockDim.y * gridDim.y; //how many threads ber bock * how many blocks in the grid;
	if(blockIdx.x==0 && threadIdx.x==0){
	printf("blockInd.x= %d ,blockDim.x= %d, threadIdx.x= %d, GridDim.x= %d\n", blockIdx.x ,blockDim.x , threadIdx.x ,gridDim.x);
	printf("blockInd.y= %d ,blockDim.y= %d, threadIdx.y= %d, GridDim.y= %d\n", blockIdx.y ,blockDim.y , threadIdx.y ,gridDim.y);
	printf("ijS= %d ,stride= %d \n", ijS, stride);
}
	//std::cout << "blockInd.x="<< blockIdx.x << " blockDim.x=" << blockDim.x << " threadIdx.x=" << threadIdx.x << " GridDim.x" << gridDim.x << std::endl;
	//std::cout << "blockInd.y="<< blockIdx.y << " blockDim.y=" << blockDim.y << " threadIdx.y=" << threadIdx.y << " GridDim.y" << gridDim.y << std::endl;


    //for(i=iS; i < an; i+=iStride){
    //    for(j=0; j<bm; j++){
    //        // go threw all fields in output
    //        // initialize output field as 0
    //        out[i][j]=0;
    //        
    //        for(k=0; k < am; k++){
    //                out[i][j] += A[i][k] * B[k][j];
    //        }
    //    }
    //}
	//version with single loop
    
    for(ij=ijS; ij < an*bm; ij+=stride){
	i = ij/bm;
	j = ij%bm;
	out[i][j]=0;
	for(k=0; k<am; k++){
	    out[i][j] += A[i][k]*B[k][j];
	}
	//printf("(i:%d, j:%d)=%d \n", i,j, out[i][j]);
    }
}	


int** matMul(int** A, int an, int am, int** B, int bn, int bm){
    if( am != bn ){
        std::cout << "dimenstion don't fit" << std::endl;
        throw 0;
    }
    int N = std::max(an, bm);
    int blockSize = 256;
    int numBlocks = ((an * bm) + blockSize - 1) / blockSize;
    int i;
    int **out;
    //out = new int*[an];
	cudaMallocManaged(&out, an*sizeof(int*));
    for(i=0; i < an ; i++){
        //out[i] = new int[bm];
	cudaMallocManaged(&out[i], bm*sizeof(int));
    }
    cudaMatMul<<<numBlocks,blockSize>>>(A, an, am, B, bn, bm, out);
    //cudaMatMul<<<(2,2),(3,3)>>>(A, an, am, B, bn, bm, out);
    cudaDeviceSynchronize();
    return out;
}

/// initialize matrix
int** init(int an, int am, int value){
    //int** out = new int*[an];
    int** out;
    cudaMallocManaged(&out, an*sizeof(int*));

    for(int i=0; i<an; i++){
        //out[i] = new int[am];
	cudaMallocManaged(&out[i], am*sizeof(int));
        for(int j=0; j<am; j++){
            out[i][j] = value;
        }
    }
    return out;
}

void print(int** A, int an, int am){
    std::cout << "A = " << std::endl;

    for(int i=0; i<an; i++){
        for(int j=0; j<am; j++){
            std::cout << A[i][j] << ",";
        }
        std::cout << std::endl;
    }
    std::cout << "("<<an<<","<<am<<")"<<std::endl;
}

template <class retType>
retType measureTime(func& f){
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	void* result = f();
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<milliseconds>( t2 - t1 ).count();
	std::cout << "execution took " << duration << " milliseconds" << std::endl;
	return (retType)result;
}


int main(){
    std::cout << "gpu version" << std::endl;

    int A[2][2]={{1,2},{3,4}};
    int B[2][3]={{1,1,1},{1,1,1}};
    
    int **out;
    int **C = init(300, 500, 1);
    int **D = init(500, 900, 1);

    func f = [C,D](){return (void*)matMul(C, 300, 500, D, 500, 900);};

    out=measureTime<int**>(f);
    std::cout << "the new one " << std:: endl;
    //print(out, 300, 900);


    Matrix MA(300,500,1);
    Matrix MB(500,900,1);
    //func f = [A,B](){return (void*)A.matMul(B);}
    //Matrix C = measureTime<Matrix>(f);
    Matrix MC = MA.matMul(MB);
    //MC.print();
    return 0;
}
