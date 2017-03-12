#include <iostream>
#include <chrono>
#include <functional>
#include <cmath>
#include <stdio.h>

using namespace std::chrono;

typedef std::function<void*() > func;

/**
 * multiply A (an x am ) with B (bn x bm). result will be a pointer to array of an x bm
 */
//template <size_t an, size_t am, size_t bn, size_t bm>
//int** matMul(int (&A)[an][am], int (&B)[bn][bm]){
//    if( am != bn ){
//        std::cout << "dimenstion don't fit" << std::endl;
//        throw 0;
//    }
//    int i,j,k;
//    int** out;
//    return out;
//    out = new int*[an];
//    for(i=0; i < an ; i++){
//        out[i] = new int[bm];
//    }
//
//    for(i=0; i < an; i++){
//        for(j=0; j<bm; j++){
//            // go threw all fields in output
//            // initialize output field as 0
//            out[i][j]=0;
//            
//            for(k=0; k < am; k++){
//                    out[i][j] += A[i][k] * B[k][j];
//            }
//        }
//    }
//    return out;
//}


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

    //int A[2][2]={{1,2},{3,4}};
    //int B[2][3]={{1,1,1},{1,1,1}};
    //

    //int** out = matMul(A, B);
    //std::cout << "matmul stack allocated" << std::endl;
    //print(out, 2, 3);


    int **out;
    int **C = init(300, 500, 1);
    int **D = init(500, 900, 1);

    func f = [C,D](){return (void*)matMul(C, 300, 500, D, 500, 900);};

    out=measureTime<int**>(f);
    std::cout << "the new one " << std:: endl;
//    print(out, 300, 900);
    return 0;
}
