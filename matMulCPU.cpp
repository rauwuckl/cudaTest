#include <iostream>
#include <chrono>
#include <functional>

using namespace std::chrono;

typedef std::function<void*() > func;

/**
 * multiply A (an x am ) with B (bn x bm). result will be a pointer to array of an x bm
 */
template <size_t an, size_t am, size_t bn, size_t bm>
int** matMul(int (&A)[an][am], int (&B)[bn][bm]){
    if( am != bn ){
        std::cout << "dimenstion don't fit" << std::endl;
        throw 0;
    }
    int i,j,k;
    int** out = new int*[an];
    for(i=0; i < an ; i++){
        out[i] = new int[bm];
    }

    for(i=0; i < an; i++){
        for(j=0; j<bm; j++){
            // go threw all fields in output
            // initialize output field as 0
            out[i][j]=0;
            
            for(k=0; k < am; k++){
                    out[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return out;
}

int** matMul(int** A, int an, int am, int** B, int bn, int bm){
    if( am != bn ){
        std::cout << "dimenstion don't fit" << std::endl;
        throw 0;
    }
    int i,j,k;
    int** out = new int*[an];
    for(i=0; i < an ; i++){
        out[i] = new int[bm];
    }

    for(i=0; i < an; i++){
        for(j=0; j<bm; j++){
            // go threw all fields in output
            // initialize output field as 0
            out[i][j]=0;
            
            for(k=0; k < am; k++){
                    out[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return out;
}

/// initialize matrix
int** init(int an, int am, int value){
    int** out = new int*[an];

    for(int i=0; i<an; i++){
        out[i] = new int[am];
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
}

template <class retType>
retType measureTime(func& f){
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	void* result = f();
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<microseconds>( t2 - t1 ).count();
	std::cout << "execution took " << duration << " microseconds" << std::endl;
	return (retType)result;
}


int main(){

    int A[2][2]={{1,2},{3,4}};
    int B[2][3]={{1,1,1},{1,1,1}};
    

    int** out = matMul(A, B);
    std::cout << "matmul stack allocated" << std::endl;
    print(out, 2, 3);


    int **C = init(300, 500, 1);
    int **D = init(500, 900, 2);

    func f = [C,D](){return (void*)matMul(C, 300, 500, D, 500, 900);};

    out=measureTime<int**>(f);
    std::cout << "the new one " << std:: endl;
//    print(out, 300, 900);
    return 0;
}
