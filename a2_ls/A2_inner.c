#include <mpi.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


// declare functions
float** createMatrix(int n, int m);
void initializeMatrix(int n, int m, float **matrix);
void printMatrix(int n, int m, float **matrix);
float innerProduct(float* matrixOne, float* matrixTwo, int m);
void product(float** matrix, int rows, int m, float* results);
void printTriangle(float* result, int n);
void selfCheck(float *result, float *resultTwo, int n);

int main(int argc, char** argv) {
    int n, m;                           //inputs
    int myid, numprocs;
    MPI_Status status;
    MPI_Request sendRequest, receiveRequest;
    float** initialVector = NULL;
    float** processVector = NULL;
    float** sendVector = NULL;
    float** receiveVector = NULL;
    float** selfCheckVector = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (argc != 3) {
        if (myid == 0) {
            printf("Requirement: [proc num] %% 2 = 1; [N] %% [proc num] = 0");
            printf("Please enter the command in the following format:\n");
            printf("mpirun -np [proc num] inner_products [N] [M]\n");
            printf("\n");
        }
        goto END;
    }

   
    n = atoi(argv[1]);
    m = atoi(argv[2]);

    if ((numprocs % 2 != 1) || (n % numprocs != 0)) {
        if (myid == 0) {
            printf("Requirement: [proc num] %% 2 = 1; [N] %% [proc num] = 0");
            printf("Please enter the command in the following format:\n");
            printf("mpirun -np [proc num] inner_products [N] [M]\n");
            printf("\n");
        }
        goto END;
    }

    int vectorPerProcs = n / numprocs;
    // total results count for the final
    int resultCount = (n+1) * n / 2; 
    // results count for every process
    int localResultCount = (n+1) * n / (2 * numprocs);

    float* result = (float*)malloc(sizeof(float) * resultCount );
    float* localResult = (float*)malloc(sizeof(float) * localResultCount );
    
    initialVector = createMatrix(n, m);
    // vector for self check
    selfCheckVector = createMatrix(n, m);
    processVector = createMatrix(vectorPerProcs, m);
    sendVector = createMatrix(vectorPerProcs, m);
    receiveVector = createMatrix(vectorPerProcs, m);

    if (numprocs == 1) {
        printf("Sequential computation: \n");
        initializeMatrix(n, m, initialVector);
        printf("The initial N*M vector is:\n");
        printMatrix(n, m, initialVector);
        float* sequentialResult = (float*)malloc(sizeof(float) * resultCount );
        product(initialVector, n, m, sequentialResult);
        printf("Sequential result is: \n");
        printTriangle(sequentialResult, n);
    } else {
        if (myid == 0) {
            //initialize and print initial vector
            initializeMatrix(n, m, initialVector);
            printf("The initial N*M vector is:\n");
            printMatrix(n, m, initialVector);
            //copy initial vector to self check
            for(int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    selfCheckVector[i][j] = initialVector[i][j];
                }
            }

            for(int i = 1; i < numprocs; i++) {
                //send data to other processes
                MPI_Send(initialVector[i * vectorPerProcs], vectorPerProcs * m, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
            }
            //memcpy(processVector, initialVector, vectorPerProcs * m * sizeof (float));
            for(int i = 0; i < vectorPerProcs; i++) {
                for (int j = 0; j < m; j++) {
                    processVector[i][j] = initialVector[i][j];
                }
            }
            for(int i = 0; i < vectorPerProcs; i++) {
                for (int j = 0; j < m; j++) {
                    sendVector[i][j] = processVector[i][j];
                }
            }
            //memcpy(sendVector, processVector, vectorPerProcs * m * sizeof(float));
        } else {
            //receive data from the 0
            MPI_Recv(*processVector, vectorPerProcs * m, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
            //memcpy(sendVector, processVector, vectorPerProcs * m * sizeof (float));
            for(int i = 0; i < vectorPerProcs; i++) {
                for (int j = 0; j < m; j++) {
                    sendVector[i][j] = processVector[i][j];
                }
            }

        }

        /*
         *  suppose 5 processes, n is 5.
         *  (0,0) (0,1) (0,2) (0,3) (0,4)
         *        (1,1) (1,2) (1,3) (1,4)
         *              (2,2) (2,3) (2,4)
         *                    (3,3) (3,4)
         *                          (4,4)
         * in iteration 0 compute (0,0), (1,1) ... (4,4)
         * as in process 0 store the process 1 data in receiveVector, copy this data to sendVector
         * then no.0 vector get the value of no.1 vector  .... no.4 get the value of no.0 vector
         * in iteration 1 compute (0,1), (1,2) ... (0,4).
         * process 2 store process 3 data in receiveVector -> sendVector in iteartion 0
         * process 1 get process 2's sendVector which is process 3 data and store in receiveVector -> sendVector in iteration 1
         * in iteration 2, compute (0,2), (1,3) ... (0,3), (1,4)
         */
        for (int iter = 0; iter < (numprocs+1)/2; iter++) {
            //every process sends its own sendVector data and store the received data in receiveVector.
            MPI_Isend(sendVector[0], vectorPerProcs * m, MPI_FLOAT, (myid - 1 + numprocs) % numprocs, 1, MPI_COMM_WORLD, &sendRequest);
            MPI_Irecv(receiveVector[0], vectorPerProcs * m, MPI_FLOAT, (myid + 1) % numprocs, 1, MPI_COMM_WORLD, &receiveRequest);
            if (iter == 0) {
                //in first iteration only need to compute with itself
                for (int i = 0; i < vectorPerProcs; i++) {
                    for(int j = i; j < vectorPerProcs; j++) {
                        // vector product between i-th in local vec, j-th in send vec
                        float triangle_ij = innerProduct(processVector[i], sendVector[j], m);
                        // compute Cij
                        /*  for every loop plus vectorPerProcs * ((numprocs + 1)/2) first
                         *  as it is a triangle need to minus the number of # which is a small triangle too.
                         *  (0,0) (0,1) (0,2) (0,3) (0,4)
                         *    #   (1,1) (1,2) (1,3) (1,4)
                         *    #     #   (2,2) (2,3) (2,4)
                         *    #     #     #   (3,3) (3,4)
                         *    #     #     #     #   (4,4)
                         */
                        int counter  = vectorPerProcs * ((numprocs + 1)/2) * i - (i - 1)*i/2 + j -i;
                        localResult[counter] = triangle_ij;
                    }
                }
            } else {
                for (int i = 0; i < vectorPerProcs; i++) {
                    for (int j = 0; j < vectorPerProcs; j++) {
                        float triangle_ij = innerProduct(processVector[i], sendVector[j], m);
                        /* the same as 0 iter, just need to notice j starts from the vectorPerProcs
                         * as vector per process is 2, it can calculate (0,0), (0,1)
                         * as vector per process is 3, it can calculate (0,0), (0,1), (0,2)
                         */
                        int triangle_j = iter * vectorPerProcs + j;
                        int counter = vectorPerProcs * ((numprocs + 1)/2) * i - (i-1) * i / 2 + triangle_j - i;
                        localResult[counter] = triangle_ij;
                    }
                }
            }
            MPI_Wait(&sendRequest, &status);
            MPI_Wait(&receiveRequest, &status);
            if(iter < (numprocs+1)/2 - 1){
                float** temp;
                temp = receiveVector;
                receiveVector = sendVector;
                sendVector = temp;
            }
        }

        //send result back to master
        if (myid == 0) {

            for (int i = 0; i < vectorPerProcs; i++) {
                for (int j = i; j < (numprocs+1)/2*vectorPerProcs; j++) {
                    int k_local = vectorPerProcs * ((numprocs+1)/2) * i - (i-1)*i/2 + j - i;
                    int k_global = n * i - (i-1)*i/2 + j - i;
                    result[k_global] = localResult[k_local];
                }
            }
            // every process send its own to the final result, to build a triangle
            /*
             * np is 3, N is 6
             * (0,0) (0,1) (0,2) (0,3) (0,4) (0,5)
             *   #   (1,1) (1,2) (1,3) (1,4) (1,5)
             *   calculated by np 0 is
             *   (0,0) (0,1) (0,2) (0,3)
             *         (1,1) (1,2) (1,3)
             *   first line is 4, second is 3.
             */
            int basic = vectorPerProcs * (numprocs + 1) / 2;

            for (int p = 1; p < numprocs; p++) {
                MPI_Recv(localResult, localResultCount, MPI_FLOAT, p, 1, MPI_COMM_WORLD, &status);
                for (int i = 0; i < vectorPerProcs; i++) {
                    for (int j = i; j < basic; j++) {
                        int triangle_i = i + p * vectorPerProcs;
                        int triangle_j = j + p * vectorPerProcs;
                        if (triangle_j >= n) {
                            /*
                             *  (0,0) (0,1) (0,2) (0,3) (0,4) (0,5)
                             *    #   (1,1) (1,2) (1,3) (1,4) (1,5)
                             *    #     #   (2,2) (2,3) (2,4) (2,5)
                             *    #     #     #   (3,3) (3,4) (3,5)
                             *    #     #     #     #   (4,4) (4,5)
                             *    #     #     #     #     #   (5,5)
                             *    np = 2 -> get (4,4) (4,5) (0,4) (1,4)
                             *    triangle_j = 6 when j = 2 need to % N -> 0
                             *    then swap from (4,6) -> (4,0) -> (0,4)
                             */
                            triangle_j = triangle_j % n;
                            int temp;
                            temp = triangle_j;
                            triangle_j = triangle_i;
                            triangle_i = temp;
                        }
                        int k_local = vectorPerProcs * ((numprocs+1)/2) * i - (i - 1)*i/2 + j - i;
                        int k_global = n * triangle_i - (triangle_i - 1) * triangle_i / 2 + triangle_j - triangle_i;
                        result[k_global] = localResult[k_local];
                    }
                }
            }

        } else {
            //send rsult back to the 0.
            MPI_Send(localResult, localResultCount, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }

        if (myid == 0) {

//            for (int i=0; i<resultCount; i++) {
//                printf("%-6.3f ", result[i]);
//            }
//            printf("\n");
            printTriangle(result, n);

            printf("Then perform self check !\n");

            float* sequentialResult = (float*)malloc(sizeof(float) * resultCount );
            product(selfCheckVector, n, m, sequentialResult);
            printf("Sequential computation: \n");
//            for (int i=0; i<resultCount; i++) {
//                printf("%-6.3f ", sequentialResult[i]);
//            }
//            printf("\n");
            printTriangle(sequentialResult, n);
            selfCheck(result, sequentialResult, n);
        }
    }

    END:
    MPI_Finalize();
    return 0;
}

float** createMatrix(int n, int m) {
    float* flat = malloc(sizeof(float) * n * m);
    float** matrix = malloc(sizeof(float*) * n);
    int i;

    for (i = 0; i < n; i++){
        matrix[i] = &flat[i * m];
    }

    return matrix;
}

void initializeMatrix(int n, int m, float** matrix) {
    time_t timeSeed;
    srand((unsigned)time(&timeSeed));

    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            matrix[i][j] = (float)rand() / (float)RAND_MAX;
            matrix[i][j] = (int) (1000.0 * matrix[i][j] + 0.5) /1000.0;
        }
    }
}

void printMatrix(int n, int m, float** matrix) {
    int i, j;

    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            printf("%-5.3f ", matrix[i][j]);
        }
        printf("\n");
    }
}

float innerProduct(float* matrixOne, float* matrixTwo, int m) {
    float result = 0;
    //printf("matrixOne[0] is: %-6.3f , matrixTwo[0] is:  %-6.3f. \n", matrixOne[0], matrixTwo[0]);
    for (int i = 0; i < m; i++) {
        result += matrixOne[i] * matrixTwo[i];
    }

    return result;
}

void product(float** matrix, int rows, int m, float* results) {
    int i, j;
    int counter = 0;
    for (i = 0; i < rows; i++) {
        for (j = i; j < rows; j++) {
            results[counter] = innerProduct(matrix[i], matrix[j], m);
            counter++;
        }
    }
}

void printTriangle(float* result, int n) {
    printf("Triangle result is: \n");
    for(int i = n; i > 0; i--) {
        for(int k = n-i; k>0; k--) {
            printf("          ");
        }
        for(int j = 0; j < i; j++) {
            int counter = (n-i)*n - (n-i-1) * (n-i) / 2 + j;
            printf("%-9.3f ", result[counter]);
        }
        printf("\n");
    }
}

void selfCheck(float *result, float *resultTwo, int n) {
    int flag = 0;
    int i;
    int total = (n+1) * n / 2;

    for (i = 0; i < total; i++) {
        if (result[i] != resultTwo[i]) {
            flag = 1;
        }
    }
    printf("\n");
    if (flag == 0) {
        printf("Result for self check: The result of parallel program is the same as sequential program.\n");
    } else {
        printf("Result for self check: The result of parallel program is not the same as sequential program.\n");
        printf("ERROR !!\n");
    }
}





