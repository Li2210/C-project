#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

#define white 0
#define red 1
#define blue 2
#define PARALLEL 3
#define SEQUENTIAL 4
#define REDEXCEED 5
#define BLUEEXCEED 6

//declare functions
void createGrid(int gridWidth, int gridHeight, int **grid1D, int ***grid);
void initGrid(int n, int **grid);
void printGrid(int n, int **grid);
void redMove(int row, int column, int **grid);
void blueMove(int row, int column, int **grid);
int* distributeRowToProcesses(int numprocs, int n, int t);
void sequentialComputation(int **grid, int tileCount, int n, int t, int c, int max_itrs);
int tileChecker(int **grid, int tileCount, float *tileInfoArray, int n, int t, int c, int type);
void printResult(float *tileInfoArray, int tileCount, int t, int n, int **grid);
void selfChecker(int **grid, int **gridCopy, int n);

int main(int argc, char **argv) {
    int **grid;	//two-dimension grid
    int *grid1D;	//one-dimension version of grid
    int **gridCopy;
    int *grid1DCopy;
    int n; //n * n grid size
    int t; //t * t tile in grid
    int c; //c threshold
    int max_itrs;	//max_itrs maximum number of iterations
    int n_itrs = 0;	//iteration times
    int finished = 0; //flag for whether the computation finished or not
    int finishedProcs = 0; //finished flag for processor
    int tileNumber; //the number of tiles in the grid
    int myid;
    int numprocs;
    int i, j;
    //initialize mpi
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (argc != 5){
        if (myid == 0) {
            printf("Wrong number of arguments.\n");
            printf("Please enter the command in the following format:\n");
            printf("mpirun np 4 main 30 10 70 10\n");
        }
        goto END;
    }

    //get the value of input
    n = atoi(argv[1]);
    t = atoi(argv[2]);
    c = atoi(argv[3]);
    max_itrs = atoi(argv[4]);

    int temp = n / t; //rows in each tile
    int *rowProcs;

    if (numprocs == 1) { //only one processor then perform sequential computation
        tileNumber = t * t;
        createGrid(n, n, &grid1D, &grid);
        printf("As only have one process. perform sequential computation !\n");
        initGrid(n, grid);
        printf("The initial grid: \n");
        printGrid(n, grid);
        sequentialComputation(grid, tileNumber, n, t, c, max_itrs);
        goto END;
    } else { //parallel computation
        rowProcs = distributeRowToProcesses(numprocs, n, t);
        if (myid == 0) {
            tileNumber = t * t;
            float *tileInfoArray = (float *)malloc(sizeof(float) * tileNumber * 3);
            //create and initialize the board.
            createGrid(n, n, &grid1D, &grid);
            createGrid(n, n, &grid1DCopy, &gridCopy);
            initGrid(n, grid);
            printf("The initial grid: \n");
            printGrid(n, grid);
            //used for self check
            memcpy(grid1DCopy, grid1D, sizeof(int) * n * n);
            //send sub-grid to each processor
            for (i = 1; i < numprocs; i++) {
                int index = 0;
                if (i > 1) {
                    for (j = 0; j < i; j++) {
                        index += rowProcs[j];
                    }
                    index -= rowProcs[i - 1];
                }
                MPI_Send(&grid1D[index * n], rowProcs[i - 1] * n, MPI_INT, i, 1, MPI_COMM_WORLD);
            }
            //terminate when the computation reach max iterations or colour exceeds threshold
            while (!finished && n_itrs < max_itrs) {
                //receive iteration number from process 1
                MPI_Recv(&n_itrs, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, &status);
                MPI_Allreduce(&finishedProcs, &finished, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            }
            //receive the computation result from other processes
            for (i = 1; i < numprocs; i++) {
                int index = 0;
                if (i > 1) {
                    for (j = 0; j < i; j++) {
                        index += rowProcs[j];
                    }
                    index -= rowProcs[i - 1];
                }
                MPI_Recv(&tileInfoArray[(index * n) / (temp * temp) * 3], (n * rowProcs[i - 1]) / (temp * temp) * 3, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(&grid1D[index * n], rowProcs[i - 1] * n, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
            }
            printf("The parallel computation result: \n");
            printf("After %d interations, the final grid: \n", n_itrs);
            printGrid(n, grid);
            printResult(tileInfoArray, tileNumber, t, n, grid);
            //self check
            printf("Self-checking after parallel computation !\n");
            sequentialComputation(gridCopy, tileNumber, n, t, c, max_itrs);
            selfChecker(grid, gridCopy, n);
        } else {
            //tile numbers assigned in every process
            tileNumber = (n * rowProcs[myid - 1]) / (temp * temp);
            float *tileInfoArrayInProcessor = (float *)malloc(sizeof(float) * tileNumber * 3);
            int* subGrid1D;
            int** subGrid;
            //create subGrid
            createGrid(n, rowProcs[myid - 1] + 2, &subGrid1D, &subGrid); //将(row[myid - 1] + 2) * n分配给了各个process，每个多分配了两行
            //receive data from processor 0
            MPI_Recv(&subGrid1D[n], rowProcs[myid - 1] * n, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
            while (!finished && n_itrs < max_itrs) {
                n_itrs = n_itrs + 1;
                //send the iteration number to process 0
                if (myid == 1) {
                    MPI_Send(&n_itrs, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
                }
                redMove(n, rowProcs[myid - 1] + 2, subGrid);
                //collect ghost line data
                MPI_Sendrecv(&subGrid1D[(rowProcs[myid - 1]) * n], n, MPI_INT,myid % (numprocs - 1) + 1,1,
                             &subGrid1D[0], n, MPI_INT,(myid - 2 + (numprocs - 1)) % (numprocs - 1) + 1,1, MPI_COMM_WORLD, &status);
                MPI_Sendrecv(&subGrid1D[n], n, MPI_INT,(myid - 2 + (numprocs - 1)) % (numprocs - 1) + 1, 2,
                             &subGrid1D[rowProcs[myid - 1] * n + n], n, MPI_INT, myid % (numprocs - 1) + 1, 2, MPI_COMM_WORLD, &status);

                blueMove(n, rowProcs[myid - 1] + 2, subGrid);
                finishedProcs = tileChecker(subGrid, tileNumber, tileInfoArrayInProcessor, n, t, c, PARALLEL);
                MPI_Allreduce(&finishedProcs, &finished, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            }
            //send the final red blue ratio result to processor 0
            MPI_Send(&tileInfoArrayInProcessor[0], tileNumber * 3, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
            //send the sub-grid to processor 0
            MPI_Send(&subGrid1D[n], rowProcs[myid - 1] * n, MPI_INT, 0, 1, MPI_COMM_WORLD);
        }
    }
    END:
    MPI_Finalize();
    return 0;
}

//create grid
void createGrid(int gridWidth,
                int gridHeight,
                int **grid1D,
                int ***grid) {
    int count = gridWidth * gridHeight;
    *grid1D = (int *)malloc(sizeof(int) * count);
    *grid = (int **)malloc(sizeof(int *) * gridHeight);
    int i;
    for (i = 0; i < gridHeight; i++) {
        (*grid)[i] = &((*grid1D)[i * gridWidth]);
    }
}

//initialize grid
void initGrid(int n, int **grid) {
    //set a random seed
    time_t timeSeed;
    srand((unsigned)time(&timeSeed));
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            grid[i][j] = rand() % 3;
        }
    }
}

//print grid
void printGrid(int n, int **grid) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%d ", grid[i][j]);
            if (j == n-1) {
                printf("\n\n");
            }
        }
    }
}

//red color movement
void redMove(int row,
             int column,
             int **grid) {
    int i, j;
    for (i = 0; i < column; i++) {
        if (grid[i][0] == red && grid[i][1] == white) {
            grid[i][0] = 4;
            grid[i][1] = 3;
        }
        for (j = 1; j < row; j++) {
            if (grid[i][j] == red && (grid[i][(j + 1) % row] == white)) {
                grid[i][j] = white;
                grid[i][(j + 1) % row] = 3;
            }
            else if (grid[i][j] == 3)
                grid[i][j] = red;
        }
        if (grid[i][0] == 3)
            grid[i][0] = red;
        else if (grid[i][0] == 4)
            grid[i][0] = white;
    }
}

//blue color movement
void blueMove(int row,
              int column,
              int **grid) {
    int i, j;
    for (j = 0; j < row; j++) {
        if (grid[0][j] == blue && grid[1][j] == white) {
            grid[0][j] = 4;
            grid[1][j] = 3;
        }
        for (i = 1; i < column; i++) {
            if (grid[i][j] == blue && grid[(i + 1) % column][j] == white) {
                grid[i][j] = white;
                grid[(i + 1) % column][j] = 3;
            }
            else if (grid[i][j] == 3)
                grid[i][j] = blue;
        }
        if (grid[0][j] == 3)
            grid[0][j] = blue;
        else if (grid[0][j] == 4)
            grid[0][j] = white;
    }
}

//calculate rows distributed to each process
int* distributeRowToProcesses(int numprocs,  //number if processes
                              int n,  //n*n cells in grid
                              int t   //t*t tiles in grid
) {
    int *rowProcs = (int *)malloc(sizeof(int) * (numprocs - 1));
    int tileRowsCount = n / t;  //how much rows in one tile
    int a = t / (numprocs - 1); //how much tiles should a process have
    int b = t % (numprocs - 1); //tiles not distributed
    int i;
    // calculate the tile row distributed to each process
    for (i = 0; i < (numprocs - 1); i++) {
        rowProcs[i] = a;
    }
    for (i = 0; i < b; i++) {
        rowProcs[i] = rowProcs[i] + 1;
    }
    // calculate the row distributed to each process
    for (i = 0; i < (numprocs - 1); i++) {
        rowProcs[i] = rowProcs[i] * tileRowsCount;
    }
    return rowProcs;
}

//sequential computation
void sequentialComputation(int **grid,
                           int tileCount,
                           int n,
                           int t,
                           int c,
                           int max_itrs) {
    int finished = 0; //whether finished
    int itrs = 0;
    int type = SEQUENTIAL;
    float *tileInfoArray = (float *)malloc(sizeof(float) * tileCount * 3);	// store the red and blue ratio in grid
    printf("The sequential computation starts !\n");
    while (!finished && itrs < max_itrs) {
        itrs = itrs + 1;
        redMove(n, n, grid);
        blueMove(n, n, grid);
        finished = tileChecker(grid, tileCount, tileInfoArray, n, t, c, type); //check each tile
        printf("This is Number %d iteration, the grid is \n\n", itrs);
        printGrid(n, grid);
    }
    printf("The sequential computation result\n");
    printf("After %d interations, the final grid: \n", itrs);
    printGrid(n, grid);
    printResult(tileInfoArray, tileCount, t, n, grid);
}

//check the situation of each tile and calculate the color ratio for each tile
int tileChecker(int **grid,
                int tileCount,
                float *tileInfoArray,
                int n,
                int t,
                int c,
                int type) {
    int redcount = 0, bluecount = 0;
    int tileRow, tileColumn;
    float redRatio, blueRatio;
    int finished = 0;
    int temp = n / t; //rows in each tile
    float cellsInTile = temp * temp; //cells in each tile
    int i, j, k;

    for (i = 0; i < tileCount; i++) {
        tileRow = i / t;
        tileColumn = i % t;
        for (j = temp * tileRow; j < temp * (tileRow + 1); j++) {
            for (k = temp * tileColumn; k < temp * (tileColumn + 1); k++) {
                if ( type == PARALLEL ) {
                    if ( grid[j + 1][k] == 1 ) {
                        redcount += 1;
                    }
                    if ( grid[j + 1][k] == 2 ) {
                        bluecount += 1;
                    }
                } else {
                    if ( grid[j][k] == 1 ) {
                        redcount += 1;
                    }
                    if ( grid[j][k] == 2 ) {
                        bluecount += 1;
                    }
                }
            }
        }
        redRatio = (redcount * 100) / cellsInTile;
        blueRatio = (bluecount * 100) / cellsInTile;
        //set default value
        tileInfoArray[3 * i] = 0;
        //record red and blue ratio
        tileInfoArray[3 * i + 1] = redRatio;
        tileInfoArray[3 * i + 2] = blueRatio;
        //whether exceed threshold
        if ( redRatio > c ) {
            tileInfoArray[3 * i] = REDEXCEED;
            finished = 1;
        }
        if ( blueRatio > c ) {
            tileInfoArray[3 * i] = BLUEEXCEED;
            finished = 1;
        }
        redcount = 0;
        bluecount = 0;
    }
    return finished;
}

//print result
void printResult(float *tileInfoArray,
                 int tileCounter,
                 int t,
                 int n,
                 int **grid) {
    int counter = 0;
    int tileRow, tileColumn;
    int temp = n / t;
    int i, j, k;

    for (i = 0; i < tileCounter; i++) {
        tileRow = i / t;
        tileColumn = i % t;
        if (tileInfoArray[3 * i] == REDEXCEED) {
            printf("This is [%d, %d] tile\n", tileRow, tileColumn);
            for (j = tileRow * temp; j < (tileRow + 1) * temp; j++) {
                for (k = tileColumn * temp; k < (tileColumn + 1) * temp; k++) {
                    printf("%d ", grid[j][k]);
                    if (k == (tileColumn + 1) * temp - 1) {
                        printf("\n");
                    }
                }
            }
            printf("In this tile, the red color exceeds the threshold with the red ratio %.2f %%, blue ratio %.2f %%.\n", tileInfoArray[3 * i + 1], tileInfoArray[3 * i + 2]);
            counter += 1;
        }
        if (tileInfoArray[3 * i] == BLUEEXCEED) {
            printf("This is [%d, %d] tile\n", tileRow, tileColumn);
            for (j = tileRow * temp; j < (tileRow + 1) * temp; j++) {
                for (k = tileColumn * temp; k < (tileColumn + 1) * temp; k++) {
                    printf("%d ", grid[j][k]);
                    if (k == (tileColumn + 1) * temp - 1) {
                        printf("\n");
                    }
                }
            }
            printf("In this tile, the blue color exceeds the threshold with the red ratio %.2f %%, blue ratio %.2f %%.\n", tileInfoArray[3 * i + 1], tileInfoArray[3 * i + 2]);
            counter += 1;
        }
    }
    if (counter == 0) {
        printf("There is no tile exceeding threshold.\n");
        printf("The red blue computation terminated as reaching the maximum iteration number!\n");
    }
    printf("\n");
}

//self check
void selfChecker(int **grid,
                 int **gridCopy,
                 int n) {
    int flag = 0;
    int i, j;
    printf("The result of self check is: \n");
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (grid[i][j] != gridCopy[i][j]) {
                flag = 1;
                goto RESULT;
            }
        }
    }
    RESULT:
    if (flag == 0) {
        printf("Parallel computation and sequential computation are same.\n");
    } else {
        printf("Parallel computation and sequential computation are different.\n");
    }
    printf("\n");
}