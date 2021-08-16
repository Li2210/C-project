#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

#define white 0
#define red 1
#define blue 2
#define REDEXCEED 5
#define BLUEEXCEED 6

void createGrid(int gridWidth, int gridHeight, int **grid1D, int ***grid);
void initGrid(int n, int **grid);
void printGrid(int n, int **grid);
void redMove(int row_start, int row_end, int **grid, int n);
void blueMove(int row_start, int end, int **grid, int n);
int *distributeWorkToThread(int threadsNumber, int workNumber);
void sequentialComputation(int **grid, int n, int t, float c, int max_iters, int tileCount);
int tileChecker(int **grid, float *tileInfoArray, int n, int t, int c, int tile_start, int tile_end);
void printResult(float *tileInfoArray, int tileCounter, int t, int n, int **grid);
void *redBlueComputation(void *thrd_arg);
void selfChecker(int **grid, int **gridCopy, int n);

// struct for thread data
struct thrd_data{
    int id;
    int start_row;
    int end_row;
    int start_tile;
    int end_tile;
};

// struct for barrier function
typedef struct {
    pthread_mutex_t count_lock;     /* mutex semaphore for the barrier */
    pthread_cond_t ok_to_proceed;   /* condition variable for leaving */
    int count;                      /* count of the number who have arrived */
} mylib_barrier_t;

int *initialGrid_1D;
int **initialGrid;
int *initialGrid_1D_copy;
int **initialGrid_copy;
int n_threads, N, T, Max_iters;
float C;
int n_tile;
int finished = 0;
int n_iters = 0;
float* tileInfoArrayResult;

mylib_barrier_t barrier;

void mylib_barrier_init(mylib_barrier_t *b);
void mylib_barrier(mylib_barrier_t *b, int num_threads);
void mylib_barrier_destroy(mylib_barrier_t *b);

int main(int argc, char **argv) {
    if (argc != 6) {
        printf("Please enter the command in the following format:\n");
        printf("./pthreads_red_blue_computation [the number of threads] [cell grid size] [tile grid size] [terminating threshold] [maximum number of iterations]\n");
        printf("Note: [cell grid size] %% [tile grid size] = 0; [the number of threads >= 0]\n");
        printf("\n");
        return 0;
    }

    n_threads = atoi(argv[1]);
    N = atoi(argv[2]);
    T = atoi(argv[3]);
    C = atoi(argv[4]);
    Max_iters = atoi(argv[5]);

    if ((N % T != 0) || (n_threads < 0)) {
        printf("Please enter the command in the following format:\n");
        printf("./pthreads_red_blue_computation [the number of threads] [cell grid size] [tile grid size] [terminating threshold] [maximum number of iterations]\n");
        printf("Note: [cell grid size] %% [tile grid size] = 0; [the number of threads >= 0]\n");
        printf("\n");
        return 0;
    }

    int i;

    //tile_volume = (N * N) / (T * T);
    n_tile = T * T;

    pthread_t *thread_id;
    struct thrd_data *t_arg;
    thread_id = (pthread_t *)malloc(sizeof(pthread_t)*n_threads);
    t_arg = (struct thrd_data *)malloc(sizeof(struct thrd_data)*n_threads);
    //calculate the row number for every thread (difference no more than one row)
    int *row_for_thread = distributeWorkToThread(n_threads, N);
    //calculate the tile number for every thread to check (difference no more than one tile)
    int *tile_for_Thread = distributeWorkToThread(n_threads, n_tile);

    tileInfoArrayResult = (float *)malloc(sizeof(float) * n_tile * 3);
    // initialize the barrier
    mylib_barrier_init(&barrier);

    createGrid(N, N, &initialGrid_1D, &initialGrid);
    createGrid(N, N, &initialGrid_1D_copy, &initialGrid_copy);
    initGrid(N, initialGrid);
    memcpy(initialGrid_1D_copy, initialGrid_1D, sizeof(int) * N * N);
    printf("The initial grid: \n");
    printGrid(N, initialGrid);

    if (n_threads == 1) {
        sequentialComputation(initialGrid, N, T, C, Max_iters, n_tile);
    } else {
        // distribute the workload among the threads
        for (i = 0; i < n_threads; i++) {
            //assign id
            t_arg[i].id = i;
            if (i == 0) {
                //first thread start from row 0, tile 0
                t_arg[i].start_row = 0;
                t_arg[i].start_tile = 0;
            } else {
                //other thread start from the end of previous thread
                t_arg[i].start_row = t_arg[i - 1].end_row;
                t_arg[i].start_tile = t_arg[i - 1].end_tile;
            }
            //end = start + num of work assigned
            t_arg[i].end_row = t_arg[i].start_row + row_for_thread[i];
            t_arg[i].end_tile = t_arg[i].start_tile + tile_for_Thread[i];
            pthread_create(&thread_id[i], NULL, redBlueComputation, (void *) &t_arg[i]);
        }

        printf("\n");
        printf("The parallel computation result: \n");
        for (i = 0; i < n_threads; i++) {
            pthread_join(thread_id[i], NULL);
        }

        printf("After %d interations, the final grid: \n", n_iters);
        printGrid(N, initialGrid);
        printResult(tileInfoArrayResult, n_tile, T, N, initialGrid);

        sequentialComputation(initialGrid_copy, N, T, C, Max_iters, n_tile);
        selfChecker(initialGrid, initialGrid_copy, N);

        mylib_barrier_destroy(&barrier);
        pthread_exit(NULL);
    }
}

// barrier initialization
void mylib_barrier_init(mylib_barrier_t *b) {
    b->count = 0;
    pthread_mutex_init(&b->count_lock, NULL);
    pthread_cond_init(&b->ok_to_proceed, NULL);
}

// barrier function
void mylib_barrier(mylib_barrier_t *b, int num_threads) {
    pthread_mutex_lock(&b->count_lock);
    b->count += 1;
    if (b->count == num_threads) {
        b->count = 0;
        pthread_cond_broadcast(&b->ok_to_proceed);
    } else {
        pthread_cond_wait(&b->ok_to_proceed, &b->count_lock);
    }
    pthread_mutex_unlock(&b->count_lock);
}

// barrier destroy
void mylib_barrier_destroy(mylib_barrier_t *b) {
    pthread_mutex_destroy(&b->count_lock);
    pthread_cond_destroy(&b->ok_to_proceed);
}

// create grid
void createGrid(int gridWidth, int gridHeight, int **grid1D, int ***grid) {
    int count = gridWidth * gridHeight;
    *grid1D = (int *)malloc(sizeof(int) * count);
    *grid = (int **)malloc(sizeof(int *) * gridHeight);
    int i;
    for (i = 0; i < gridHeight; i++) {
        (*grid)[i] = &((*grid1D)[i * gridWidth]);
    }
}

// initialize grid
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

// print grid
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

// red color movement
void redMove(int row_start, int row_end, int **grid, int n) {
    int i, j;
    for (i = row_start; i < row_end; i++) {
        if ( grid[i][0] == red && grid[i][1] == white) {
            grid[i][0] = 4;
            grid[i][1] = 3;
        }
        for (j = 1; j < n; j++){
            if ( grid[i][j] == red && (grid[i][(j + 1) % n] == white) ) {
                grid[i][j] = white;
                grid[i][(j + 1) % n] = 3;
            }
            else if (grid[i][j] == 3)
                grid[i][j] = red;
        }
        if ( grid[i][0] == 3)
            grid[i][0] = red;
        else if (grid[i][0] == 4)
            grid[i][0] = white;
    }
}

// blue color movement
void blueMove(int row_start, int end, int **grid, int n) {
    int i, j;
    for (j = row_start; j < end; j++) {
        if (grid[0][j] == blue && grid[1][j] == white) {
            grid[0][j] = 4;
            grid[1][j] = 3;
        }
        for (i = 1; i < n; i++) {
            if (grid[i][j] == blue && grid[(i + 1) % n][j] == white) {
                grid[i][j] = white;
                grid[(i + 1) % n][j] = 3;
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

int *distributeWorkToThread(int threadsNumber, int workNumber) {
    int *work = (int*)malloc(sizeof(int) * (threadsNumber));
    int a =  workNumber / threadsNumber;
    int b = workNumber % threadsNumber;
    int i;
    for (i = 0; i < threadsNumber; i++) {
        work[i] = a;
    }
    for (i = 0; i < b; i++) {
        work[i] = work[i] + 1;
    }
    return work;
}

// sequential movement
void sequentialComputation(int **grid, int n, int t, float c, int max_iters, int tileCount) {
    int Sequential_finished = 0; //whether finished
    int iters = 0;
    float *tileInfo = (float *)malloc(sizeof(float) * tileCount * 3);
    printf("The sequential computation starts !\n");
    while (!Sequential_finished && iters < max_iters) {
        iters = iters + 1;// count for the iteration number
        redMove(0, n, grid, n);
        blueMove( 0, n, grid, n);
        Sequential_finished = tileChecker(grid, tileInfo, n, t, c, 0, tileCount);
        printf("This is Number %d iteration, the grid is \n\n", iters);
        printGrid(n, grid);
    }
    printf("After %d interations, the final grid: \n", iters);
    printGrid(n, grid);
    printResult(tileInfo, tileCount, t, n, grid);
}

//check the situation of each tile and calculate the color ratio for each tile
int tileChecker(int **grid, float *tileInfo, int n, int t, int c, int tile_start, int tile_end) {

    int redcount = 0, bluecount = 0;
    int tileRow, tileColumn;
    float redRatio, blueRatio;
    int finished_flag = 0;

    int temp = n / t; //rows in each tile
    float cellsInTile = temp * temp; //cells in each tile

    int i, j, k;

    for (i = tile_start; i < tile_end; i++) {
        tileRow = i / t;   //属于第几行
        tileColumn = i % t;  //属于第几列
        for (j = temp * tileRow; j < temp * (tileRow + 1); j++) {
            for (k = temp * tileColumn; k < temp * (tileColumn + 1); k++) {
                if ( grid[j][k] == 1 ) {
                        redcount += 1;
                }
                if ( grid[j][k] == 2 ) {
                        bluecount += 1;
                }
            }
        }

        redRatio = (redcount * 100) / cellsInTile;
        blueRatio = (bluecount * 100) / cellsInTile;
        //set default value
        tileInfo[3 * i] = 0;
        //record red and blue ratio
        tileInfo[3 * i + 1] = redRatio;
        tileInfo[3 * i + 2] = blueRatio;

        //whether exceed threshold
        if ( redRatio > c ) {
            tileInfo[3 * i] = REDEXCEED;
            finished_flag = 1;
        }
        if ( blueRatio > c ) {
            tileInfo[3 * i] = BLUEEXCEED;
            finished_flag = 1;
        }
        redcount = 0;
        bluecount = 0;
    }
    return finished_flag;
}

void *redBlueComputation(void *thrd_arg) {
    int finished_flag = 0;
    int iters_p = 0;   // the iteration times
    struct thrd_data *t_data;
    t_data = (struct thrd_data *)thrd_arg;

    while (!finished && iters_p < Max_iters) {
        iters_p += 1;   // renew the iteration number
        redMove(t_data->start_row, t_data->end_row, initialGrid, N);
        mylib_barrier(&barrier, n_threads);
        blueMove(t_data->start_row, t_data->end_row, initialGrid, N);
        mylib_barrier(&barrier, n_threads);
        finished_flag = tileChecker(initialGrid, tileInfoArrayResult, N, T, C, t_data->start_tile, t_data->end_tile);
        if (finished_flag == 1) {
            finished = finished_flag;
        }
        mylib_barrier(&barrier, n_threads);
    }

    if (t_data ->id == 0) {
        n_iters = iters_p;
    }

    pthread_exit(NULL);
}

void printResult(float *tileInfo, int tileCounter, int t, int n, int **grid) {
    int counter = 0;
    int tileRow, tileColumn;
    int temp = n / t;
    int i, j, k;

    for (i = 0; i < tileCounter; i++) {
        tileRow = i / t;
        tileColumn = i % t;
        if (tileInfo[3 * i] == REDEXCEED) {
            printf("This is [%d, %d] tile\n", tileRow, tileColumn);
            for (j = tileRow * temp; j < (tileRow + 1) * temp; j++) {
                for (k = tileColumn * temp; k < (tileColumn + 1) * temp; k++) {
                    printf("%d ", grid[j][k]);
                    if (k == (tileColumn + 1) * temp - 1) {
                        printf("\n");
                    }
                }
            }
            printf("In this tile, the red color exceeds the threshold with the red ratio %.2f %%, blue ratio %.2f %%.\n", tileInfo[3 * i + 1], tileInfo[3 * i + 2]);
            counter += 1;
        }
        if (tileInfo[3 * i] == BLUEEXCEED) {
            printf("This is [%d, %d] tile\n", tileRow, tileColumn);
            for (j = tileRow * temp; j < (tileRow + 1) * temp; j++) {
                for (k = tileColumn * temp; k < (tileColumn + 1) * temp; k++) {
                    printf("%d ", grid[j][k]);
                    if (k == (tileColumn + 1) * temp - 1) {
                        printf("\n");
                    }
                }
            }
            printf("In this tile, the blue color exceeds the threshold with the red ratio %.2f %%, blue ratio %.2f %%.\n", tileInfo[3 * i + 1], tileInfo[3 * i + 2]);
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
void selfChecker(int **grid, int **gridCopy, int n) {
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
        printf("ERROR!\n");
    }
    printf("\n");
}

