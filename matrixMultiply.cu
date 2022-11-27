#include <cublas_v2.h>
#include <fstream>
#include <iostream>
#include <random>

using namespace std;

const int TILE_LENGTH = 32;

// checkbounds A,B,C
#define CBA ((block * TILE_LENGTH + threadIdx.y) < nElements && col < nElements)
#define CBB ((block * TILE_LENGTH + threadIdx.x) < nElements && row < nElements)
#define CBC (row < nElements && col < nElements)

void DisplayHeader()
{
   const int kb = 1024;
   const int mb = kb * kb;
   wcout << "NBody.GPU" << endl << "=========" << endl << endl;

   wcout << "CUDA version:   v" << CUDART_VERSION << endl;
   // wcout << "Thrust version: v" << THRUST_MAJOR_VERSION << "." <<
   // THRUST_MINOR_VERSION << endl << endl;

   int devCount;
   cudaGetDeviceCount(&devCount);
   wcout << "CUDA Devices: " << endl << endl;

   for (int i = 0; i < devCount; ++i) {
      cudaDeviceProp props;
      cudaGetDeviceProperties(&props, i);
      wcout << i << ": " << props.name << ": " << props.major << "."
            << props.minor << endl;
      wcout << "  Global memory:   " << props.totalGlobalMem / mb << "mb"
            << endl;
      wcout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb"
            << endl;
      wcout << "  Constant memory: " << props.totalConstMem / kb << "kb"
            << endl;
      wcout << "  Block registers: " << props.regsPerBlock << endl << endl;

      wcout << "  Warp size:         " << props.warpSize << endl;
      wcout << "  Threads per block: " << props.maxThreadsPerBlock << endl;
      wcout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", "
            << props.maxThreadsDim[1] << ", " << props.maxThreadsDim[2] << " ]"
            << endl;
      wcout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", "
            << props.maxGridSize[1] << ", " << props.maxGridSize[2] << " ]"
            << endl;
      wcout << endl;
   }
}

// inspired by this
// https://stackoverflow.com/questions/18815489/cuda-tiled-matrix-matrix-multiplication-with-shared-memory-and-matrix-size-whic
__global__ void optMM(const double* a, const double* b, double* c,
                      const long nElements, const int nBlocks)
{
   // positions in the blocked matrix which could be out of bounds
   long row = (long)TILE_LENGTH * blockIdx.y + threadIdx.y;
   long col = (long)TILE_LENGTH * blockIdx.x + threadIdx.x;

   // shared memory tiles for a and b
   __shared__ double tileA[TILE_LENGTH][TILE_LENGTH];
   __shared__ double tileB[TILE_LENGTH][TILE_LENGTH];

   double cValue = 0.0;

   for (int block = 0; block < nBlocks; block++) {

      // fill tiles and pad tiles

      if (CBB) {
         tileB[threadIdx.y][threadIdx.x] =
          b[(row * nElements + block * TILE_LENGTH + threadIdx.x)];
      }
      
      if (CBA) {
         tileA[threadIdx.y][threadIdx.x] =
          a[((block * TILE_LENGTH + threadIdx.y) * nElements + col)];
      }
      

      // wait till A and B tiles are fully populated
      __syncthreads();

      for (int i = 0; i < TILE_LENGTH; i++) {

         cValue += tileA[i][threadIdx.x] * tileB[threadIdx.y][i];
      }
   }
   // CBC bitwise divergence doesn't work here because too many threads read 0
   // and write back 0. preventing the actual thread from working
   if (CBC) {
      c[(row * nElements + col)] += cValue;
   }

   __syncthreads();
}

// NAIVE
__global__ void matrixMultiply(const double* a, const double* b, double* c,
                               const long nElements)
{

   long i = (long)blockDim.x * blockIdx.x + threadIdx.x;
   long row = i / nElements;
   long col = i % nElements;

   if (i < nElements * nElements) {
      for (int k = 0; k < nElements; k++) {
         c[i] += b[row * nElements + k] * a[k * nElements + col];
      }
   }
}

// row-major -> col-major
double* transpose(double* X, long N)
{
   double* Y = (double*)malloc(sizeof(double) * N * N);
   for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
         Y[j * N + i] = X[i * N + j];
      }
   }
   return Y;
}

void printVector(const int* vec, const int size)
{
   for (int i = 0; i < size; i++) {
      cout << vec[i] << ", ";
   }
}

// generate matrix of random numbers between -10 and 10
void generateRandomMV(double* A, long size)
{
   for (double* ptr = A; size > 0; ptr++, size--) {

      *ptr = (double)(rand() % 100) * 1.456344;
   }
}

double profileKernel(int nBlocks, int nThreads, int nElements)
{
   // These variables are used to convert occupancy to warps
   int device;
   cudaDeviceProp prop;
   int activeWarps;
   int maxWarps;

   cudaGetDevice(&device);
   cudaGetDeviceProperties(&prop, device);

   cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nBlocks, matrixMultiply,
                                                 nThreads, 0);

   activeWarps = nBlocks * nThreads / prop.warpSize;
   maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

   std::cout << "Occupancy: " << (double)activeWarps / maxWarps * 100 << "%"
             << std::endl;
   return (double)activeWarps / maxWarps * 100;
}

void matrixMultiplyTest(long nElements, int nThreads, int nBlocks,
                        ofstream& output, int type)
{
   printf("Elements: %d\nThreads: %d\nBlocks: %d\n", nElements, nThreads,
          nBlocks);
   printf("TOTAL ELEMENTS: %d\n", nElements * nElements);
   // host matrices
   double *a, *b, *c;
   // device vectors
   double *da, *db, *dc;
   long matrixSize = sizeof(double) * nElements * nElements;

   // setup memory on host
   a = (double*)malloc(matrixSize);
   b = (double*)malloc(matrixSize);
   c = (double*)calloc(nElements * nElements, sizeof(double));

   generateRandomMV(a, nElements * nElements);
   generateRandomMV(b, nElements * nElements);

   // allocate memory on device
   cudaMalloc((void**)&da, matrixSize);
   cudaMalloc((void**)&db, matrixSize);
   cudaMalloc((void**)&dc, matrixSize);

   cudaEvent_t startEvent, stopEvent, startCalc, endCalc;
   cudaEventCreate(&startEvent);
   cudaEventCreate(&stopEvent);
   cudaEventCreate(&startCalc);
   cudaEventCreate(&endCalc);

   // start timer
   cudaEventRecord(startEvent);

   // copy inputs to device
   cudaMemcpy(da, a, matrixSize, cudaMemcpyHostToDevice);
   cudaMemcpy(db, b, matrixSize, cudaMemcpyHostToDevice);
   cudaMemcpy(dc, c, matrixSize, cudaMemcpyHostToDevice);

   cudaEventRecord(startCalc);
   // comput vector addition on device
   if (type == 0) {
      matrixMultiply<<<nBlocks, nThreads>>>(da, db, dc, nElements);
   } else if (type == 1) {
      cublasHandle_t handle;
      cublasCreate(&handle);
      double alpha = 1.0, beta = 0.0;
      cublasDgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, nElements, nElements,
                     nElements, &alpha, da, nElements, db, nElements, &beta, dc,
                     nElements);
   } else if (type == 2) {
      int gridDim = (int)ceil(sqrt(nBlocks));
      dim3 grid(gridDim, gridDim);
      dim3 threads(TILE_LENGTH, TILE_LENGTH);
      optMM<<<grid, threads>>>(da, db, dc, nElements, gridDim);
   }

   cudaEventRecord(endCalc);

   // Copy result back to host
   cudaMemcpy(c, dc, matrixSize, cudaMemcpyDeviceToHost);

   cudaEventRecord(stopEvent);

   cudaEventSynchronize(stopEvent);

   float elapsedTime;
   float calcTime;
   cudaEventElapsedTime(&calcTime, startCalc, endCalc);
   cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);

   cout << "Elapsed time total: " << elapsedTime << " ms\n";

   cudaEventDestroy(startEvent);
   cudaEventDestroy(stopEvent);
   cudaEventDestroy(endCalc);
   cudaEventDestroy(startCalc);

   // print a b and c vectors for verification
   // cout << "A = ";
   // printVector(a, nElements);
   // cout << "\n";
   // cout << "B = ";
   // printVector(b, nElements);
   // cout << "\n";
   // cout << "C = ";
   // printVector(c, nElements);
   // cout << "\n";
   bool isRight = true;
   // check if correct
   if (nElements < 10000) {
      double* cTrue = (double*)calloc(nElements * nElements, sizeof(double));
      double* dcTrue;
      cudaMalloc((void**)&dc, matrixSize);
      cudaMalloc((void**)&dcTrue, matrixSize);
      cudaMemcpy(dcTrue, cTrue, matrixSize, cudaMemcpyHostToDevice);
      cudaMemcpy(da, a, matrixSize, cudaMemcpyHostToDevice);

      double alpha = 1.0, beta = 0.0;
      cublasHandle_t handle;
      cublasCreate(&handle);
      cublasStatus_t flag = cublasDgemm_v2(
          handle, CUBLAS_OP_N, CUBLAS_OP_N, nElements, nElements, nElements,
          &alpha, da, nElements, db, nElements, &beta, dcTrue, nElements);
      printf("%d\n", flag);

      cudaMemcpy(cTrue, dcTrue, matrixSize, cudaMemcpyDeviceToHost);

      for (int i = 0; i < nElements * nElements; i++) {
         if (abs(cTrue[i] - c[i]) > 1) {
            printf("INCORRECT\n");
            isRight = false;
            break;
         }
      }
      int i, j;
      printf(" Top left corner of matrix C: \n");
      for (i = 0; i < min(nElements, (long)10); i++) {
         for (j = 0; j < min(nElements, (long)10); j++) {
            printf("%12.0f", c[j + i * nElements]);
         }
         printf("\n");
      }

      printf("\n Top left corner of matrix CORRECT C: \n");
      for (i = 0; i < min(nElements, (long)10); i++) {
         for (j = 0; j < min(nElements, (long)10); j++) {
            printf("%12.0f", cTrue[j + i * nElements]);
         }
         printf("\n");
      }

      cudaFree(cTrue);
      cublasDestroy(handle);
   }

   output << type << "," << nElements << "," << nThreads << "," << nBlocks
          << "," << elapsedTime << "," << calcTime << "," << isRight << ","
          << profileKernel(nBlocks, nThreads, nElements) << ",\n";

   free(a);
   free(b);
   free(c);
   // Cleanup
   cudaFree(da);
   cudaFree(db);
   cudaFree(dc);
   cudaDeviceReset();
}

// TYPE 0 is naive
// TYPE 1 is cublas
int main()
{
   cudaDeviceReset();

   // DisplayHeader();
   ofstream output;
   output.open("test.csv");
   // int maxElements = pow(2, 20);
   // int startElements = pow(2, 5);
   int nThreads = 1024;
   output << "Type,Elements(NxN),Threads,Blocks,Full Execution time (ms), "
             "Calculation "
             "Execution time (ms), isCorrect, Occupancy (%),\n";

   for (int type = 0; type < 3; type++) {
      for (long nElements = 10; nElements <= 11000; nElements += 100) {
         matrixMultiplyTest(
             nElements, nThreads,
             // use only as many blocks as needed
             (int)ceil((double)(nElements * nElements) / (double)nThreads),
             output, type);
      }
   }

   output.close();
}