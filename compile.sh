#!/bin/bash

nvcc matrixMultiply.cu -std=c++11 -lcublas $1