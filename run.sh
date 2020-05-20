#!/bin/bash

echo "Compiling..."
nvcc -std=c++11 main.cu -I./include -lcublas -lcurand -o gemm
echo "Runnning..."
srun -p gpu ./gemm --m 300 --n 400 --k 500 --iter 10
