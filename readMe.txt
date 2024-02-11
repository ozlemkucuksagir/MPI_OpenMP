mpicc -fopenmp 19050111021.c -o output -lm
mpirun -np 4 ./output testA.txt
mpirun -np 4 ./output BigA.txt