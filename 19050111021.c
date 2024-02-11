#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>

#define MAX_ITERATIONS 10

void readMatrix(char *filename, double **matrix, int *rows, int *cols);
void writeVector(char *filename, double *vector, int size);
double dotProduct(double *a, double *b, int size);
void normalizeVector(double *vector, int size);
void parallelMatrixVectorMultiply(double *matrix, double *vector, double *result, int rows, int cols);
void parallelMatrixMatrixSubtract(double *matrixA, double *matrixB, double *result, int rows, int cols);
double parallelTwoNorm(double *vector, int size);

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <matrix_file>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double *eigenvector1, *eigenvector2;
    double *matrix;
    int rows, cols;

    if (rank == 0)
    {
        // Read matrix from file
        readMatrix(argv[1], &matrix, &rows, &cols);

        // Allocate memory for eigenvectors
        eigenvector1 = (double *)malloc(rows * sizeof(double));
        eigenvector2 = (double *)malloc(rows * sizeof(double));
        if (eigenvector1 == NULL || eigenvector2 == NULL)
        {
            fprintf(stderr, "Memory allocation failed.\n");
            exit(EXIT_FAILURE);
        }

        // Initialize eigenvectors
        srand(123); // Any integer as a seed
#pragma omp parallel for
        for (int i = 0; i < rows; i++)
        {
            eigenvector1[i] = 1.0;
            eigenvector2[i] = 1.0;
        }
    }

    // Broadcast matrix dimensions
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for local matrix
    double *localMatrix = (double *)malloc(rows * cols / size * sizeof(double));
    if (localMatrix == NULL)
    {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    // Scatter matrix to all processes
    MPI_Scatter(matrix, rows * cols / size, MPI_DOUBLE, localMatrix, rows * cols / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Allocate memory for local eigenvectors
    double *localEigenvector1 = (double *)malloc(rows * sizeof(double));
    double *localEigenvector2 = (double *)malloc(rows * sizeof(double));
    if (localEigenvector1 == NULL || localEigenvector2 == NULL)
    {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    // Initialize eigenvectors
    srand(123);
#pragma omp parallel for
    for (int i = 0; i < rows; i++)
    {
        localEigenvector1[i] = ((double)rand() / RAND_MAX) - 0.5;
        localEigenvector2[i] = ((double)rand() / RAND_MAX) - 0.5;
    }

    double eigenvalue1, eigenvalue2;
    double prevEigenvalue;

    // Elapsed time for power iteration
    double startTimePower = MPI_Wtime();

    // Perform power iteration
    for (int iter = 0; iter < MAX_ITERATIONS; iter++)
    {
        // Parallel matrix-vector multiplication
        parallelMatrixVectorMultiply(localMatrix, localEigenvector1, localEigenvector1, rows, cols);

        // Parallel normalization
        normalizeVector(localEigenvector1, rows);

        // Parallel eigenvalue calculation
        double localEigenvalue1 = dotProduct(localEigenvector1, localMatrix, rows);
        MPI_Allreduce(&localEigenvalue1, &eigenvalue1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Broadcast the updated eigenvector to all processes
        MPI_Bcast(localEigenvector1, rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Check convergence based on the difference between consecutive eigenvalues
        if (iter > 0)
        {
            double diffNorm = parallelTwoNorm(localEigenvector1, rows);
            if (diffNorm < 1e-6)
            {
                break;
            }
        }

        prevEigenvalue = eigenvalue1;
    }

    // Elapsed time measurement for power iteration
    double endTimePower = MPI_Wtime();

    // Shifted-power iteration
    double *shiftedMatrix = (double *)malloc(rows * cols * sizeof(double));
    if (shiftedMatrix == NULL)
    {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

#pragma omp parallel for
    for (int i = 0; i < rows * cols; i++)
    {
        int row = i / cols;
        int col = i % cols;
        shiftedMatrix[i] = localMatrix[i] - eigenvalue1 * localEigenvector1[row] * localEigenvector1[col];
    }

    double prevEigenvalue2;
    double startTimeShifted = MPI_Wtime();

    // Perform shifted-power iteration
    for (int iter = 0; iter < MAX_ITERATIONS; iter++)
    {
        // Parallel matrix-vector multiplication
        parallelMatrixVectorMultiply(shiftedMatrix, localEigenvector2, localEigenvector2, rows, cols);

        // Parallel normalization
        normalizeVector(localEigenvector2, rows);

        // Parallel eigenvalue calculation
        double localEigenvalue2 = dotProduct(localEigenvector2, shiftedMatrix, rows);
        MPI_Allreduce(&localEigenvalue2, &eigenvalue2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        MPI_Bcast(localEigenvector2, rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);


        if (iter > 0 && fabs(eigenvalue2 - prevEigenvalue2) < 1e-6)
        {
            break;
        }


        prevEigenvalue2 = eigenvalue2;
    }

    // Elapsed time measurement for shifted-power iteration
    double endTimeShifted = MPI_Wtime();


    if (rank == 0)
    {
        if (fabs(eigenvalue2 - prevEigenvalue2) >= 1e-6)
        {
            fprintf(stderr, "Warning: Shifted-power iteration did not converge for the second largest eigenvalue.\n");
        }
        printf("Largest Eigenvalue: %lf\n", eigenvalue1);
        printf("Second Largest Eigenvalue: %lf\n", eigenvalue2);
        printf("Elapsed Time (Power Iteration): %lf seconds\n", endTimePower - startTimePower);
        printf("Elapsed Time (Shifted-Power Iteration): %lf seconds\n", endTimeShifted - startTimeShifted);
    }

    // Gather eigenvector2 from all processes to rank 0
    MPI_Gather(localEigenvector2, rows / size, MPI_DOUBLE, eigenvector2, rows / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    if (rank == 0)
    {
        writeVector("eigenvector.txt", eigenvector2, rows);
    }

    // Clean
    if (rank == 0)
    {
        free(matrix);
        free(eigenvector1);
        free(eigenvector2);
    }
    free(localMatrix);
    free(localEigenvector1);
    free(localEigenvector2);
    free(shiftedMatrix);

    MPI_Finalize();

    return 0;
}


void readMatrix(char *filename, double **matrix, int *rows, int *cols)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        fprintf(stderr, "Error opening file %s.\n", filename);

        exit(EXIT_FAILURE);
    }

    fscanf(file, "%d %d", rows, cols);

    *matrix = (double *)malloc((*rows) * (*cols) * sizeof(double));
    if (*matrix == NULL)
    {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < *rows; i++)
    {
        for (int j = 0; j < *cols; j++)
        {
            fscanf(file, "%lf", &(*matrix)[i * (*cols) + j]);
        }
    }

    fclose(file);
}

void writeVector(char *filename, double *vector, int size)
{
    FILE *file = fopen(filename, "w");
    if (file == NULL)
    {
        fprintf(stderr, "Error opening file.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < size; i++)
    {
        fprintf(file, "%lf\n", vector[i]);
    }

    fclose(file);
}

double dotProduct(double *a, double *b, int size)
{
    double result = 0.0;
#pragma omp parallel for reduction(+ : result)
    for (int i = 0; i < size; i++)
    {
        result += a[i] * b[i];
    }
    return result;
}

void normalizeVector(double *vector, int size)
{
    double norm = 0.0;
#pragma omp parallel for reduction(+ : norm)
    for (int i = 0; i < size; i++)
    {
        norm += vector[i] * vector[i];
    }

    if (norm == 0.0)
    {

        vector[0] = 1.0;
    }
    else
    {
        norm = sqrt(norm);

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            vector[i] /= norm;
        }
    }
}

void parallelMatrixVectorMultiply(double *matrix, double *vector, double *result, int rows, int cols)
{
#pragma omp parallel for
    for (int i = 0; i < rows; i++)
    {
        double localResult = 0.0;


#pragma omp parallel for reduction(+ : localResult)
        for (int j = 0; j < cols; j++)
        {
            localResult += matrix[i * cols + j] * vector[j];
        }


        result[i] = localResult;
    }
}

void parallelMatrixMatrixSubtract(double *matrixA, double *matrixB, double *result, int rows, int cols)
{
#pragma omp parallel for
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result[i * cols + j] = matrixA[i * cols + j] - matrixB[i] * matrixB[j];
        }
    }
}

double parallelTwoNorm(double *vector, int size)
{
    double norm = 0.0;
#pragma omp parallel for reduction(+ : norm)
    for (int i = 0; i < size; i++)
    {
        norm += vector[i] * vector[i];
    }

    return sqrt(norm);
}