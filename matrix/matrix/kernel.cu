#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <malloc.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_DIM 16

__global__ void transposition(int* matrix, int* matrixOut, int length, int width)
{
	__shared__ int tempMatrix[BLOCK_DIM][BLOCK_DIM];//разделяемая память

	int temp;

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < length) && (j < width))
	{
		temp = j * length + i;
		tempMatrix[threadIdx.y][threadIdx.x] = matrix[temp];
	}

	__syncthreads();

	i = blockIdx.y * blockDim.y + threadIdx.x;//индекс блока, размерность блока, индекс потока
	j = blockIdx.x * blockDim.x + threadIdx.y;

	if ((i < width) && (j < length))
	{
		temp = j * width + i;
		matrixOut[temp] = tempMatrix[threadIdx.x][threadIdx.y];
	}
}

int main()
{
	int length, width, *matrixInput, *matrixOut, size, *devMatrixInput, *devOutputMatrix;
	while (1) {
		printf("Input width:");
		scanf("%d", &width);
		printf("Input length:");
		scanf("%d", &length);
		size = length * width * sizeof(int);
		matrixInput = (int *)malloc(size);
		matrixOut = (int *)malloc(size);
		for (int i = 0; i < (length * width); i++)
		{
			matrixInput[i] = i + 1;
		}
		printf("Input matrix:\n--------------------------------------------------------------------------------------\n");
		for (int i = 0; i < width; i++)
		{
			printf("|");
			for (int j = 0; j < length; j++)
			{
				printf("%-4d|", matrixInput[i*length + j]);
			}
			printf("\n--------------------------------------------------------------------------------------\n");
		}
		printf("\n\n\n");

		cudaMalloc((void**)&devMatrixInput, size);
		cudaMalloc((void**)&devOutputMatrix, size);

		cudaMemcpy(devMatrixInput, matrixInput, size, cudaMemcpyHostToDevice);

		dim3 dimBlock(BLOCK_DIM, BLOCK_DIM, 1);
		dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (length + dimBlock.y - 1) / dimBlock.y, 1);

		transposition<<<dimGrid, dimBlock>>>(devMatrixInput, devOutputMatrix, length, width);

		cudaThreadSynchronize();
		cudaMemcpy(matrixOut, devOutputMatrix, size, cudaMemcpyDeviceToHost);

		cudaFree(devMatrixInput);
		cudaFree(devOutputMatrix);

		printf("Output matrix:\n--------------------------------------------------------------------------------------\n");
		for (int i = 0; i < length; i++)
		{
			printf("|");
			for (int j = 0; j < width; j++)
			{
				printf("%-3d|", matrixOut[i*width + j]);
			}
			printf("\n--------------------------------------------------------------------------------------\n");
		}
		printf("\n\n\n");

		free(matrixInput);
		free(matrixOut);
	}
	return 0;
}