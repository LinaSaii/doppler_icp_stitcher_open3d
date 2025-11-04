#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <iostream>

__global__ void nearestNeighborKernel(const float* query_points, const float* target_points,
                                     int* indices, float* distances,
                                     int num_queries, int num_targets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_queries) return;
    
    float min_dist = FLT_MAX;
    int min_index = -1;
    
    float qx = query_points[idx * 3];
    float qy = query_points[idx * 3 + 1];
    float qz = query_points[idx * 3 + 2];
    
    for (int i = 0; i < num_targets; i++) {
        float tx = target_points[i * 3];
        float ty = target_points[i * 3 + 1];
        float tz = target_points[i * 3 + 2];
        
        float dx = qx - tx;
        float dy = qy - ty;
        float dz = qz - tz;
        float dist = dx*dx + dy*dy + dz*dz;
        
        if (dist < min_dist) {
            min_dist = dist;
            min_index = i;
        }
    }
    
    indices[idx] = min_index;
    distances[idx] = sqrtf(min_dist);
}

extern "C" void cudaKDTreeSearch(const float* query_points, int num_queries,
                                const float* target_points, int num_targets,
                                int* indices, float* distances) {
    float *d_query, *d_target;
    int *d_indices;
    float *d_distances;
    
    // Allocate device memory
    cudaMalloc(&d_query, num_queries * 3 * sizeof(float));
    cudaMalloc(&d_target, num_targets * 3 * sizeof(float));
    cudaMalloc(&d_indices, num_queries * sizeof(int));
    cudaMalloc(&d_distances, num_queries * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_query, query_points, num_queries * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target_points, num_targets * 3 * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (num_queries + blockSize - 1) / blockSize;
    nearestNeighborKernel<<<numBlocks, blockSize>>>(d_query, d_target, d_indices, d_distances, num_queries, num_targets);
    
    // Copy results back
    cudaMemcpy(indices, d_indices, num_queries * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(distances, d_distances, num_queries * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_query);
    cudaFree(d_target);
    cudaFree(d_indices);
    cudaFree(d_distances);
}
