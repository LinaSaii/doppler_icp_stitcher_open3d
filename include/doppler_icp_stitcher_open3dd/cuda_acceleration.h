#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <future>

struct CudaPointCloud {
    float* points;           // [x1, y1, z1, x2, y2, z2, ...]
    float* normals;          // [nx1, ny1, nz1, nx2, ny2, nz2, ...]  
    float* velocities;
    int num_points;
    
    CudaPointCloud() : points(nullptr), normals(nullptr), velocities(nullptr), num_points(0) {}
};

class CudaAccelerator {
public:
    static CudaAccelerator& getInstance();
    
    // Async preprocessing - returns future to processed data
    std::future<PointCloudData> asyncPreprocess(const PointCloudData& input);
    
    // CUDA-accelerated KD-tree nearest neighbor search
    void cudaKDTreeSearch(const float* query_points, int num_queries,
                         const float* target_points, int num_targets,
                         int* indices, float* distances);
    
    // CUDA-accelerated matrix operations
    void cudaMatrixMultiply(const float* A, const float* B, float* C, 
                           int m, int n, int k);
    void cudaSolveLinearSystem(const float* A, const float* b, float* x, int n);
    
    // Memory management
    void uploadPointCloud(const PointCloudData& cpu_data, CudaPointCloud& gpu_data);
    void downloadPointCloud(const CudaPointCloud& gpu_data, PointCloudData& cpu_data);
    void freePointCloud(CudaPointCloud& gpu_data);
    
private:
    CudaAccelerator();
    ~CudaAccelerator();
    
    // Thread pool for async processing
    class ThreadPool;
    std::unique_ptr<ThreadPool> thread_pool_;
};
