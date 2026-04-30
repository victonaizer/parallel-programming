#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__    \
                      << " — " << cudaGetErrorString(err) << "\n";          \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

using vec = std::vector<int>;

void load(const char* path, int& dim, vec& data) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Failed to open: " << path << "\n";
        std::exit(1);
    }
    in >> dim;
    data.resize(dim * dim);
    for (int i = 0; i < dim * dim; i++)
        in >> data[i];
}

void save(const char* path, int dim, const vec& data) {
    std::ofstream out(path);
    if (!out) {
        std::cerr << "Failed to write: " << path << "\n";
        std::exit(1);
    }
    out << dim << "\n";
    for (int r = 0; r < dim; r++) {
        for (int c = 0; c < dim; c++)
            out << data[r * dim + c] << " \n"[c == dim - 1];
    }
}

__global__ void matmul_kernel(const int* a, const int* b, int* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int acc = 0;
        for (int k = 0; k < n; k++)
            acc += a[row * n + k] * b[k * n + col];
        c[row * n + col] = acc;
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <A> <B> <out> [block_size]\n";
        return 1;
    }

    int block_sz = 16;
    if (argc >= 5)
        block_sz = std::atoi(argv[4]);

    int na, nb;
    vec ha, hb;
    load(argv[1], na, ha);
    load(argv[2], nb, hb);

    if (na != nb) {
        std::cerr << "Dimension mismatch: " << na << " vs " << nb << "\n";
        return 1;
    }

    int n = na;
    size_t bytes = n * n * sizeof(int);

    int *da, *db, *dc;
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMalloc(&dc, bytes));

    CUDA_CHECK(cudaMemcpy(da, ha.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(block_sz, block_sz);
    dim3 grid((n + block_sz - 1) / block_sz, (n + block_sz - 1) / block_sz);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    matmul_kernel<<<grid, block>>>(da, db, dc, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double sec = ms / 1000.0;

    vec hc(n * n);
    CUDA_CHECK(cudaMemcpy(hc.data(), dc, bytes, cudaMemcpyDeviceToHost));

    save(argv[3], n, hc);

    long long flops = 2LL * n * n * n - (long long)n * n;

    // CSV: n, block_size, flops, time, mflops
    std::cout << n << "," << block_sz << "," << flops << ","
              << sec << "," << flops / sec / 1e6 << "\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));

    return 0;
}
