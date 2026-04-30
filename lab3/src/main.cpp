#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <mpi.h>

using vec = std::vector<int>;

void load(const char* path, int& dim, vec& data) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Failed to open: " << path << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
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
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    out << dim << "\n";
    for (int r = 0; r < dim; r++) {
        for (int c = 0; c < dim; c++)
            out << data[r * dim + c] << " \n"[c == dim - 1];
    }
}

void transpose(const vec& src, vec& dst, int n) {
    dst.resize(n * n);
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++)
            dst[c * n + r] = src[r * n + c];
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 4) {
        if (rank == 0)
            std::cerr << "Usage: " << argv[0] << " <A> <B> <out>\n";
        MPI_Finalize();
        return 1;
    }

    int n = 0;
    vec a_full, bt;

    if (rank == 0) {
        vec b_full;
        load(argv[1], n, a_full);
        int nb;
        load(argv[2], nb, b_full);
        if (n != nb) {
            std::cerr << "Dimension mismatch: " << n << " vs " << nb << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        transpose(b_full, bt, n);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
        bt.resize(n * n);
    MPI_Bcast(bt.data(), n * n, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> row_counts(nprocs), row_displs(nprocs);
    int base_rows = n / nprocs;
    int extra = n % nprocs;
    for (int p = 0; p < nprocs; p++) {
        row_counts[p] = base_rows + (p < extra ? 1 : 0);
        row_displs[p] = (p == 0) ? 0 : row_displs[p - 1] + row_counts[p - 1];
    }

    std::vector<int> send_counts(nprocs), send_displs(nprocs);
    for (int p = 0; p < nprocs; p++) {
        send_counts[p] = row_counts[p] * n;
        send_displs[p] = row_displs[p] * n;
    }

    int my_rows = row_counts[rank];
    vec local_a(my_rows * n);

    MPI_Scatterv(a_full.data(), send_counts.data(), send_displs.data(), MPI_INT,
                 local_a.data(), my_rows * n, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    vec local_c(my_rows * n, 0);
    for (int i = 0; i < my_rows; i++) {
        for (int j = 0; j < n; j++) {
            int acc = 0;
            for (int k = 0; k < n; k++)
                acc += local_a[i * n + k] * bt[j * n + k];
            local_c[i * n + j] = acc;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    vec c_full;
    if (rank == 0)
        c_full.resize(n * n);

    MPI_Gatherv(local_c.data(), my_rows * n, MPI_INT,
                c_full.data(), send_counts.data(), send_displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        double sec = t1 - t0;
        long long flops = 2LL * n * n * n - (long long)n * n;
        save(argv[3], n, c_full);
        std::cout << n << "," << nprocs << "," << flops << ","
                  << sec << "," << flops / sec / 1e6 << "\n";
    }

    MPI_Finalize();
    return 0;
}
