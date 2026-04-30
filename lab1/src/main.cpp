#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstdlib>

using vec2d = std::vector<std::vector<int>>;

vec2d load(const char* path, int& dim) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Failed to open: " << path << "\n";
        std::exit(1);
    }
    in >> dim;
    vec2d m(dim, std::vector<int>(dim));
    for (int r = 0; r < dim; r++)
        for (int c = 0; c < dim; c++)
            in >> m[r][c];
    return m;
}

void save(const char* path, const vec2d& m) {
    int dim = m.size();
    std::ofstream out(path);
    if (!out) {
        std::cerr << "Failed to write: " << path << "\n";
        std::exit(1);
    }
    out << dim << "\n";
    for (int r = 0; r < dim; r++) {
        for (int c = 0; c < dim; c++)
            out << m[r][c] << " \n"[c == dim - 1];
    }
}

vec2d matmul(const vec2d& a, const vec2d& b, int n) {
    vec2d bt(n, std::vector<int>(n));
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++)
            bt[c][r] = b[r][c];

    vec2d res(n, std::vector<int>(n, 0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int acc = 0;
            for (int k = 0; k < n; k++)
                acc += a[i][k] * bt[j][k];
            res[i][j] = acc;
        }
    }
    return res;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <A> <B> <out>\n";
        return 1;
    }

    int na, nb;
    auto a = load(argv[1], na);
    auto b = load(argv[2], nb);

    if (na != nb) {
        std::cerr << "Dimension mismatch: " << na << " vs " << nb << "\n";
        return 1;
    }

    long long flops = 2LL * na * na * na - (long long)na * na;

    auto t0 = std::chrono::steady_clock::now();
    auto c = matmul(a, b, na);
    auto t1 = std::chrono::steady_clock::now();

    double sec = std::chrono::duration<double>(t1 - t0).count();

    save(argv[3], c);

    std::cout << na << "," << flops << "," << sec << ","
              << flops / sec / 1e6 << "\n";
    return 0;
}
