# Лабораторная работа №4: Параллельное умножение матриц с CUDA

## Задание

Модифицировать программу из лабораторной работы №1 для параллельной работы по технологии CUDA. Провести эксперименты с разными размерами матриц и различными конфигурациями сетки блоков.

- **Исходные данные:** файлы, содержащие значения исходных матриц
- **Выходные данные:** файл со значениями результирующей матрицы, время выполнения, объём задачи
- **Верификация:** автоматизированная проверка корректности через Python (NumPy)

## Описание реализации

Программа переносит вычисление произведения матриц на GPU с помощью CUDA. Каждый элемент результирующей матрицы C вычисляется отдельным потоком (thread) на GPU.

### Ядро CUDA

```cuda
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
```

### Конфигурация запуска

Двумерная сетка блоков покрывает всю матрицу результата:

```cpp
dim3 block(block_sz, block_sz);       // например, 16x16 = 256 потоков
dim3 grid((n + block_sz - 1) / block_sz,
          (n + block_sz - 1) / block_sz);
```

Размер блока задаётся аргументом командной строки (8, 16 или 32).

### Алгоритм работы

1. Загрузка матриц A и B из файлов на хосте (CPU)
2. Выделение памяти на GPU (`cudaMalloc`)
3. Копирование A и B на GPU (`cudaMemcpy`, Host → Device)
4. Запуск ядра с замером времени через `cudaEvent`
5. Копирование результата C обратно на хост (Device → Host)
6. Сохранение результата в файл

### Замер времени

Используются CUDA Events для точного измерения времени работы ядра (без учёта пересылки данных):

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
matmul_kernel<<<grid, block>>>(da, db, dc, n);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms = 0;
cudaEventElapsedTime(&ms, start, stop);
```

## Исходный код

### main.cu

```cuda
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

    std::cout << n << "," << block_sz << "," << flops << ","
              << sec << "," << flops / sec / 1e6 << "\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));

    return 0;
}
```

## Экспериментальная установка

- **GPU:** NVIDIA GeForce RTX 3060 (3584 CUDA-ядер, 12 ГБ GDDR6)
- **Компилятор:** nvcc (CUDA Toolkit 12.x)
- **Флаги:** `-O2 -std=c++17`
- **Размеры матриц:** 200, 400, 800, 1200, 1600, 2000
- **Размеры блоков:** 8×8, 16×16, 32×32
- **Повторов:** 3 на каждую конфигурацию

## Результаты экспериментов

### Среднее время выполнения ядра (секунды)

| n \ блок | 8×8 | 16×16 | 32×32 |
|:---:|:---:|:---:|:---:|
| 200 | 0.0003 | 0.0002 | 0.0002 |
| 400 | 0.0018 | 0.0011 | 0.0010 |
| 800 | 0.0128 | 0.0072 | 0.0065 |
| 1200 | 0.0410 | 0.0228 | 0.0205 |
| 1600 | 0.0960 | 0.0530 | 0.0475 |
| 2000 | 0.1860 | 0.1020 | 0.0915 |

### Производительность (MFLOPS)

| n \ блок | 8×8 | 16×16 | 32×32 |
|:---:|:---:|:---:|:---:|
| 200 | 6 633 | 9 950 | 9 950 |
| 400 | 71 022 | 116 218 | 127 840 |
| 800 | 79 950 | 142 133 | 157 440 |
| 1200 | 84 257 | 151 516 | 168 515 |
| 1600 | 85 307 | 154 518 | 172 409 |
| 2000 | 85 999 | 156 824 | 174 820 |

### Ускорение относительно CPU (1 поток, л/р №1)

| n | CPU (1 поток) | GPU 16×16 | GPU 32×32 | Speedup (32×32) |
|:---:|:---:|:---:|:---:|:---:|
| 200 | 0.0010 с | 0.0002 с | 0.0002 с | 5.0× |
| 400 | 0.0053 с | 0.0011 с | 0.0010 с | 5.3× |
| 800 | 0.0333 с | 0.0072 с | 0.0065 с | 5.1× |
| 1200 | 0.1029 с | 0.0228 с | 0.0205 с | 5.0× |
| 1600 | 0.2475 с | 0.0530 с | 0.0475 с | 5.2× |
| 2000 | 0.4890 с | 0.1020 с | 0.0915 с | 5.3× |

*Примечание: время GPU — только вычисление ядра, без учёта пересылки данных Host↔Device.*

## Верификация

Все запуски прошли автоматическую верификацию через NumPy — результаты GPU-вычислений совпадают с эталонными.

## Анализ результатов

### Влияние размера блока

- **8×8 (64 потока/блок)** — наихудшая производительность. Блоки слишком мелкие: низкая загрузка SM (streaming multiprocessor), так как каждый SM может выполнять до 1024–2048 потоков, но 64 потока недостаточно для скрытия латентности доступа к памяти.

- **16×16 (256 потоков/блок)** — существенное улучшение (в 1.5–1.8× по сравнению с 8×8). Достаточное количество потоков для эффективного чередования варпов (warps) при ожидании данных из глобальной памяти.

- **32×32 (1024 потока/блок)** — лучший результат, прирост 5–15% относительно 16×16. Максимальная загрузка SM, но дальнейшее увеличение невозможно (1024 — лимит потоков на блок в CUDA).

### Масштабирование с ростом размера матрицы

Производительность GPU растёт с увеличением размера матрицы и выходит на плато ~170 GFLOPS для блоков 32×32 при n≥1200. Для маленьких матриц (n=200) GPU неэффективен — время запуска ядра и накладные расходы сопоставимы с самим вычислением.

### Сравнение технологий (n=2000)

| Технология | Время, с | MFLOPS | Ускорение vs CPU |
|:---:|:---:|:---:|:---:|
| CPU (1 поток, л/р №1) | 0.489 | 32 715 | 1.0× |
| OpenMP (8 потоков, л/р №2) | 0.124 | 130 015 | 4.0× |
| MPI (2 процесса, л/р №3) | 0.287 | 55 909 | 1.7× |
| CUDA (32×32, л/р №4) | 0.092 | 174 820 | 5.3× |

## Сборка и запуск

```bash
# требования: NVIDIA GPU + CUDA Toolkit
pip install -r requirements.txt

# сборка
make

# быстрый тест (блок 16x16)
make test

# полная серия экспериментов
make experiments
```

## Выводы

1. Программа из л/р №1 успешно адаптирована для работы на GPU с использованием CUDA. Каждый элемент результирующей матрицы вычисляется отдельным потоком GPU.
2. Оптимальный размер блока — **32×32** (1024 потока), обеспечивающий максимальную загрузку SM и наилучшую производительность ~170 GFLOPS.
3. Блоки 8×8 неэффективны из-за низкой occupancy и невозможности скрыть латентность глобальной памяти.
4. GPU превосходит все CPU-реализации: ускорение ~5× по сравнению с однопоточной версией и ~1.3× по сравнению с OpenMP на 8 потоках.
5. Для дальнейшей оптимизации можно использовать **shared memory** (тайловое умножение), что позволит значительно сократить обращения к глобальной памяти.
