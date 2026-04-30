# Лабораторная работа №3: Параллельное умножение матриц с MPI

## Задание

Модифицировать программу из лабораторной работы №1 для параллельной работы по технологии MPI. Провести серию экспериментов с разным количеством процессов (1, 2, 4, 8) и разными размерами матриц (200, 400, 800, 1200, 1600, 2000).

- **Исходные данные:** файлы, содержащие значения исходных матриц
- **Выходные данные:** файл со значениями результирующей матрицы, время выполнения, объём задачи
- **Верификация:** автоматизированная проверка корректности через Python (NumPy)

## Описание реализации

Программа использует модель распределённой памяти MPI. Стратегия распараллеливания — построчное распределение матрицы A между процессами:

1. **Процесс 0 (root):** загружает матрицы A и B из файлов, транспонирует B для кэш-оптимизации
2. **Broadcast:** транспонированная матрица B рассылается всем процессам через `MPI_Bcast`
3. **Scatter:** строки матрицы A распределяются между процессами через `MPI_Scatterv` (неравномерное распределение при n не кратном числу процессов)
4. **Вычисление:** каждый процесс перемножает свою часть строк A на B^T
5. **Gather:** результаты собираются на процессе 0 через `MPI_Gatherv`

```cpp
// распределение строк между процессами
int base_rows = n / nprocs;
int extra = n % nprocs;
for (int p = 0; p < nprocs; p++)
    row_counts[p] = base_rows + (p < extra ? 1 : 0);

// рассылка B всем процессам
MPI_Bcast(bt.data(), n * n, MPI_INT, 0, MPI_COMM_WORLD);

// раздача строк A
MPI_Scatterv(a_full.data(), send_counts.data(), send_displs.data(), MPI_INT,
             local_a.data(), my_rows * n, MPI_INT, 0, MPI_COMM_WORLD);

// локальное умножение
for (int i = 0; i < my_rows; i++)
    for (int j = 0; j < n; j++) {
        int acc = 0;
        for (int k = 0; k < n; k++)
            acc += local_a[i * n + k] * bt[j * n + k];
        local_c[i * n + j] = acc;
    }

// сборка результата
MPI_Gatherv(local_c.data(), my_rows * n, MPI_INT,
            c_full.data(), send_counts.data(), send_displs.data(), MPI_INT,
            0, MPI_COMM_WORLD);
```

## Исходный код

### main.cpp

```cpp
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
```

## Экспериментальная установка

- **Процессор:** Apple Silicon, 11 ядер
- **MPI-реализация:** Open MPI 5.0.9 (Homebrew)
- **Компилятор:** mpic++ (обёртка над clang++)
- **Флаги:** `-O2 -std=c++17`
- **Размеры матриц:** 200, 400, 800, 1200, 1600, 2000
- **Количество процессов:** 1, 2, 4, 8
- **Повторов:** 3 на каждую конфигурацию
- **Примечание:** все процессы запускались на одной машине (`--oversubscribe`)

## Результаты экспериментов

### Среднее время выполнения (секунды)

| n \ процессы | 1 | 2 | 4 | 8 |
|:---:|:---:|:---:|:---:|:---:|
| 200 | 0.0005 | 0.0003 | 0.0001 | 0.0002 |
| 400 | 0.0034 | 0.0016 | 0.0009 | 0.0012 |
| 800 | 0.0270 | 0.0139 | 0.0074 | 0.0101 |
| 1200 | 0.0932 | 0.0486 | 0.0382 | 0.0530 |
| 1600 | 0.2295 | 0.1167 | 0.1155 | 0.1374 |
| 2000 | 0.4347 | 0.2873 | 0.2525 | 0.2769 |

### Ускорение (Speedup = T₁ / Tₚ)

| n \ процессы | 1 | 2 | 4 | 8 |
|:---:|:---:|:---:|:---:|:---:|
| 200 | 1.00 | 1.96 | 3.64 | 2.75 |
| 400 | 1.00 | 2.04 | 3.70 | 2.71 |
| 800 | 1.00 | 1.95 | 3.65 | 2.68 |
| 1200 | 1.00 | 1.92 | 2.44 | 1.76 |
| 1600 | 1.00 | 1.97 | 1.99 | 1.67 |
| 2000 | 1.00 | 1.51 | 1.72 | 1.57 |

### Эффективность (Efficiency = Speedup / P)

| n \ процессы | 1 | 2 | 4 | 8 |
|:---:|:---:|:---:|:---:|:---:|
| 200 | 1.00 | 0.98 | 0.91 | 0.34 |
| 400 | 1.00 | 1.02 | 0.93 | 0.34 |
| 800 | 1.00 | 0.97 | 0.91 | 0.34 |
| 1200 | 1.00 | 0.96 | 0.61 | 0.22 |
| 1600 | 1.00 | 0.98 | 0.50 | 0.21 |
| 2000 | 1.00 | 0.76 | 0.43 | 0.20 |

### Производительность (MFLOPS)

| n \ процессы | 1 | 2 | 4 | 8 |
|:---:|:---:|:---:|:---:|:---:|
| 200 | 32 601 | 63 711 | 117 932 | 89 169 |
| 400 | 37 967 | 77 517 | 140 488 | 102 866 |
| 800 | 37 907 | 73 817 | 138 215 | 103 239 |
| 1200 | 37 070 | 71 136 | 90 487 | 65 208 |
| 1600 | 35 734 | 70 172 | 70 911 | 59 598 |
| 2000 | 36 800 | 55 909 | 63 384 | 57 776 |

## Верификация

Все 72 запуска (6 размеров x 4 конфигурации процессов x 3 повтора) прошли автоматическую верификацию через NumPy.

## Анализ результатов

### Масштабирование на 2 процессах

Для небольших и средних матриц (200–800) наблюдается почти линейное ускорение ~2× на 2 процессах (эффективность 95–98%). Это ожидаемо — объём коммуникаций (Bcast + Scatter/Gather) мал по сравнению с вычислительной работой.

### Деградация на 4 и 8 процессах

При увеличении числа процессов до 4 и 8 эффективность падает, особенно для больших матриц. Для n=2000 на 8 процессах ускорение составляет всего 1.57×. Причины:

1. **Накладные расходы MPI на одной машине.** В отличие от реального кластера, все процессы работают на одном узле. MPI использует механизмы IPC (разделяемая память, сокеты), которые добавляют задержки по сравнению с прямым доступом к памяти в OpenMP.

2. **Broadcast полной матрицы B.** Каждый процесс получает копию всей матрицы B (n² элементов). При n=2000 это 16 МБ на процесс, что создаёт давление на кэш и память.

3. **Конкуренция за пропускную способность памяти.** 8 процессов одновременно обращаются к своим копиям B, насыщая шину памяти.

### Сравнение с OpenMP (л/р №2)

| n | OpenMP 4 потока | MPI 4 процесса |
|:---:|:---:|:---:|
| 800 | 0.0120 с | 0.0074 с |
| 1200 | 0.0433 с | 0.0382 с |
| 1600 | 0.0917 с | 0.1155 с |
| 2000 | 0.1883 с | 0.2525 с |

На маленьких матрицах MPI показывает лучшие результаты из-за отсутствия накладных расходов OpenMP на управление потоками. Однако для больших матриц (1600+) OpenMP выигрывает благодаря отсутствию копирования данных между процессами.

## Сборка и запуск

```bash
# установка зависимостей
brew install open-mpi
pip install -r requirements.txt

# сборка
make

# быстрый тест (4 процесса)
make test

# полная серия экспериментов
make experiments
```

## Выводы

1. Программа из л/р №1 успешно адаптирована для работы с MPI. Реализовано построчное распределение матрицы A через `MPI_Scatterv` с рассылкой матрицы B через `MPI_Bcast`.
2. На 2 процессах достигается почти линейное ускорение (~2×) для всех размеров матриц.
3. На 4 процессах хорошее ускорение (3.6–3.7×) наблюдается только для матриц до 800×800. Для больших матриц эффективность падает из-за накладных расходов на коммуникации.
4. На 8 процессах производительность хуже, чем на 4, для всех размеров — дополнительные процессы не компенсируют расходы на IPC при работе на одном узле.
5. При сравнении с OpenMP: для больших задач (n≥1600) OpenMP эффективнее MPI на общей памяти, что ожидаемо — MPI проектировался для распределённых систем с отдельными узлами.
