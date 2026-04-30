CXX      = clang++
CXXFLAGS = -O2 -std=c++17 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
LDFLAGS  = -L/opt/homebrew/opt/libomp/lib -lomp
TARGET   = matrix_mul_omp
SRC      = src/main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

generate:
	python3 scripts/generate_matrix.py -n 500

run: $(TARGET)
	./$(TARGET) data/matrix_a.txt data/matrix_b.txt data/result.txt 4

verify:
	python3 scripts/verify.py data/matrix_a.txt data/matrix_b.txt data/result.txt

experiments: $(TARGET)
	python3 scripts/run_experiments.py

test: generate run verify

clean:
	rm -f $(TARGET) data/result.txt results/experiments.csv

.PHONY: all generate run verify experiments test clean
