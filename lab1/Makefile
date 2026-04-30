CXX      = g++
CXXFLAGS = -O2 -std=c++17
TARGET   = matrix_mul
SRC      = src/main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

generate:
	python3 scripts/generate_matrix.py -n 500

run: $(TARGET)
	./$(TARGET) data/matrix_a.txt data/matrix_b.txt data/result.txt

verify:
	python3 scripts/verify.py data/matrix_a.txt data/matrix_b.txt data/result.txt

experiments: $(TARGET)
	python3 scripts/run_experiments.py

test: generate run verify

clean:
	rm -f $(TARGET) data/result.txt results/experiments.csv

.PHONY: all generate run verify experiments test clean
