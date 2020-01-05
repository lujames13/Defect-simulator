XX = g++
MKLROOT ?= ${HOME}/opt/conda
LDFLAGS += -L${MKLROOT}/lib -lmkl_rt -lpthread -lm -ldl

PYBIND = $(shell python3 -m pybind11 --includes)
CXXFLAGS += -I${MKLROOT}/include \
	${PYBIND} \
	-shared -Wall \
	-std=c++17 -O3 -g -m64 -fPIC
SUFFIX = $(shell python3-config --extension-suffix)
TARGET = _matrix${SUFFIX}

all: $(TARGET)

$(TARGET): matrix.cpp
	${CXX} ${CXXFLAGS} $< -o $@ ${LDFLAGS}

test: test_matrix.py $(TARGET)
	python3 -m unittest test_matrix.py

bench: bench.py $(TARGET)
	python3 bench.py

.PHONY: clean
clean:
	rm -rf *.so __pycache__ performance.txt
