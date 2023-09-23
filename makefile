# targets: dg, debug, mpi, mpi_debug, examples
.DEFAULT_GOAL := dg

CXX ?= g++
MPI_CXX ?= mpic++
FLAGS = -std=c++20 -Wall
EXTRN_LIBS ?= -llapack

SRC_DIR = source/
SRC = ${wildcard $(SRC_DIR)*.cpp}
OBJ = ${patsubst $(SRC_DIR)%.cpp, build/%.o, $(SRC)}
INCLUDE = -I./include/

EXAMPLES_DIR = examples/
EXAMPLES_SRC = ${wildcard $(EXAMPLES_DIR)*.cpp}
EXAMPLES_OUT = ${patsubst $(EXAMPLES_DIR)%.cpp, $(EXAMPLES_DIR)%, $(EXAMPLES_SRC)}

LIB = dg
LIBF = lib$(LIB).a

debug mpi_debug: FLAGS += -g -DDG_DEBUG
dg mpi: FLAGS += -O3

mpi mpi_debug: CXX = $(MPI_CXX)

dg debug mpi mpi_debug: $(OBJ)
	ar rcs $(LIBF) $(OBJ)

build/%.o: $(SRC_DIR)%.cpp
	@mkdir -p ${dir $@}
	$(CXX) $(FLAGS) -o $@ $< -c $(INCLUDE)

examples: $(EXAMPLES_OUT)

$(EXAMPLES_DIR)%: $(EXAMPLES_DIR)%.cpp $(LIBF)
	$(CXX) $(FLAGS) -g -o $@ $< $(INCLUDE) -L. -l$(LIB) $(EXTRN_LIBS)

clean:
	rm -rf build
	rm -f $(LIBF)
	rm -f $(EXAMPLES_OUT)