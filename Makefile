OBJ = gemm.o
CXX = clang++
EXE = gemm
OPT = -O3 -framework Accelerate
CXXFLAGS = -std=c++11 -g $(OPT)
CFLAGS = -std=c11 -g $(OPT)
DEP = $(OBJ:.o=.d)

.PHONY: all clean

all: $(EXE)

$(EXE) : $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) $(LIBS) -o $(EXE)

%.o: %.cc
	$(CXX) -MMD $(CXXFLAGS) -c $< 

-include $(DEP)

clean:
	rm -f $(EXE) $(OBJ) $(DEP) *.csv *~
