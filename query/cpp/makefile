# (make clean && make all) &> result

CXX = g++ -std=c++17 
# CXXFLAGS = -g -pthread 
CXXFLAGS = -O3 -DNDEBUG -pthread

INCLUDES = -I. -I./spatial -I./utils \
			-I/home/yanchuan/Documents/code/HighFive/install/include \
			-I/usr/include/hdf5/serial \
			-I/home/yanchuan/.conda/envs/pt3/include/python3.7m \
			-I/home/yanchuan/.conda/envs/pt3/lib/python3.7/site-packages/numpy/core/include

LIBS = -L/usr/local/lib -lroaring \
		-L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5 \
		-L/home/yanchuan/.conda/envs/pt3/lib -lpython3.7m \
		-L/home/yanchuan/.conda/envs/pt3/lib/python3.7/site-packages/numpy/core/lib -lnpymath

APPS = knnquery_exp

APPSSRC = $(patsubst %,%.cpp,$(APPS))

ROOTOBJS = DITA.o DFT.o
UTILSOBJS = ./utils/tool_funcs.o
SPATIALOBJS = $(patsubst %.cpp, %.o, $(wildcard ./spatial/*.cpp))
DEPEND_OBJS = $(ROOTOBJS) $(UTILSOBJS) $(SPATIALOBJS) 

all: $(DEPEND_OBJS)
	$(foreach APP, $(APPS), $(CXX) $(CXXFLAGS) $(INCLUDES) $(DEPEND_OBJS) $(LIBS) $(APP).cpp -o $(APP);)
	@echo "done"

test: $(SPATIALOBJS) $(UTILSOBJS) 
	$(CXX) test.cpp  $(CXXFLAGS) $(INCLUDES) $(UTILSOBJS) $(SPATIALOBJS) $(LIBS) -o test
	@echo "done"

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBS) -c $< -o $@

.PHONY: clean  

clean:  
	find . -name "*.o" -exec rm -rf {} \;

cleanall:  
	rm -f ${APPS} && find . -name "*.o" -exec rm -rf {} \;


# g++ -std=c++11 -g -I./btree -I./spatial -I./rtree -I./quadtree -I./bqtree -I./utils \
		./spatial/*.cpp ./rtree/*.cpp ./quadtree/*.cpp ./bqtree/*.cpp \
		./utils/utils.cpp ./utils/DP.cpp ./utils/cloaking.cpp \
		testbqtree_sc.cpp -o testbqtree_sc