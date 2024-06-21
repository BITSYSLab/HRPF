INCLUDE := -I/usr/local/cuda-11/include/\
		-I/usr/include/c++/7\
		-I/opt/intel/mkl/include/\
		-I./

CXX = @echo compile $@ && /usr/bin/g++-9
CFLAGS = -fopenmp -std=c++14 -O0  -fPIC -g
LIBS = -L/usr/local/cuda-11/lib64 -lcudart -lcublas -Wl,--no-as-needed -ldl -lmkl_gnu_thread \
	-lmkl_intel_lp64 -liomp5 -m64 -lmkl_core -L/opt/intel/mkl/lib/intel64 -L/opt/intel/lib/intel64 \
	-lpthread -lm -L./algorithm/merge_sort


COMMON_PATH = common
FRAMEWORK_PATH = framework
DATASTRUCT_PATH = datastructture
TOOL_PATH = tool
#THREAD_POOL_PATH = threadpool
# STRASSEN_PROBLEM_PATH = algorithm/strassen_problem
# MERGE_PROBLEM_PATH = algorithm/merge_sort
#STRASSEN_NAIVE_PATH = algorithm/matrix_opera_problem
#MULT_PATH = algorithm/mult_problem
# TRSM_PATH = algorithm/trsm_problem
# PARALLEL_FOR_PATH = algorithm/parallel_for
#PARALLEL_REDUCE_PATH = algorithm/parallel_reduce
PARALLEL_FOR_ZB = algorithm/parallel_for_zero

SRC_H = $(wildcard $(COMMON_PATH)/*.h)
SRC_H += $(wildcard $(FRAMEWORK_PATH)/*.h)
SRC_H += $(wildcard $(DATASTRUCT_PATH)/*.h)
SRC_H += $(wildcard $(TOOL_PATH)/*.h)
#SRC_H += $(wildcard $(THREAD_POOL_PATH)/*.h)
# SRC_H += $(wildcard $(STRASSEN_PROBLEM_PATH)/*.h)
# SRC_H += $(wildcard $(MERGE_PROBLEM_PATH)/*.h)
#SRC_H += $(wildcard $(STRASSEN_NAIVE_PATH)/*.h)
#SRC_H += $(wildcard $(MULT_PATH)/*.h)
# SRC_H += $(wildcard $(TRSM_PATH)/*.h)
# SRC_H += $(wildcard $(PARALLEL_FOR_PATH)/*.h)
#SRC_H += $(wildcard $(PARALLEL_REDUCE_PATH)/*.cuh)
SRC_H += $(wildcard $(PARALLEL_FOR_ZB)/*.h)
INCLUDE += -I$(SRC_H)


SRC_COMMON = $(wildcard $(COMMON_PATH)/*.cpp)

SRC_FRAME = $(wildcard $(FRAMEWORK_PATH)/*.cpp)
SRC_DATA = $(wildcard $(DATASTRUCT_PATH)/*.cpp)
SRC_TOOL = $(wildcard $(TOOL_PATH)/*.cpp)
# SRC_STRASSEN = $(wildcard $(STRASSEN_PROBLEM_PATH)/*.cpp)
# SRC_MERGE = $(wildcard $(MERGE_PROBLEM_PATH)/*.cpp)
#SRC_MATRIX = $(wildcard $(STRASSEN_NAIVE_PATH)/*.cpp)
#SRC_MULT = $(wildcard $(MULT_PATH)/*.cpp)
# SRC_TRSM = $(wildcard $(TRSM_PATH)/*.cpp)
# SRC_PARALLEL_FOR = $(wildcard $(PARALLEL_FOR_PATH)/*.cpp)
#SRC_PARALLEL_REDUCE = $(wildcard $(PARALLEL_REDUCE_PATH)/*.cpp)
SRC_PARALLEL_FOR_ZB = $(wildcard $(PARALLEL_FOR_ZB)/*.cpp)

# OBJ = obj
# OBJ = merge_obj
#OBJ = parallel_for_obj
OBJ = parallel_for_zb_obj
#OBJ = parallel_reduce_obj
OBJ_COMMON = $(patsubst $(COMMON_PATH)/%.cpp, $(OBJ)/%.o, $(SRC_COMMON))
OBJ_FRAME = $(patsubst $(FRAMEWORK_PATH)/%.cpp, $(OBJ)/%.o, $(SRC_FRAME))
OBJ_DATA = $(patsubst $(DATASTRUCT_PATH)/%.cpp, $(OBJ)/%.o, $(SRC_DATA))
OBJ_TOOL = $(patsubst $(TOOL_PATH)/%.cpp, $(OBJ)/%.o, $(SRC_TOOL))
# OBJ_STRASSEN = $(patsubst $(STRASSEN_PROBLEM_PATH)/%.cpp, $(OBJ)/%.o, $(SRC_STRASSEN))
# OBJ_MERGE = $(patsubst $(MERGE_PROBLEM_PATH)/%.cpp, $(OBJ)/%.o, $(SRC_MERGE))
#OBJ_MATRIX = $(patsubst $(STRASSEN_NAIVE_PATH)/%.cpp, $(OBJ)/%.o, $(SRC_MATRIX))
#OBJ_MULT = $(patsubst $(MULT_PATH)/%.cpp, $(OBJ)/%.o, $(SRC_MULT))
# OBJ_TRSM = $(patsubst $(TRSM_PATH)/%.cpp, $(OBJ)/%.o, $(SRC_TRSM))
# OBJ_PARALLEL_FOR = $(patsubst $(PARALLEL_FOR_PATH)/%.cpp, $(OBJ)/%.o, $(SRC_PARALLEL_FOR))
#OBJ_PARALLEL_REDUCE = $(patsubst $(PARALLEL_REDUCE_PATH)/%.cpp, $(OBJ)/%.o, $(SRC_PARALLEL_REDUCE))
OBJ_PARALLEL_FOR_ZB = $(patsubst $(PARALLEL_FOR_ZB)/%.cpp, $(OBJ)/%.o, $(SRC_PARALLEL_FOR_ZB))

TARGET = smerge_harness #parallel_reduce_harness  #parallel_for_harness#trsm_harness #mult_harness #strassen_naive_harness#merge_harness #strassen_harness

all : OBJ_FILE #$(TARGET)

OBJ_FILE:
	#@echo $(OBJ_COMMON)
	#@echo $(OBJ_FRAME)
	$(CXX) -c $(CFLAGS) $(INCLUDE) $(COMMON_PATH)/cpu_device.cpp 	 -fPIC
	$(CXX) -c $(CFLAGS) $(INCLUDE) $(COMMON_PATH)/gpu_device.cpp -fPIC
	$(CXX) -c $(CFLAGS) $(INCLUDE) $(COMMON_PATH)/runtime.cpp -fPIC
	$(CXX) -c $(CFLAGS) $(INCLUDE) $(FRAMEWORK_PATH)/framework.cpp -fPIC
	$(CXX) -c $(CFLAGS) $(INCLUDE) $(FRAMEWORK_PATH)/problem.cpp -fPIC
	$(CXX) -c $(CFLAGS) $(INCLUDE) $(FRAMEWORK_PATH)/task.cpp -fPIC
	$(CXX) -c $(CFLAGS) $(INCLUDE) $(DATASTRUCT_PATH)/arraylist.cpp -fPIC
	$(CXX) -c $(CFLAGS) $(INCLUDE) $(DATASTRUCT_PATH)/matrix.cpp -fPIC
	# $(CXX) -c $(CFLAGS) $(INCLUDE) $(DATASTRUCT_PATH)/reduce_value.cpp -fPIC
	$(CXX) -c $(CFLAGS) $(INCLUDE) $(TOOL_PATH)/helper.cpp -fPIC
	# $(CXX) -c $(CFLAFS) $(INCLUDE) $(STRASSEN_PROBLEM_PATH)/strassenProblem.cpp
	# $(CXX) -c $(CFLAGS) $(INCLUDE) $(STRASSEN_PROBLEM_PATH)/strassen_harness.cpp
	#$(CXX) -c $(CFLAGS) $(INCLUDE) $(STRASSEN_NAIVE_PATH)/matrixProblem.cpp
	#$(CXX) -c $(CFLAGS) $(INCLUDE) $(STRASSEN_NAIVE_PATH)/strassenProblem.cpp
	#$(CXX) -c $(CFLAGS) $(INCLUDE) $(STRASSEN_NAIVE_PATH)/strassen_harness.cpp
	#$(CXX) -c $(CFLAGS) $(INCLUDE) $(STRASSEN_NAIVE_PATH)/test.cpp
	# $(CXX) -c $(CFLAGS) $(INCLUDE) $(MERGE_PROBLEM_PATH)/mergeSortProblem.cpp 
	# $(CXX) -c $(CFLAGS) $(INCLUDE) $(MERGE_PROBLEM_PATH)/merge_harness.cpp
	#$(CXX) -c $(CFLAGS) $(INCLUDE) $(MULT_PATH)/multProblem.cpp
	#$(CXX) -c $(CFLAGS) $(INCLUDE) $(MULT_PATH)/strassen_harness.cpp
	# $(CXX) -c $(CFLAGS) $(INCLUDE) $(TRSM_PATH)/multProblem.cpp
	# $(CXX) -c $(CFLAGS) $(INCLUDE) $(TRSM_PATH)/TrsmProblem.cpp
	# $(CXX) -c $(CFLAGS) $(INCLUDE) $(TRSM_PATH)/trsm_harness.cpp
	
	# $(CXX) -c $(CFLAFS) $(INCLUDE) $(PARALLEL_FOR_ZB)/parallel_for_zb.cpp -fPIC
	# $(CXX) -c $(CFLAFS) $(INCLUDE) $(PARALLEL_FOR_ZB)/parallel_for_inc.cpp -fPIC
	# $(CXX) -c $(CFLAFS) $(INCLUDE) $(PARALLEL_FOR_ZB)/parallel_for_dec.cpp -fPIC
	# $(CXX) -c $(CFLAFS) $(INCLUDE) $(PARALLEL_FOR_PATH)/parallel_for_problem.cpp -fPIC
	# $(CXX) -c $(CFLAFS) $(INCLUDE) $(PARALLEL_FOR_PATH)/parallelFor_2D.cpp -fPIC
	## $(CXX) -c $(CFLAFS) $(INCLUDE) $(PARALLEL_FOR_PATH)/test.cpp -fPIC
	## nvcc -c $(CFLAFS) $(INCLUDE) $(PARALLEL_FOR_PATH)/parallel_for_harness.cu -Xcompiler -fPIC
	#$(CXX) -c $(CFLAFS) $(INCLUDE) $(PARALLEL_REDUCE_PATH)/parallelReduce.cpp -fPIC
	# nvcc -c $(CFLAFS) $(INCLUDE) $(PARALLEL_REDUCE_PATH)/parallel_reduce_harness.cpp -Xcompiler -fPIC
	# nvcc -c $(CFLAFS) $(INCLUDE) $(PARALLEL_REDUCE_PATH)/parallelReduce.cu -Xcompiler -fPIC
	
	mv *.o $(OBJ)
	# $(CXX) -g  -o strassen_harness $(OBJ)/*.o $(LIBS) $(STRASSEN_PROBLEM_PATH)/cuAdd.so
	# $(CXX) -g -o merge_harness $(OBJ)/*.o $(LIBS) $(MERGE_PROBLEM_PATH)/cudaMerge.so
	#$(CXX) -g -o strassen_naive_harness $(OBJ)/*.o $(LIBS) $(STRASSEN_PROBLEM_PATH)/cuAdd.so
	#$(CXX) -o test $(OBJ)/*.o $(LIBS)
	#$(CXX) -g -o mult_harness $(OBJ)/*.o $(LIBS)
	# $(CXX) -g -o trsm_harness $(OBJ)/*.o $(LIBS)
	# nvcc -o parallelReduce.so -I. -std=c++14 -shared $(PARALLEL_REDUCE_PATH)/parallelReduce.cu -Xcompiler -fPIC
	$(CXX) -o framework_t.so  $(OBJ)/*.o $(LIBS)  -shared -fPIC 
	# nvcc -o  parallel_for_harness.so -shared  $(PARALLEL_FOR_PATH)/parallel_for_harness.cu -Xcompiler -fPIC
	
	## nvcc -c $(INCLUDE) $(PARALLEL_FOR_PATH)/test.cu 
	## nvcc -o parallel_for_harness $(INCLUDE) $(PARALLEL_FOR_PATH)/test.cu framework.so parallel_for_harness.so
	## nvcc -o test framework.so parallel_for_harness.so test.o
	# #nvcc -c $(CFLAFS) $(INCLUDE) $(PARALLEL_REDUCE_PATH)/parallel_reduce_harness.cpp -Xcompiler -fPIC
	#nvcc -o parallel_reduce_harness -I. $(PARALLEL_REDUCE_PATH)/parallel_reduce_harness.cu framework.so test_return.so #parallelReduce.so
#OBJ_SET = $(wildcard $(OBJ)/*.o)
#$(TARGET) : obj/cpu_device.o obj/gpu_device.o obj/runtime.o obj/framework.o obj/problem.o obj/arraylist.o obj/matrix.o obj/helper.o obj/strassenProblem.o obj/strassen_harness.o
#	@ $(CXX) $(CFLAGS) -o $(TARGET) $^ $(LIBS) $(INCLUDE)
#	@ echo Create $(TARGET) ok...

.PHONY:clean
clean:
	rm -rf $(TARGET) $(OBJ_COMMON) $(OBJ_FRAME) $(OBJ_MERGE) $(OBJ_DATA) $(OBJ_TOOL)
	@ echo clean all...
