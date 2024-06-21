INCLUDE := -I/usr/local/cuda-11/include/\
		-I/usr/include/c++/9\
		-I/opt/intel/mkl/include/\
		-I./\
		-I ../common/\
		-I ../framework/\
		-I ../datastructture/\
		-I ../tool/\
		# -I../algorithm/parallel_for/\

CXX = @echo compile $@ && /usr/bin/g++
CC = nvcc
CFLAFS = -std=c++14
LDLIBS = -L../framework.so

# COMMON_PATH = ../common
# FRAMEWORK_PATH = ../framework
# DATASTRUCT_PATH = ../datastructture
# TOOL_PATH = ../tool
# PARALLEL_FOR_PATH = ../algorithm/parallel_for

# SRC_H = $(wildcard $(COMMON_PATH)/*.h)
# SRC_H += $(wildcard $(FRAMEWORK_PATH)/*.h)
# SRC_H += $(wildcard $(DATASTRUCT_PATH)/*.h)
# SRC_H += $(wildcard $(TOOL_PATH)/*.h)
# SRC_H += $(wildcard $(PARALLEL_FOR_PATH)/*.h)
# INCLUDE += -I$(SRC_H)

TARGET = parallel_for_harness

# loopfunc.o : loopfunc.cu
# 	$(CC) -c loopfunc.cu $(INCLUDE) 

parallel_for_harness : parallel_for_harness.cu
	$(CC) -o $(TARGET) parallel_for_harness.cu $(INCLUDE)  $(LDLIBS)  