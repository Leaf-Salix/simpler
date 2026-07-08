ASCEND_HOME ?= /usr/local/Ascend/ascend-toolkit/latest
CXX := mpic++
CXXFLAGS ?= -std=c++17 -O2 -g -Wall -Wextra
ACL_LIB ?= acl_rt

INCLUDES := -I$(ASCEND_HOME)/include
LIBS := -L$(ASCEND_HOME)/lib64 -lhccl -l$(ACL_LIB) -pthread
TARGET := hccl_allgather_smoke

.PHONY: all clean run

all: $(TARGET)

$(TARGET): hccl_allgather_smoke.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGET)

run: $(TARGET) ranktable_a3_2host_32rank.json
	LD_LIBRARY_PATH=$(ASCEND_HOME)/lib64:$$LD_LIBRARY_PATH \
	mpirun --hostfile hostfile.mpi -np 32 ./$(TARGET) ranktable_a3_2host_32rank.json 16

ranktable_a3_2host_32rank.json:
	python3 gen_ranktable.py -o $@
