CXX := g++
CXXFLAGS := -O3 -std=c++11 -I ../sdk -shared -fPIC -march=native

DFLAG += -DUSEOMP
CXXFLAGS += -fopenmp


UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	CXXFLAGS += --stdlib=libc++ -undefined dynamic_lookup
endif

libffm : libffm.so 

libffm.so: src/libffm.cpp lib/ffm.o
	$(CXX) -o libffm.so $(CXXFLAGS) src/libffm.cpp lib/ffm.o 

ffm-train: lib/ffm-train.cpp lib/ffm.o
	$(CXX) $(CXXFLAGS) -o $@ $^

ffm-predict: lib/ffm-predict.cpp lib/ffm.o
	$(CXX) $(CXXFLAGS) -o $@ $^

lib/ffm.o: lib/ffm.cpp lib/ffm.h
	$(CXX) $(CXXFLAGS) $(DFLAG) -c -o $@ $<

clean:
	rm *.so

#### All targets ####
all: libffm ffm-train ffm-predict lib/ffm.o
