CXX ?= g++
CXXFLAGS += -O3 -std=c++14 -I../sdk -march=native -fopenmp
# CXXFLAGS += -O0 -std=c++14 -I../sdk -g
LDFLAGS +=  -shared -fPIC

SOURCE_DIR = src
OUTPUT_DIR = release

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	CXXFLAGS += --stdlib=libc++ -undefined dynamic_lookup
endif

build: libffm
.PHONY: build

clean:
	rm -rf $(OUTPUT_DIR)
.PHONY: clean

libffm: $(OUTPUT_DIR)/libffm.so

$(OUTPUT_DIR)/libffm.so: $(OUTPUT_DIR)/libffm.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

$(OUTPUT_DIR)/%.o: $(SOURCE_DIR)/%.cpp $(SOURCE_DIR)/%.hpp $(OUTPUT_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(OUTPUT_DIR):
	@mkdir -p $@

#### All targets ####
all: libffm
.PHONY: all
