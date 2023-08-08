

CXX:=${CXX}
CXX:=g++
# CPP_FLAGS=-std=c++17 -O3 -I/usr/local/include/eigen3 -Ibenchmark/include -march=native  -fno-lto
# CPP_FLAGS=-std=c++17 -I/usr/local/include/eigen3 -Ibenchmark/include -march=native  -O1 -fno-lto
CPP_FLAGS=-std=c++17 -I/usr/local/include/eigen3 -Ibenchmark/include -march=native  -O3 -fno-lto
LD_FLAGS=-Lbenchmark/build/src -l benchmark -lpthread

all: benchmark bm

out:
	mkdir out
out/ml.s: ml.cc out
	$(CXX) $(CPP_FLAGS) ml.cc -S -o out/ml.s -fverbose-asm
asm: out/ml.s

ml.o: ml.cc ml.h
	$(CXX) ml.cc -c -o $@ $(CPP_FLAGS) -fPIC

bm: bm.cc ml.o
	$(CXX) bm.cc -o bm $(CPP_FLAGS) $(LD_FLAGS) ml.o

clean:
	rm *.o
	rm bm

benchmark:
	git clone https://github.com/google/benchmark.git
	cd benchmark; mkdir build; cd build; \
		cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBENCHMARK_DOWNLOAD_DEPENDENCIES=on -DBENCHMARK_ENABLE_GTEST_TESTS=OFF -DBENCHMARK_ENABLE_TESTING=OFF; make -j4
