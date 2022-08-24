OPENCV_FLAGS=$(shell pkg-config --cflags --libs opencv4)
EIGEN_FLAGS=$(shell pkg-config --cflags eigen3)

.PHONY: all
all: stereo-reconstruct

stereo-reconstruct: main.cxx
	g++ -std=gnu++17 $^ $(OPENCV_FLAGS) $(EIGEN_FLAGS) -O3 -g -o $@
