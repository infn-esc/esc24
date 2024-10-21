gif-h:
	git clone https://github.com/charlietangora/gif-h

serial: gif-h
	g++ circle_of_life.cpp -std=c++20 -o circle_of_life -O2 -I gif-h

clean:
	rm -rf circle_of_life circle_of_life_cuda

cuda:
	nvcc circle_of_life.cu -o circle_of_life_cuda -O2 -std=c++20 -I gif-h

all: serial cuda

run: all
	./circle_of_life
	./circle_of_life_cuda

