.PHONY: all clean

CXX := g++

all: test

clean:
	rm -f test

test: test.cc Makefile
	$(CXX) -std=c++20 -O3 -g -Wall -march=native $< -ltbb -o $@
