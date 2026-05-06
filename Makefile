CXX = g++
CXXFLAGS = -O3 -Wall -Wextra -I. -fopenmp -ffast-math -std=c++17

SRCS = $(wildcard *.cpp) $(wildcard *.c)

OBJS = $(SRCS:.cpp=.o)
OBJS := $(OBJS:.c=.o)
TARGET = pathtracer

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.c
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGET)
