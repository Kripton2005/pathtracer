CXX = g++
CXXFLAGS = -O3 -Wall -I.
# This finds all .cpp and .c files in your folder
SRCS = $(wildcard *.cpp) $(wildcard *.c)
# This turns the list of source files into a list of .o files
OBJS = $(SRCS:.cpp=.o)
OBJS := $(OBJS:.c=.o)
TARGET = main

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET)

# A generic rule: how to turn any .cpp into a .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# A generic rule: how to turn any .c into a .o
%.o: %.c
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGET)
