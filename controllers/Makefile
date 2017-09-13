COMPILER = g++
FLAGS = -std=c++14 -O3 -Wall -Werror -Wextra

SO_DEPS = $(shell pkg-config --libs --cflags opencv libSimpleAmqpClient msgpack librabbitmq)
SO_DEPS += -lboost_program_options -lboost_system -lboost_filesystem -lpthread -larmadillo -I../include/

TARGET = avoidance follower

all: $(TARGET)

clean:
	rm -f $(TARGET)

avoidance: avoidance.cpp
	$(COMPILER) $^ -o $@ $(FLAGS) $(SO_DEPS)

follower: follower.cpp
	$(COMPILER) $^ -o $@ $(FLAGS) $(SO_DEPS)