COMPILER = g++
FLAGS = -std=c++14 -O3 #-Wall -Werror -Wextra

SO_DEPS = $(shell pkg-config --libs --cflags libSimpleAmqpClient msgpack librabbitmq opencv)
SO_DEPS += -lboost_program_options -lpthread -larmadillo -I../include

all: bbs-viewer

clean:
	rm -f bbs-viewer
	
bbs-viewer: bbs-viewer.cpp
	$(COMPILER) $^ -o $@ $(FLAGS) $(SO_DEPS)