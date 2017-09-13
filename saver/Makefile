COMPILER = g++
FLAGS = -std=c++14 -O3 -Wall -Werror -Wextra

SO_DEPS = $(shell pkg-config --libs --cflags opencv libSimpleAmqpClient msgpack librabbitmq)
SO_DEPS += -lboost_program_options -lboost_system -lboost_filesystem -lpthread -larmadillo -I../include/

TARGET = saver

all: $(TARGET)

clean:
	rm -f $(TARGET)

$(TARGET): $(TARGET).cpp
	$(COMPILER) $^ -o $@ $(FLAGS) $(SO_DEPS)