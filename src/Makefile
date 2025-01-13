CC = gcc
CFLAGS = -fopenmp -I../libs -I../../cJSON/include
LDFLAGS = -L../../cJSON/build -lcjson -Wl,-rpath,../../cJSON/build
LD_LIBRARY_PATH = ../../cJSON/build

SRCS = main_app.c hll_ellpack_Tool.c hll_Operations.c functionsIO.c csrTool.c csrOperations.c
OBJS = $(patsubst %.c,../libs/%.o,$(SRCS))

TARGET = a.out

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

../libs/%.o: %.c
	$(CC) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

run: $(TARGET)
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) ./$(TARGET)