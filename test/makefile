XX=g++ -O3 -Wall -shared -std=c++11 -fPIC
SRCS := $(shell find ./ -name "*.cpp")   # 查找src目录下所有的cpp文件
OBJS=$(SRCS:.cpp=.cpython-36m-x86_64-linux-gnu.so)
EXEC=test

include_paths := /usr/include/python3.6m \
/disk/223/person/huheng/tools/pybind11/include

include_paths := $(include_paths:%=-I%)



start:
	$(XX)   $(include_paths) -o $(OBJS)  $(SRCS)

clean:
	rm -rf $(OBJS)

#g++ -O3 -Wall -shared -std=c++11 -fPIC -I/usr/include/python3.8 `python3 -m pybind11 --includes` syszux.cpp -o syszux`python3-config --extension-suffix`