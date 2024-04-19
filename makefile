XX=g++ -std=c++11
SRCS := $(shell find src_ori -name "*.cpp")   # 查找src目录下所有的cpp文件
OBJS=$(SRCS:.cpp=.o)
EXEC=yolo_ort

include_paths := ./include \
/root/onnxruntime/include/onnxruntime/core/session \
/root/opencv-4.6.0/build/installed/include 
library_paths :=  /root/onnxruntime/build/Linux/Release 
ld_librarys   := onnxruntime


run_paths     := $(library_paths:%=-Wl,-rpath=%)
include_paths := $(include_paths:%=-I%)
library_paths := $(library_paths:%=-L%)
ld_librarys   := $(ld_librarys:%=-l%)

link_flags := $(library_paths) $(ld_librarys) $(run_paths)


start:$(OBJS)
	$(XX) -o $(EXEC) $(OBJS) `pkg-config --cflags --libs opencv4` $(link_flags)
.cpp.o:
	$(XX) -o $@ -c $< `pkg-config --cflags --libs opencv4` $(include_paths)

clean:
	rm -rf $(OBJS)
