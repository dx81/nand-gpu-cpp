default:
	nvcc main.cu -o main

run: default
	./main.exe

prof: default
	nvprof ./main.exe

mem: default
	cuda-memcheck ./main.exe
