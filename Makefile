default:
	nvcc main.cu -o main

run: default
	./main

prof: default
	nvprof ./main

mem: default
	cuda-memcheck ./main
