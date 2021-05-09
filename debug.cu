unsigned long long convert(unsigned long long n, int b1, int b2) {
    long long r, b = 1, out = 0;
    while (n > 0) {
        r = n % b2;
        out += r * b;
        b *= b1;
        n /= b2;
    }
    return out;
}

void dump(int *a, int *b, int *out, int n, int i = 0) {
    printf(
        "%-4c %-10c %-10c %-10c %-16c %-16c %-16c\n",
        'I', 'A', 'B', 'O', 'A', 'B', 'O'
    );
    n += i;
    while (i < n) {
        printf(
            "%4i %10u %10u %10u %016lld %016lld %016lld\n",
            i, a[i], b[i], out[i],
            convert((unsigned short)a[i], 10, 2),
            convert((unsigned short)b[i], 10, 2),
            convert((unsigned short)out[i++], 10, 2)
        );
    }
}

void test(int *out, int n) {
    int a, b, o, e;
    for (int i = 0; i < n; i++) {
        generate_data(&a, &b, i);
        o = (unsigned int)out[i];
        e = ~(a & b);
        if (o != e) printf("ERR [%i] expected %u got %u\n", i, o, e);
    }
    std::cout << "Test finished!" << std::endl;
}

__global__ void vector_nand_1d(int *out, int *a, int *b, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = index; i < n; i += stride) {
        out[i] = ~(a[i] & b[i]);
    }
}
