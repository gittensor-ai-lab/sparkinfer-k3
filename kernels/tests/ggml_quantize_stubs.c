// Stubs for symbols reached only by ggml's QUANTIZE paths. This test uses the DEQUANT
// path exclusively, so none of these is called; they exist to satisfy the linker
// without pulling in ggml.c. Each aborts if ever reached, so a silent fallback into a
// stub can't masquerade as a passing test.
#include <stdio.h>
#include <stdlib.h>
void ggml_abort(const char* f, int l, const char* fmt, ...) {
    fprintf(stderr, "ggml_abort at %s:%d\n", f, l); abort();
}
size_t ggml_row_size(int t, long long n) { (void)t;(void)n; abort(); }
const char* ggml_type_name(int t) { (void)t; abort(); }
size_t ggml_type_size(int t) { (void)t; abort(); }
