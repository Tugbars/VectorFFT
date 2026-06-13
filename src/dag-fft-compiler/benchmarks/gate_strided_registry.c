#include <string.h>
#include "strided_registry_avx512.h"
int main(void){ strided_codelets_t r; memset(&r,0,sizeof r); strided_register_all_avx512(&r); return r.n1_fwd[8]?0:0; }
