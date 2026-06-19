#include <string.h>
#include "trig_registry_avx512.h"
int main(void){ trig_codelets_t r; memset(&r,0,sizeof r); trig_register_all_avx512(&r); return r.dct2[8]?0:0; }
