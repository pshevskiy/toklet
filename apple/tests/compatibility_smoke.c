#include "toklet.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
_Static_assert(TOKLET_COMPATIBILITY_JSON_ALLOWANCE_BYTES==2u*TOKLET_MAX_COMPATIBILITY_JSON_BYTES+256u, "JSON allowance");
_Static_assert(sizeof(TokletCompatibilityOptions)==24, "compatibility options ABI");
_Static_assert(offsetof(TokletCompatibilityOptions,memory_limit_bytes)==16, "budget ABI");
int main(int argc, char **argv) {
    assert(argc==2);
    TokletCompatibilityOptions options={TOKLET_ABI_VERSION,sizeof(options),0,32,128u*1024u*1024u};
    TokletCompatibility *handle=NULL;TokletError error={0};size_t count=0;
    assert(toklet_mlx_inspect_create((const uint8_t *)argv[1],strlen(argv[1]),&options,&handle,&error)==TOKLET_OK);
    assert(handle!=NULL);
    assert(toklet_mlx_inspect_json(handle,NULL,0,&count,&error)==TOKLET_OK);
    assert(count>0 && count<=TOKLET_MAX_COMPATIBILITY_JSON_BYTES);
    size_t length=count;
    uint8_t *bytes=malloc(length+1);assert(bytes!=NULL);memset(bytes,0xa5,length+1);
    assert(toklet_mlx_inspect_json(handle,bytes,length-1,&count,&error)==TOKLET_INVALID);
    assert(count==length);
    for(size_t i=0;i<=length;i++) assert(bytes[i]==0xa5);
    assert(toklet_mlx_inspect_json(handle,bytes,length,&count,&error)==TOKLET_OK);
    assert(count==length && bytes[length]==0xa5);bytes[length]=0;
    assert(strstr((const char *)bytes,"\"status\":\"compatible_untested\"")!=NULL);
    assert(strstr((const char *)bytes,"\"qualification_evaluated\":false")!=NULL);
    assert(strstr((const char *)bytes,"\"requested_memory_limit_bytes\":134217728")!=NULL);
    assert(toklet_mlx_inspect_release(handle,&error)==TOKLET_OK);free(bytes);
    options.backend=99;handle=(TokletCompatibility *)(uintptr_t)1;
    assert(toklet_mlx_inspect_create((const uint8_t *)argv[1],strlen(argv[1]),&options,&handle,&error)==TOKLET_INVALID);
    assert(handle==NULL);
    assert(toklet_mlx_inspect_release(NULL,&error)==TOKLET_OK);
    puts("{\"compatibility_c_abi\":true,\"passed\":true}");
    return 0;
}
