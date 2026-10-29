#include "toklet.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
_Static_assert(sizeof(TokletNativeLimitsOptions)==16,"native limits ABI");
_Static_assert(offsetof(TokletNativeLimitsOptions,load_profile)==8,"profile offset");
_Static_assert(offsetof(TokletNativeLimitsOptions,reserved)==12,"reserved offset");
_Static_assert(sizeof(TokletCompatibilityOptions)==24,"existing compatibility ABI");
_Static_assert(sizeof(TokletRuntimePlanningOptions)==56,"existing planner ABI");
int main(int argc, char **argv) {
    assert(argc==2); /* portable tiny MLX Qwen3 tied fixture */
    TokletNativeLimitsOptions options={TOKLET_ABI_VERSION,sizeof(options),0,0};
    TokletCompatibility *handle=NULL;TokletError error={0};size_t count=0;
    assert(toklet_mlx_context_limits_create((const uint8_t *)argv[1],strlen(argv[1]),&options,&handle,&error)==TOKLET_OK);
    assert(toklet_mlx_inspect_json(handle,NULL,0,&count,&error)==TOKLET_OK);
    assert(count>0 && count<=TOKLET_MAX_COMPATIBILITY_JSON_BYTES);
    size_t size=count;uint8_t *bytes=malloc(size+1);assert(bytes);memset(bytes,0xa5,size+1);
    assert(toklet_mlx_inspect_json(handle,bytes,size-1,&count,&error)==TOKLET_INVALID);
    assert(count==size);for(size_t i=0;i<=size;i++)assert(bytes[i]==0xa5);
    assert(toklet_mlx_inspect_json(handle,bytes,size,&count,&error)==TOKLET_OK);
    assert(count==size && bytes[size]==0xa5);bytes[size]=0;
    assert(strstr((char *)bytes,"\"report_kind\":\"native_mlx_context_limits\""));
    assert(strstr((char *)bytes,"\"load_profile\":\"ordinary\""));
    assert(strstr((char *)bytes,"\"factory_position_upper\":32"));
    assert(strstr((char *)bytes,"\"budget_fit_evaluated\":false"));
    assert(strstr((char *)bytes,"\"qualification_evaluated\":false"));
    assert(strstr((char *)bytes,"\"payload_validation_required\":true"));
    assert(toklet_mlx_inspect_release(handle,&error)==TOKLET_OK);free(bytes);
    options.load_profile=2;handle=(TokletCompatibility *)(uintptr_t)1;
    assert(toklet_mlx_context_limits_create((const uint8_t *)argv[1],strlen(argv[1]),&options,&handle,&error)==TOKLET_INVALID);
    assert(handle==NULL);
    options.load_profile=0;options.reserved=1;handle=(TokletCompatibility *)(uintptr_t)1;
    assert(toklet_mlx_context_limits_create((const uint8_t *)argv[1],strlen(argv[1]),&options,&handle,&error)==TOKLET_INVALID);
    assert(handle==NULL);
    puts("{\"native_limits_c_abi\":true,\"passed\":true}");return 0;
}
