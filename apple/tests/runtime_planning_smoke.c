#include "toklet.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
_Static_assert(sizeof(TokletRuntimePlanningOptions)==56,"runtime options ABI");
_Static_assert(offsetof(TokletRuntimePlanningOptions,load_memory_limit_bytes)==40,"load budget offset");
_Static_assert(offsetof(TokletRuntimePlanningOptions,session_memory_limit_bytes)==48,"session budget offset");
int main(int argc,char **argv) {
    assert(argc==2);
    TokletRuntimePlanningOptions options={TOKLET_ABI_VERSION,sizeof(options),32,31,3,1,1,0,0,0,256u*1024u*1024u,256u*1024u*1024u};
    TokletCompatibility *handle=NULL;TokletError error={0};size_t count=0;
    assert(toklet_mlx_plan_create((const uint8_t *)argv[1],strlen(argv[1]),&options,&handle,&error)==TOKLET_OK);
    assert(handle!=NULL);
    assert(toklet_mlx_inspect_json(handle,NULL,0,&count,&error)==TOKLET_OK);
    assert(count>0 && count<=TOKLET_MAX_COMPATIBILITY_JSON_BYTES);
    size_t length=count;uint8_t *bytes=malloc(length+1);assert(bytes!=NULL);memset(bytes,0xa5,length+1);
    assert(toklet_mlx_inspect_json(handle,bytes,length-1,&count,&error)==TOKLET_INVALID);
    assert(count==length);
    for(size_t i=0;i<=length;i++)assert(bytes[i]==0xa5);
    assert(toklet_mlx_inspect_json(handle,bytes,length,&count,&error)==TOKLET_OK);
    assert(count==length && bytes[length]==0xa5);bytes[length]=0;
    assert(strstr((const char *)bytes,"\"fits_requested_budgets\":true")!=NULL);
    assert(strstr((const char *)bytes,"\"requested_position_offset\":1")!=NULL);
    assert(strstr((const char *)bytes,"\"requested_load_budget_bytes\":268435456")!=NULL);
    assert(strstr((const char *)bytes,"\"numerical_buffers_allocated\":false")!=NULL);
    assert(strstr((const char *)bytes,"\"qualification_evaluated\":false")!=NULL);
    assert(toklet_mlx_inspect_release(handle,&error)==TOKLET_OK);free(bytes);
    options.reserved=1;handle=(TokletCompatibility *)(uintptr_t)1;
    assert(toklet_mlx_plan_create((const uint8_t *)argv[1],strlen(argv[1]),&options,&handle,&error)==TOKLET_INVALID);
    assert(handle==NULL);
    puts("{\"runtime_planning_c_abi\":true,\"passed\":true}");return 0;
}
