#include "toklet.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
_Static_assert(TOKLET_MODEL_SESSION_BRIDGE_BYTES==20480u,"owner allowance");
_Static_assert(sizeof(TokletRuntimePlanningOptions)==56,"planning ABI");
int main(int argc,char **argv) {
    assert(argc==3);
    const uint64_t whole=256u*1024u*1024u,numerical=whole-TOKLET_MODEL_SESSION_BRIDGE_BYTES;
    TokletRuntimePlanningOptions plan={TOKLET_ABI_VERSION,sizeof(plan),32,32,3,0,1,0,2,0,numerical,numerical};
    TokletCompatibility *handle=NULL;TokletError error={0};size_t count=0;
    const uint8_t *path=(const uint8_t *)argv[1];size_t length=strlen(argv[1]);
    assert(toklet_mlx_plan_create(path,length,&plan,&handle,&error)==TOKLET_OK);
    assert(toklet_mlx_inspect_json(handle,NULL,0,&count,&error)==TOKLET_OK);
    assert(count>0 && count<=TOKLET_MAX_COMPATIBILITY_JSON_BYTES);
    char *json=calloc(count+1,1);assert(json!=NULL);
    assert(toklet_mlx_inspect_json(handle,(uint8_t *)json,count,&count,&error)==TOKLET_OK);
    char expected[64];int n=snprintf(expected,sizeof(expected),"\"attention\":\"%s\"",argv[2]);assert(n>0 && (size_t)n<sizeof(expected));
    assert(strstr(json,expected)!=NULL);
    const char *admission=strstr(json,"\"admitted_bytes\":");assert(admission!=NULL);
    uint64_t planned=strtoull(admission+strlen("\"admitted_bytes\":"),NULL,10);assert(planned>0);
    TokletOptions options={TOKLET_ABI_VERSION,sizeof(options),32,3,whole};
    TokletModel *model=NULL;TokletSession *session=NULL;TokletReport state={0};
    assert(toklet_model_load_mlx(path,length,&options,&model,&error)==TOKLET_OK);
    assert(toklet_session_create(model,&options,&session,&error)==TOKLET_OK);
    assert(toklet_session_state(session,&state,&error)==TOKLET_OK);
    assert(state.admitted_bytes==planned+TOKLET_MODEL_SESSION_BRIDGE_BYTES);
    assert(toklet_session_release(session,&error)==TOKLET_OK);
    assert(toklet_model_release(model,&error)==TOKLET_OK);
    assert(toklet_mlx_inspect_release(handle,&error)==TOKLET_OK);free(json);
    puts("{\"native_policy_c_abi\":true,\"plan_matches_factory\":true,\"passed\":true}");return 0;
}
