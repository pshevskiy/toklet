#include "toklet.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
_Static_assert(sizeof(TokletContextInfo)==136,"context ABI");
_Static_assert(offsetof(TokletContextInfo,model_instance)==32,"instance alignment");
_Static_assert(sizeof(TokletRequestAdmission)==192,"preflight ABI");
_Static_assert(offsetof(TokletRequestAdmission,base_history_tokens)==144,"request alignment");
int main(void) {
    TokletContextInfo context={.session_capacity=777},old=context;
    TokletRequestAdmission admission={.required_capacity=777},previous=admission;
    TokletError error={0};
    assert(toklet_session_context(NULL,&context,&error)==TOKLET_INVALID);
    assert(memcmp(&context,&old,sizeof(context))==0);
    assert(toklet_session_preflight(NULL,0,NULL,0,1,NULL,0,&admission,&error)==TOKLET_INVALID);
    assert(memcmp(&admission,&previous,sizeof(admission))==0);
    printf("{\"passed\":true,\"context_bytes\":%zu,\"admission_bytes\":%zu}\n",sizeof(context),sizeof(admission));
    return 0;
}
