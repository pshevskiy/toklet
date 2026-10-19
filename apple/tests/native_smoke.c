#include "toklet.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
_Static_assert(sizeof(TokletOptions)==24, "options ABI");
_Static_assert(sizeof(TokletTokenEvent)==16, "event ABI");
_Static_assert(sizeof(TokletReport)==40, "report ABI");
_Static_assert(sizeof(TokletError)==260, "error ABI");
struct Sink { TokletSession *session; uint32_t ids[4]; size_t count; };
static uint32_t emit(void *opaque, TokletTokenEvent event) {
    struct Sink *sink=opaque;
    assert(sink->count<4);
    sink->ids[sink->count++]=event.token_id;
    assert(toklet_session_reset(sink->session, NULL)==TOKLET_BUSY);
    return 0;
}
int main(int argc, char **argv) {
    assert(argc==2 || argc==3);
    assert(toklet_abi_version()==TOKLET_ABI_VERSION);
    TokletOptions o={TOKLET_ABI_VERSION,sizeof(o),32,3,128u*1024u*1024u};
    TokletModel *m=NULL;TokletSession *s=NULL;TokletCancel *c=NULL;TokletError error={0};
    uint32_t code=argc==3 ? toklet_model_load_metal((const uint8_t *)argv[1],strlen(argv[1]),NULL,0,&o,&m,&error) : toklet_model_load((const uint8_t *)argv[1],strlen(argv[1]),&o,&m,&error);
    if(code) { fprintf(stderr,"model load: %u %s\n",code,error.message);return 1; }
    assert(toklet_session_create(m,&o,&s,&error)==TOKLET_OK);
    assert(toklet_model_release(m,&error)==TOKLET_OK);
    assert(toklet_cancel_create(&c,&error)==TOKLET_OK);
    uint32_t prompt[]={1,7,13,3};TokletReport report={0};struct Sink sink={.session=s};
    assert(toklet_session_generate(s,0,prompt,4,2,NULL,0,c,emit,&sink,&report,&error)==TOKLET_OK);
    assert(report.history_tokens==6 && report.evaluated_tokens==5 && report.pending_tokens==1);
    assert(toklet_session_generate(s,2,NULL,0,2,NULL,0,c,emit,&sink,&report,&error)==TOKLET_OK);
    assert(report.history_tokens==8 && report.evaluated_tokens==7 && report.pending_tokens==1);
    const uint32_t expected[]={47,47,47,47};
    assert(sink.count==4 && memcmp(sink.ids,expected,sizeof(expected))==0);
    assert(toklet_cancel_set(c,1,&error)==TOKLET_OK);
    assert(toklet_session_generate(s,2,NULL,0,1,NULL,0,c,NULL,NULL,&report,&error)==TOKLET_OK);
    assert(report.reason==3 && report.history_tokens==8);
    assert(toklet_session_reset(s,&error)==TOKLET_OK);
    assert(toklet_session_state(s,&report,&error)==TOKLET_OK && report.history_tokens==0);
    assert(toklet_session_release(s,&error)==TOKLET_OK);
    assert(toklet_cancel_release(c,&error)==TOKLET_OK);
    printf("{\"ids\":[%u,%u,%u,%u],\"abi\":1,\"passed\":true}\n",sink.ids[0],sink.ids[1],sink.ids[2],sink.ids[3]);
    return 0;
}
