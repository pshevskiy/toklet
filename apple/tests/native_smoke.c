#include "toklet.h"
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
_Static_assert(sizeof(TokletOptions)==24, "options ABI");
_Static_assert(sizeof(TokletTokenEvent)==16, "event ABI");
_Static_assert(sizeof(TokletReport)==40, "report ABI");
_Static_assert(sizeof(TokletError)==260, "error ABI");
_Static_assert(sizeof(TokletScore)==48, "score ABI");
_Static_assert(sizeof(TokletGenerationMetrics)==96, "metrics ABI");
_Static_assert(_Alignof(TokletGenerationMetrics)==8, "metrics alignment");
_Static_assert(sizeof(TokletGenerationMetricsV2)==104, "metrics V2 ABI");
_Static_assert(_Alignof(TokletGenerationMetricsV2)==8, "metrics V2 alignment");
_Static_assert(offsetof(TokletGenerationMetricsV2,base)==0, "metrics V2 base");
_Static_assert(offsetof(TokletGenerationMetricsV2,decode_evaluations)==96, "metrics V2 rows");
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
    int mlx=argc==3 && strcmp(argv[2],"mlx")==0;
    uint32_t code=mlx ? toklet_model_load_mlx((const uint8_t *)argv[1],strlen(argv[1]),&o,&m,&error) : argc==3 ? toklet_model_load_metal((const uint8_t *)argv[1],strlen(argv[1]),NULL,0,&o,&m,&error) : toklet_model_load((const uint8_t *)argv[1],strlen(argv[1]),&o,&m,&error);
    if(code) { fprintf(stderr,"model load: %u %s\n",code,error.message);return 1; }
    assert(toklet_session_create(m,&o,&s,&error)==TOKLET_OK);
    assert(toklet_model_release(m,&error)==TOKLET_OK);
    assert(toklet_cancel_create(&c,&error)==TOKLET_OK);
    uint32_t prompt[]={1,7,13,3};TokletReport report={0};struct Sink sink={.session=s};
    TokletScore score={0};
    assert(toklet_session_score(s,47,&score,&error)==TOKLET_INVALID);
    assert(toklet_session_generate(s,0,prompt,4,0,NULL,0,c,NULL,NULL,&report,&error)==TOKLET_OK);
    assert(report.history_tokens==4 && report.evaluated_tokens==4 && report.pending_tokens==0);
    assert(toklet_session_score(s,47,&score,&error)==TOKLET_OK);
    assert(score.abi_version==TOKLET_ABI_VERSION && score.struct_size==sizeof(score));
    assert(score.target_token==47 && score.greedy_token==47 && score.vocab_size==64 && score.evaluated_tokens==4);
    assert(isfinite(score.target_logit) && score.target_logit==score.greedy_logit);
    assert(isfinite(score.negative_log_likelihood) && score.negative_log_likelihood>0);
    /* Frozen independent tiny FP32 prompt-logit oracle; Metal's existing
       independent FP16 gates remain in its separately qualified fixture suite. */
    if(argc==2) assert(fabs(score.negative_log_likelihood-1.6877391136919981)<1e-5);
    assert(toklet_session_generate(s,0,prompt,4,2,NULL,0,c,emit,&sink,&report,&error)==TOKLET_OK);
    assert(report.history_tokens==6 && report.evaluated_tokens==5 && report.pending_tokens==1);
    assert(toklet_session_score(s,47,&score,&error)==TOKLET_INVALID);
    TokletGenerationMetrics metrics={0};
    assert(toklet_session_generate_observed(s,2,NULL,0,2,NULL,0,c,emit,&sink,&report,&metrics,&error)==TOKLET_OK);
    assert(metrics.abi_version==1 && metrics.struct_size==96 && metrics.reserved==0);
    assert(metrics.known_fields==(TOKLET_GENERATION_REPORT_KNOWN|TOKLET_GENERATION_FIRST_TOKEN_KNOWN));
    assert(metrics.prefill_tokens==1 && metrics.reused_tokens==5 && metrics.sampled_tokens==2 && metrics.emitted_tokens==2);
    assert(isfinite(metrics.total_seconds) && metrics.total_seconds>=metrics.first_token_seconds);
    assert(report.history_tokens==8 && report.evaluated_tokens==7 && report.pending_tokens==1);
    uint32_t history[32]={0};size_t count=0;
    assert(toklet_session_history(s,history,32,&count,&error)==TOKLET_OK);
    const uint32_t expected_history[]={1,7,13,3,47,47,47,47};
    assert(count==8 && memcmp(history,expected_history,sizeof(expected_history))==0);
    const uint32_t expected[]={47,47,47,47};
    assert(sink.count==4 && memcmp(sink.ids,expected,sizeof(expected))==0);
    assert(toklet_cancel_set(c,1,&error)==TOKLET_OK);
    assert(toklet_session_generate(s,2,NULL,0,1,NULL,0,c,NULL,NULL,&report,&error)==TOKLET_OK);
    assert(report.reason==3 && report.history_tokens==8);
    assert(toklet_cancel_set(c,0,&error)==TOKLET_OK);
    /* Legacy callers still own exactly 96 writable bytes. */
    struct { TokletGenerationMetrics value; uint64_t canary; } old_metrics={.canary=UINT64_MAX};
    assert(toklet_session_generate_observed(s,3,prompt,4,0,NULL,0,c,NULL,NULL,&report,&old_metrics.value,&error)==TOKLET_OK);
    assert(old_metrics.canary==UINT64_MAX && old_metrics.value.struct_size==96);
    struct { TokletGenerationMetricsV2 value; uint64_t canary; } v2={.canary=UINT64_MAX};
    struct Sink v2_sink={.session=s};
    assert(toklet_session_generate_observed_v2(s,3,prompt,4,2,NULL,0,c,emit,&v2_sink,&report,&v2.value,&error)==TOKLET_OK);
    assert(v2.canary==UINT64_MAX && v2.value.base.abi_version==1 && v2.value.base.struct_size==96);
    assert(v2.value.base.known_fields==(TOKLET_GENERATION_REPORT_KNOWN|TOKLET_GENERATION_FIRST_TOKEN_KNOWN));
    assert(v2.value.base.prefill_tokens==4 && v2.value.base.sampled_tokens==2 && v2.value.base.emitted_tokens==2 && v2.value.decode_evaluations==1);
    assert(report.history_tokens==6 && report.evaluated_tokens==5 && report.pending_tokens==1);
    assert(v2_sink.count==2 && v2_sink.ids[0]==47 && v2_sink.ids[1]==47);
    assert(toklet_session_generate_observed_v2(s,2,NULL,0,1,NULL,0,c,emit,&v2_sink,&report,&v2.value,&error)==TOKLET_OK);
    assert(v2.value.base.reused_tokens==5 && v2.value.base.prefill_tokens==1 && v2.value.decode_evaluations==0);
    assert(report.history_tokens==7 && report.evaluated_tokens==6 && report.pending_tokens==1 && v2_sink.count==3);
    uint32_t stop=47;
    assert(toklet_session_generate_observed_v2(s,3,prompt,4,2,&stop,1,c,emit,&v2_sink,&report,&v2.value,&error)==TOKLET_OK);
    assert(report.reason==2 && v2.value.base.sampled_tokens==1 && v2.value.base.emitted_tokens==0 && v2.value.decode_evaluations==0);
    assert(v2.value.base.known_fields==TOKLET_GENERATION_REPORT_KNOWN && v2_sink.count==3);
    assert(toklet_cancel_set(c,1,&error)==TOKLET_OK);
    assert(toklet_session_generate_observed_v2(s,2,NULL,0,1,NULL,0,c,NULL,NULL,&report,&v2.value,&error)==TOKLET_OK);
    assert(report.reason==3 && v2.value.base.known_fields==TOKLET_GENERATION_REPORT_KNOWN && v2.value.decode_evaluations==0);
    v2.value.decode_evaluations=UINT64_MAX;
    v2.value.base.known_fields=UINT32_MAX;
    assert(toklet_session_generate_observed_v2(s,99,NULL,0,1,NULL,0,c,NULL,NULL,&report,&v2.value,&error)==TOKLET_INVALID);
    assert(v2.value.base.known_fields==0 && v2.value.decode_evaluations==0 && v2.canary==UINT64_MAX);
    assert(toklet_session_reset(s,&error)==TOKLET_OK);
    assert(toklet_session_state(s,&report,&error)==TOKLET_OK && report.history_tokens==0);
    assert(toklet_session_release(s,&error)==TOKLET_OK);
    assert(toklet_cancel_release(c,&error)==TOKLET_OK);
    printf("{\"ids\":[%u,%u,%u,%u],\"abi\":1,\"passed\":true}\n",sink.ids[0],sink.ids[1],sink.ids[2],sink.ids[3]);
    return 0;
}
