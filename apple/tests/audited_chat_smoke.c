#include "toklet.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
_Static_assert(sizeof(TokletOptions)==24,"options ABI");
_Static_assert(sizeof(TokletTokenEvent)==16,"event ABI");
_Static_assert(sizeof(TokletReport)==40,"report ABI");
_Static_assert(sizeof(TokletError)==260,"error ABI");
_Static_assert(sizeof(TokletScore)==48,"score ABI");
/* Unchanged independent app-chat-v1 Hello case, reference SHA 3b962a75... */
static const uint32_t expected_ids[]={1,9690,198,2683,359,253,5356,5646,11173,3365,3511,308,34519,28,7018,411,407,19712,8182,2,198,1,4093,198,19556,17,2,198,1,520,9531,198};
static const char expected_text[]="system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face\nuser\nHello!\nassistant\n";
static void statuses(const char *label,uint32_t code,size_t count,int unchanged) {
    printf("NATIVE_SMOL_C_STATUS {\"label\":\"%s\",\"code\":%u,\"count\":%zu,\"unchanged\":%s}\n",label,code,count,unchanged?"true":"false");
    assert(fflush(stdout)==0);
}
static void ids_buffer(const char *label,const uint32_t *values,size_t count,size_t capacity) {
    printf("NATIVE_SMOL_C_BUFFER {\"label\":\"%s\",\"reported_count\":%zu,\"capacity\":%zu,\"values\":[",label,count,capacity);
    size_t n=count<capacity?count:capacity;for(size_t i=0;i<n;i++)printf("%s%u",i?",":"",values[i]);puts("]}");assert(fflush(stdout)==0);
}
static void text_buffer(const char *label,const uint8_t *values,size_t count,size_t capacity) {
    printf("NATIVE_SMOL_C_BYTES {\"label\":\"%s\",\"reported_count\":%zu,\"capacity\":%zu,\"hex\":\"",label,count,capacity);
    size_t n=count<capacity?count:capacity;for(size_t i=0;i<n;i++)printf("%02x",values[i]);puts("\"}");assert(fflush(stdout)==0);
}
int main(int argc,char **argv) {
    assert(argc==2);
    assert(toklet_abi_version()==TOKLET_ABI_VERSION);
    /* Model policy +64MiB text+bounded diagnostic buffers are independently
       reserved within a1GiB Mac diagnostic policy before either factory. */
    const uint64_t model_policy=512u*1024u*1024u,text_policy=64u*1024u*1024u;
    TokletOptions options={TOKLET_ABI_VERSION,sizeof(options),8,3,model_policy};
    TokletModel *model=NULL;TokletChat *chat=NULL;TokletError error={0};
    uint32_t code=toklet_model_load_mlx(NULL,0,&options,&model,&error);
    statuses("legacy_invalid",code,0,model==NULL);assert(code==TOKLET_INVALID && model==NULL);
    code=toklet_model_load_mlx_smol_audited(NULL,0,&options,&model,&error);
    statuses("audited_invalid",code,0,model==NULL);assert(code==TOKLET_INVALID && model==NULL);
    code=toklet_chat_create_smol_audited(NULL,text_policy,&chat,&error);
    statuses("chat_null",code,0,chat==NULL);assert(code==TOKLET_INVALID && chat==NULL);
    code=toklet_model_load_mlx_smol_audited((const uint8_t *)argv[1],strlen(argv[1]),&options,&model,&error);
    statuses("load",code,0,model!=NULL);assert(code==TOKLET_OK && model!=NULL);
    code=toklet_chat_create_smol_audited(model,1,&chat,&error);
    statuses("chat_allowance",code,0,chat==NULL);assert(code==TOKLET_INVALID && chat==NULL);
    code=toklet_chat_create_smol_audited(model,text_policy,&chat,&error);
    statuses("chat_create",code,0,chat!=NULL);assert(code==TOKLET_OK && chat!=NULL);
    code=toklet_model_release(model,&error);model=NULL;
    statuses("external_release",code,0,1);assert(code==TOKLET_OK);
    const uint8_t input[]="[{\"role\":\"user\",\"content\":\"Hello!\"}]";
    size_t count=0;uint32_t ids[512];for(size_t i=0;i<512;i++)ids[i]=99;
    code=toklet_chat_encode(chat,input,sizeof(input)-1,NULL,0,&count,&error);
    statuses("encode_query",code,count,1);assert(code==TOKLET_OK && count==32);
    code=toklet_chat_encode(chat,input,sizeof(input)-1,ids,1,&count,&error);
    int unchanged=1;for(size_t i=0;i<512;i++)if(ids[i]!=99)unchanged=0;
    statuses("encode_short",code,count,unchanged);ids_buffer("encode_short",ids,count,512);assert(code==TOKLET_INVALID && count==32 && unchanged);
    code=toklet_chat_encode(chat,input,sizeof(input)-1,ids,512,&count,&error);
    statuses("encode",code,count,1);ids_buffer("encode",ids,count,512);
    assert(code==TOKLET_OK && count<=512);
    assert(count==32 && memcmp(ids,expected_ids,sizeof(expected_ids))==0);
    size_t bytes=0;code=toklet_chat_decode(chat,ids,count,NULL,0,&bytes,&error);
    statuses("decode_query",code,bytes,1);assert(code==TOKLET_OK && bytes==sizeof(expected_text)-1);
    uint8_t text[65536];memset(text,99,sizeof(text));
    code=toklet_chat_decode(chat,ids,count,text,1,&bytes,&error);
    unchanged=1;for(size_t i=0;i<sizeof(text);i++)if(text[i]!=99)unchanged=0;
    statuses("decode_short",code,bytes,unchanged);text_buffer("decode_short",text,bytes,sizeof(text));assert(code==TOKLET_INVALID && bytes==sizeof(expected_text)-1 && unchanged);
    code=toklet_chat_decode(chat,ids,count,text,sizeof(text),&bytes,&error);
    statuses("decode",code,bytes,1);text_buffer("decode",text,bytes,sizeof(text));
    assert(code==TOKLET_OK && bytes<=sizeof(text));
    assert(bytes==sizeof(expected_text)-1 && memcmp(text,expected_text,bytes)==0);
    uint32_t stops[8]={99};count=0;code=toklet_chat_stop_ids(chat,stops,8,&count,&error);
    statuses("stops",code,count,1);ids_buffer("stops",stops,count,8);assert(code==TOKLET_OK && count==2 && stops[0]==0 && stops[1]==2);
    code=toklet_chat_release(chat,&error);chat=NULL;
    statuses("chat_release",code,0,1);assert(code==TOKLET_OK);
    puts("NATIVE_SMOL_C_DONE {\"abi\":1,\"model_policy_bytes\":536870912,\"text_allowance_bytes\":67108864,\"process_policy_bytes\":1073741824,\"live_pairing\":true,\"new_numerical_forward\":false,\"app_capability_granted\":false,\"phone_fit_qualified\":false,\"passed\":true}");
    return fflush(stdout)==0?0:1;
}
