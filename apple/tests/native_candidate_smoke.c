#include "toklet.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
_Static_assert(sizeof(TokletTransitionPlanningOptions) == 64, "existing transition ABI");
_Static_assert(sizeof(TokletOptions) == 24, "existing options ABI");
_Static_assert(TOKLET_CANCELLED == 10, "additive cancellation status");
int main(int argc, char **argv) {
    assert(argc == 3);
    assert(strcmp(argv[2], "0") == 0 || strcmp(argv[2], "1") == 0);
    TokletOptions options = {TOKLET_ABI_VERSION, sizeof(options), 32, 3, 128u * 1024u * 1024u};
    TokletModel *model = NULL; TokletSession *current = NULL, *candidate = NULL;
    TokletCancel *cancel = NULL; TokletError error = {0};
    uint32_t code = strcmp(argv[2], "0") == 0
        ? toklet_model_load((const uint8_t *)argv[1], strlen(argv[1]), &options, &model, &error)
        : toklet_model_load_mlx((const uint8_t *)argv[1], strlen(argv[1]), &options, &model, &error);
    assert(code == TOKLET_OK);
    assert(toklet_session_create(model, &options, &current, &error) == TOKLET_OK);
    assert(toklet_cancel_create(&cancel, &error) == TOKLET_OK);
    TokletContextInfo before = {0}, after = {0}, prepared = {0};
    assert(toklet_session_context(current, &before, &error) == TOKLET_OK);
    TokletTransitionPlanningOptions transition = {
        TOKLET_ABI_VERSION, sizeof(transition), before.model_instance, before.session_instance,
        256u * 1024u * 1024u, 0, 4096, 8192, 0
    };
    options.capacity = 16;
    assert(toklet_cancel_set(cancel, 1, &error) == TOKLET_OK);
    candidate = (TokletSession *)(uintptr_t)1;
    assert(toklet_session_prepare_empty(model, current, &options, &transition, cancel, &candidate, &error) == TOKLET_CANCELLED);
    assert(candidate == NULL);
    assert(toklet_cancel_set(cancel, 0, &error) == TOKLET_OK);
    assert(toklet_session_prepare_empty(model, current, &options, &transition, cancel, &candidate, &error) == TOKLET_OK);
    assert(candidate != NULL);
    assert(toklet_session_context(candidate, &prepared, &error) == TOKLET_OK);
    assert(prepared.model_instance == before.model_instance);
    assert(prepared.session_instance != before.session_instance);
    assert(prepared.session_capacity == 16);
    assert(prepared.history_tokens == 0 && prepared.committed_tokens == 0 && prepared.pending_tokens == 0);
    assert(toklet_session_context(current, &after, &error) == TOKLET_OK);
    assert(memcmp(&before, &after, sizeof(before)) == 0);
    assert(toklet_session_release(candidate, &error) == TOKLET_OK);
    transition.expected_session_instance++;
    candidate = (TokletSession *)(uintptr_t)1;
    assert(toklet_session_prepare_empty(model, current, &options, &transition, cancel, &candidate, &error) == TOKLET_INVALID);
    assert(candidate == NULL);
    assert(toklet_cancel_release(cancel, &error) == TOKLET_OK);
    assert(toklet_session_release(current, &error) == TOKLET_OK);
    assert(toklet_model_release(model, &error) == TOKLET_OK);
    puts("{\"native_empty_candidate_c_abi\":true,\"passed\":true}");
    return 0;
}
