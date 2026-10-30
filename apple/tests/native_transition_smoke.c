#include "toklet.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
_Static_assert(sizeof(TokletTransitionPlanningOptions) == 64, "transition ABI");
_Static_assert(offsetof(TokletTransitionPlanningOptions, expected_model_instance) == 8, "model offset");
_Static_assert(offsetof(TokletTransitionPlanningOptions, reserved) == 56, "reserved offset");
_Static_assert(sizeof(TokletOptions) == 24, "existing options ABI");

int main(int argc, char **argv) {
    assert(argc == 3); /* loadable CPU Qwen artifact or tiny MLX + backend 0/1 */
    assert(strcmp(argv[2], "0") == 0 || strcmp(argv[2], "1") == 0);
    TokletOptions options = {TOKLET_ABI_VERSION, sizeof(options), 32, 3, 128u * 1024u * 1024u};
    TokletModel *model = NULL; TokletSession *session = NULL; TokletError error = {0};
    uint32_t code = strcmp(argv[2], "0") == 0
        ? toklet_model_load((const uint8_t *)argv[1], strlen(argv[1]), &options, &model, &error)
        : toklet_model_load_mlx((const uint8_t *)argv[1], strlen(argv[1]), &options, &model, &error);
    assert(code == TOKLET_OK);
    assert(toklet_session_create(model, &options, &session, &error) == TOKLET_OK);
    TokletContextInfo context = {0};
    assert(toklet_session_context(session, &context, &error) == TOKLET_OK);
    TokletTransitionPlanningOptions transition = {
        TOKLET_ABI_VERSION, sizeof(transition), context.model_instance, context.session_instance,
        256u * 1024u * 1024u, 128, 4096, 8192, 0
    };
    TokletCompatibility *handle = NULL;
    assert(toklet_session_transition_plan_create(model, session, &options, &transition, &handle, &error) == TOKLET_OK);
    size_t count = 0;
    assert(toklet_mlx_inspect_json(handle, NULL, 0, &count, &error) == TOKLET_OK);
    assert(count > 0 && count <= TOKLET_MAX_COMPATIBILITY_JSON_BYTES);
    size_t length = count; uint8_t *bytes = malloc(length + 1); assert(bytes);
    memset(bytes, 0xa5, length + 1);
    assert(toklet_mlx_inspect_json(handle, bytes, length - 1, &count, &error) == TOKLET_INVALID);
    assert(count == length);
    for (size_t i = 0; i <= length; ++i) assert(bytes[i] == 0xa5);
    assert(toklet_mlx_inspect_json(handle, bytes, length, &count, &error) == TOKLET_OK);
    assert(count == length && bytes[length] == 0xa5); bytes[length] = 0;
    assert(strstr((char *)bytes, "\"report_kind\":\"same_model_session_transition\""));
    assert(strstr((char *)bytes, "\"fits_transition_budget\":true"));
    assert(strstr((char *)bytes, "\"candidate_allocated\":false"));
    assert(strstr((char *)bytes, "\"creation_revalidation_required\":true"));
    assert(strstr((char *)bytes, "\"reserves_verified\":false"));
    assert(toklet_mlx_inspect_release(handle, &error) == TOKLET_OK); free(bytes);
    transition.expected_session_instance++;
    handle = (TokletCompatibility *)(uintptr_t)1;
    assert(toklet_session_transition_plan_create(model, session, &options, &transition, &handle, &error) == TOKLET_INVALID);
    assert(handle == NULL);
    assert(toklet_session_release(session, &error) == TOKLET_OK);
    assert(toklet_model_release(model, &error) == TOKLET_OK);
    puts("{\"native_transition_c_abi\":true,\"passed\":true}");
    return 0;
}
