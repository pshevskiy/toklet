#include "toklet.h"
#include <assert.h>
#include <stdint.h>
int main(void) {
    _Static_assert(sizeof(TokletOptions) == 24, "options ABI");
    _Static_assert(sizeof(TokletError) == 260, "error ABI");
    TokletModel *model = (TokletModel *)(uintptr_t)1;
    TokletError error = {0};
    assert(toklet_abi_version() == TOKLET_ABI_VERSION);
    assert(toklet_model_load_mlx_llama3_audited(NULL, 0, NULL, &model, &error) == TOKLET_INVALID);
    assert(model == NULL && error.code == TOKLET_INVALID);
    assert(toklet_model_release(model, &error) == TOKLET_OK);
    return 0;
}
