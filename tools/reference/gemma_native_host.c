/* Named Darwin host-provider witness; reference-only, never linked into Toklet.
 * Caller supplies aligned disjoint arrays of the declared lengths. Provider
 * headers are trusted, permanently loaded system images returned by dladdr.
 */
#include <dlfcn.h>
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <mach-o/loader.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#pragma STDC FP_CONTRACT OFF

int gemma_native_sincos(const double *angles, float *output, size_t count) {
    if (!angles || !output || count == 0 || count > 65536) return 1;
    if (fegetround() != FE_TONEAREST) return 2;
    volatile float tiny = FLT_TRUE_MIN, one = 1.0f;
    float preserved = tiny * one;
    uint32_t bits;
    memcpy(&bits, &preserved, 4);
    if (bits != 1) return 2;
    for (size_t i = 0; i < count; ++i)
        if (!isfinite(angles[i]) || fabs(angles[i]) > 512.) return 3;
    for (size_t i = 0; i < count; ++i) {
        double sine, cosine;
        __sincos(angles[i], &sine, &cosine);
        output[2*i] = (float)sine;
        output[2*i+1] = (float)cosine;
    }
    return 0;
}

int gemma_native_provider(size_t symbol, char *path, size_t capacity,
                          uint8_t *uuid, uint64_t *addresses) {
    const char *names[] = {"tanh", "exp", "pow", "sqrt", "__sincos_stret"};
    if (symbol >= 5 || !path || capacity == 0 || capacity > 8192 || !uuid || !addresses) return 1;
    void *address = dlsym(RTLD_DEFAULT, names[symbol]);
    Dl_info image;
    if (!address || !dladdr(address, &image) || !image.dli_fbase || !image.dli_fname) return 2;
    const struct mach_header_64 *header = image.dli_fbase;
    if (header->magic != MH_MAGIC_64 || header->ncmds > 1024 || header->sizeofcmds > 65536) return 3;
    size_t length = strlen(image.dli_fname);
    if (length >= capacity) return 4;
    const uint8_t *cursor = (const uint8_t *)(header+1);
    size_t remaining = header->sizeofcmds;
    for (uint32_t i = 0; i < header->ncmds; ++i) {
        if (remaining < sizeof(struct load_command)) return 3;
        const struct load_command *command = (const struct load_command *)cursor;
        if (command->cmdsize < sizeof(*command) || command->cmdsize > remaining) return 3;
        if (command->cmd == LC_UUID) {
            if (command->cmdsize != sizeof(struct uuid_command)) return 3;
            memcpy(uuid, ((const struct uuid_command *)command)->uuid, 16);
            memcpy(path, image.dli_fname, length+1);
            addresses[0] = (uintptr_t)address;
            addresses[1] = (uintptr_t)image.dli_fbase;
            return 0;
        }
        cursor += command->cmdsize;
        remaining -= command->cmdsize;
    }
    return 5;
}
