#include <stdint.h>

extern uint32_t _stack_top, _data_start, _data_end, _data_load, _bss_start, _bss_end;
void firmware_main(void);

static void Default_Handler(void)
{
    for (;;) {
        __asm__ volatile("wfi");
    }
}

void Reset_Handler(void)
{
    const uint32_t *source = &_data_load;
    for (uint32_t *target = &_data_start; target < &_data_end; ++target) {
        *target = *source++;
    }
    for (uint32_t *target = &_bss_start; target < &_bss_end; ++target) {
        *target = 0;
    }
    firmware_main();
    Default_Handler();
}

__attribute__((section(".isr_vector"), used))
const uintptr_t vectors[16] = {
    (uintptr_t)&_stack_top, (uintptr_t)Reset_Handler,
    (uintptr_t)Default_Handler, (uintptr_t)Default_Handler,
    (uintptr_t)Default_Handler, (uintptr_t)Default_Handler,
    (uintptr_t)Default_Handler, 0, 0, 0, 0, (uintptr_t)Default_Handler,
    (uintptr_t)Default_Handler, 0, (uintptr_t)Default_Handler, (uintptr_t)Default_Handler
};

/* Freestanding replacements for structure copies emitted by GCC. */
void *memcpy(void *dst, const void *src, unsigned length)
{
    uint8_t *out = dst;
    const uint8_t *in = src;
    for (unsigned i = 0; i < length; ++i) {
        out[i] = in[i];
    }
    return dst;
}

void *memset(void *dst, int value, unsigned length)
{
    uint8_t *out = dst;
    for (unsigned i = 0; i < length; ++i) {
        out[i] = (uint8_t)value;
    }
    return dst;
}
