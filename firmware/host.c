#include "har.h"

typedef struct {
    uint32_t *words;
    uint32_t writes;
    int32_t cut;
    uint32_t mask;
    unsigned count;
    uint32_t bank_words;
} HostNv;

static uint32_t host_read(void *context, uint32_t word)
{
    HostNv *flash = context;
    return flash->words[word];
}

static int host_program(void *context, uint32_t word, uint32_t value)
{
    HostNv *flash = context;
    if (word >= flash->count || (flash->words[word] & value) != value) {
        return -1;
    }
    if ((int32_t)flash->writes == flash->cut) {
        flash->words[word] &= value | ~flash->mask;
        return -1;
    }
    flash->words[word] = value;
    flash->writes++;
    return 0;
}

static int host_erase(void *context, uint32_t bank)
{
    HostNv *flash = context;
    uint32_t base = bank * flash->bank_words;
    if ((int32_t)flash->writes == flash->cut) {
        for (unsigned i = 0; i < flash->bank_words; ++i) {
            flash->words[base + i] |= flash->mask;
        }
        return -1;
    }
    for (unsigned i = 0; i < flash->bank_words; ++i) {
        flash->words[base + i] = UINT32_MAX;
    }
    flash->writes++;
    return 0;
}

int host_open(HarState *state, const HarView *view, uint32_t *words, uint32_t slots)
{
    uint32_t span = har_bank_words(slots);
    HostNv flash = {words, 0, -1, 0, 2 * span, span};
    HarNv nv = {host_read, host_program, &flash, slots, host_erase};
    return har_open(state, view, &nv);
}

int host_edit(HarState *state, const HarView *view, uint32_t *words, uint32_t slots,
              HarEdit edit, int32_t cut, uint32_t mask, uint32_t *writes, int snapshot)
{
    uint32_t span = har_bank_words(slots);
    HostNv flash = {words, 0, cut, mask, 2 * span, span};
    HarNv nv = {host_read, host_program, &flash, slots, host_erase};
    HarStatus status = snapshot ? har_edit_snapshot(state, view, &nv, edit) :
                       har_edit(state, view, &nv, edit);
    *writes = flash.writes;
    return status;
}

int host_compact(HarState *state, const HarView *view, uint32_t *words, uint32_t slots,
                 int32_t cut, uint32_t mask, uint32_t *operations)
{
    uint32_t span = har_bank_words(slots);
    HostNv flash = {words, 0, cut, mask, 2 * span, span};
    HarNv nv = {host_read, host_program, &flash, slots, host_erase};
    HarStatus status = har_compact(state, view, &nv);
    *operations = flash.writes;
    return status;
}

void host_features(const int16_t *windows, uint32_t count, int32_t *out)
{
    for (uint32_t i = 0; i < count; ++i) {
        har_features(windows + i * HAR_WINDOW * HAR_CHANNELS, out + i * HAR_DIM);
    }
}

void host_predict(const int16_t *features, uint32_t count, const int16_t *centroids,
                  uint8_t *out)
{
    for (uint32_t i = 0; i < count; ++i) {
        out[i] = har_predict(features + i * HAR_DIM, centroids);
    }
}
