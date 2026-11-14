#include "har.h"

#define ERASED UINT32_MAX
#define COMMITTED UINT32_C(0xA53CC35A)
#define EDIT_TAG UINT32_C(0x48410000)
#define CHECKPOINT_TAG UINT32_C(0x4841434B)
#define SNAPSHOT_TAG UINT32_C(0x4841534E)

_Static_assert(sizeof(HarBlock) == 100, "Summary format changed");
_Static_assert(sizeof(HarVectors) == 196, "Vector format changed");
_Static_assert(sizeof(HarPacked) == 44, "Packed format changed");
_Static_assert(HAR_DIM % 4 == 0 && HAR_PACKED_BYTES == HAR_DIM * 14 / 8,
               "Packing requires complete groups");
_Static_assert(HAR_FEATURES_PER_BLOCK * HAR_FEATURE_LIMIT <= 8191,
               "A block sum must fit a signed 14-bit coordinate");

static uint32_t data_crc(const uint32_t *data, unsigned size)
{
    /* CRC-32/ISO-HDLC over little-endian words. */
    uint32_t crc = UINT32_MAX;
    for (unsigned word = 0; word < size; ++word) {
        uint32_t value = data[word];
        for (unsigned bit = 0; bit < 32; ++bit) {
            uint32_t mix = (crc ^ value) & 1u;
            crc = (crc >> 1) ^ (UINT32_C(0xEDB88320) & (0u - mix));
            value >>= 1;
        }
    }
    return ~crc;
}

static uint32_t inverse_words(const uint32_t *data, unsigned size)
{
    /* A complemented population count detects any partial erase of protected bits:
       erase only increases the ones in data or in the count's complement. */
    uint32_t ones = 0;
    for (unsigned i = 0; i < size; ++i) {
        uint32_t value = data[i];
        for (; value != 0; value &= value - 1u) {
            ++ones;
        }
    }
    return ~ones;
}

static void encode(const HarView *view, HarEdit edit, uint32_t *words)
{
    words[0] = edit.id;
    words[1] = ((uint32_t)edit.end << 16) | edit.first;
    words[2] = EDIT_TAG | ((uint32_t)edit.to << 8) | edit.from;
    words[3] = view->epoch;
    words[4] = data_crc(words, 4);
    words[5] = COMMITTED;
}

static HarEdit decode(const uint32_t *words)
{
    HarEdit edit;
    edit.id = words[0];
    edit.first = (uint16_t)words[1];
    edit.end = (uint16_t)(words[1] >> 16);
    edit.from = (uint8_t)words[2];
    edit.to = (uint8_t)(words[2] >> 8);
    return edit;
}

static int is_blank(const uint32_t *words)
{
    uint32_t bits = ERASED;
    for (unsigned w = 0; w < HAR_RECORD_WORDS; ++w) {
        bits &= words[w];
    }
    return bits == ERASED;
}

static int is_committed(const HarView *view, const uint32_t *words)
{
    return words[5] == COMMITTED && words[3] == view->epoch &&
           (words[2] & UINT32_C(0xFFFF0000)) == EDIT_TAG &&
           ((view->flags & HAR_SKIP_CRC_VALIDATION) != 0 || words[4] == data_crc(words, 4));
}

uint32_t har_bank_words(uint32_t slots)
{
    return HAR_CHECKPOINT_WORDS + slots * HAR_RECORD_WORDS;
}

static uint32_t slot_word(const HarNv *nv, uint32_t bank, uint32_t slot)
{
    return bank * har_bank_words(nv->slots) + HAR_CHECKPOINT_WORDS + slot * HAR_RECORD_WORDS;
}

static void read_slot(const HarNv *nv, uint32_t bank, uint32_t slot, uint32_t *words)
{
    for (unsigned w = 0; w < HAR_RECORD_WORDS; ++w) {
        words[w] = nv->read(nv->context, slot_word(nv, bank, slot) + w);
    }
}

static int read_checkpoint(const HarView *view, const HarNv *nv, uint32_t bank,
                           uint32_t *words, unsigned *size)
{
    uint32_t address = bank * har_bank_words(nv->slots);
    words[0] = nv->read(nv->context, address);
    if (words[0] == CHECKPOINT_TAG) {
        *size = HAR_CHECKPOINT_WORDS;
    } else if (words[0] == SNAPSHOT_TAG) {
        *size = HAR_SNAPSHOT_WORDS;
    } else {
        return 0;
    }
    for (unsigned i = 1; i < *size; ++i) {
        words[i] = nv->read(nv->context, address + i);
    }
    return words[1] == view->epoch && words[2] != 0 && words[2] != ERASED &&
           words[3] != ERASED && words[*size - 1] == COMMITTED &&
           words[*size - 3] == data_crc(words, *size - 3) &&
           words[*size - 2] == inverse_words(words, *size - 2);
}

static void load_block(const HarView *view, uint16_t index, HarBlock *block)
{
    if (view->layout == HAR_SUMMARY) {
        *block = ((const HarBlock *)view->blocks)[index];
    } else if (view->layout == HAR_VECTORS) {
        har_summarize(&((const HarVectors *)view->blocks)[index], block);
    } else {
        har_unpack_block(&((const HarPacked *)view->blocks)[index], block);
    }
}

static void clear_state(HarState *state)
{
    for (unsigned c = 0; c < HAR_CLASSES; ++c) {
        state->count[c] = 0;
        for (unsigned d = 0; d < HAR_DIM; ++d) {
            state->sum[c][d] = 0;
        }
    }
    for (unsigned i = 0; i < HAR_MAX_BLOCKS; ++i) {
        state->label[i] = UINT8_MAX;
    }
    state->last_id = 0;
    state->next_slot = 0;
    state->generation = 0;
    state->bank = 0;
    state->last_edit_known = 0;
    state->last_edit = (HarEdit){0};
}

static int valid_range(const HarView *view, HarEdit edit)
{
    return edit.id != 0 && edit.id != ERASED && edit.from < HAR_CLASSES &&
           edit.to < HAR_CLASSES && edit.from != edit.to && edit.first >= view->edit_floor &&
           edit.first < edit.end && edit.end <= view->size;
}

static int valid_edit(const HarState *state, const HarView *view, HarEdit edit)
{
    if (!valid_range(view, edit)) {
        return 0;
    }
    for (uint16_t i = edit.first; i < edit.end; ++i) {
        if (state->label[i] != edit.from) {
            return 0;
        }
    }
    return 1;
}

static void apply_edit(HarState *state, const HarView *view, HarEdit edit)
{
    for (uint16_t i = edit.first; i < edit.end; ++i) {
        HarBlock block;
        load_block(view, i, &block);
        for (unsigned d = 0; d < HAR_DIM; ++d) {
            state->sum[edit.from][d] -= block.sum[d];
            state->sum[edit.to][d] += block.sum[d];
        }
        state->count[edit.from] -= block.count;
        state->count[edit.to] += block.count;
        state->label[i] = edit.to;
    }
    state->last_id = edit.id;
    state->last_edit = edit;
    state->last_edit_known = 1;
}

HarStatus har_open(HarState *state, const HarView *view, const HarNv *nv)
{
    if (state == NULL || view == NULL || nv == NULL || view->blocks == NULL ||
        nv->read == NULL || nv->program == NULL || nv->slots > HAR_MAX_SLOTS ||
        view->size > HAR_MAX_BLOCKS || view->edit_floor > view->size ||
        (view->flags & ~(HAR_SKIP_CRC_VALIDATION | HAR_DROP_CHECKPOINT_RECEIPT | HAR_SNAPSHOT_RECEIPT)) != 0 ||
        view->layout > HAR_PACKED) {
        return HAR_INVALID;
    }
    clear_state(state);
    uint32_t cp[2][HAR_CHECKPOINT_WORDS];
    unsigned size[2] = {0, 0};
    int bank0 = read_checkpoint(view, nv, 0, cp[0], &size[0]);
    int bank1 = read_checkpoint(view, nv, 1, cp[1], &size[1]);
    if (bank0 || bank1) {
        state->bank = (uint32_t)(bank1 && (!bank0 || cp[1][2] > cp[0][2]));
        state->generation = cp[state->bank][2];
        state->last_id = cp[state->bank][3];
        if (size[state->bank] == HAR_CHECKPOINT_WORDS && state->last_id != 0 &&
            cp[state->bank][HAR_SNAPSHOT_WORDS - 3] != ERASED) {
            const uint32_t *record = cp[state->bank] + HAR_SNAPSHOT_WORDS - 3;
            HarEdit edit = decode(record);
            if (!is_committed(view, record) || edit.id != state->last_id ||
                !valid_range(view, edit)) {
                return HAR_CORRUPT;
            }
            state->last_edit = edit;
            state->last_edit_known = 1;
        }
    }
    for (uint16_t i = 0; i < view->size; ++i) {
        HarBlock block;
        if (view->layout == HAR_VECTORS &&
            ((const HarVectors *)view->blocks)[i].count > HAR_FEATURES_PER_BLOCK) {
            return HAR_CORRUPT;
        }
        load_block(view, i, &block);
        if (block.count == 0 || block.count > HAR_FEATURES_PER_BLOCK || block.label >= HAR_CLASSES) {
            return HAR_CORRUPT;
        }
        uint8_t label = state->generation == 0 ? block.label :
            (uint8_t)(cp[state->bank][4 + i / 4] >> ((i % 4) * 8));
        if (label >= HAR_CLASSES) {
            return HAR_CORRUPT;
        }
        state->label[i] = label;
        state->count[label] += block.count;
        int32_t limit = (int32_t)block.count * HAR_FEATURE_LIMIT;
        for (unsigned d = 0; d < HAR_DIM; ++d) {
            if (block.sum[d] < -limit || block.sum[d] > limit) {
                return HAR_CORRUPT;
            }
            state->sum[label][d] += block.sum[d];
        }
    }
    for (uint32_t slot = 0; slot < nv->slots; ++slot) {
        uint32_t words[HAR_RECORD_WORDS];
        read_slot(nv, state->bank, slot, words);
        if (is_blank(words)) {
            continue;
        }
        state->next_slot = slot + 1;
        if (!is_committed(view, words)) {
            continue;
        }
        HarEdit edit = decode(words);
        if (edit.id <= state->last_id || !valid_edit(state, view, edit)) {
            return HAR_CORRUPT;
        }
        apply_edit(state, view, edit);
    }
    return HAR_OK;
}

static HarStatus compare_request(HarEdit prior, HarEdit edit)
{
    return prior.first == edit.first && prior.end == edit.end &&
           prior.from == edit.from && prior.to == edit.to ? HAR_DUPLICATE : HAR_CONFLICT;
}

static HarStatus previous_request(const HarState *state, const HarView *view,
                                  const HarNv *nv, HarEdit edit)
{
    if (edit.id == state->last_id && state->last_edit_known != 0) {
        return compare_request(state->last_edit, edit);
    }
    for (uint32_t slot = state->next_slot; slot > 0; --slot) {
        uint32_t words[HAR_RECORD_WORDS];
        uint32_t base = slot_word(nv, state->bank, slot - 1);
        if (nv->read(nv->context, base) != edit.id) {
            continue;
        }
        read_slot(nv, state->bank, slot - 1, words);
        if (is_committed(view, words)) {
            return compare_request(decode(words), edit);
        }
    }
    return HAR_STALE;
}

HarStatus har_edit(HarState *state, const HarView *view, const HarNv *nv, HarEdit edit)
{
    if (edit.id == 0 || edit.id == ERASED) {
        return HAR_INVALID;
    }
    if (edit.id <= state->last_id) {
        return previous_request(state, view, nv, edit);
    }
    if (!valid_edit(state, view, edit)) {
        return HAR_INVALID;
    }
    if (state->next_slot >= nv->slots) {
        return HAR_FULL;
    }
    uint32_t words[HAR_RECORD_WORDS];
    encode(view, edit, words);
    uint32_t base = slot_word(nv, state->bank, state->next_slot);
    for (unsigned w = 0; w < HAR_RECORD_WORDS; ++w) {
        if (nv->program(nv->context, base + w, words[w]) != 0) {
            /* Caller must reopen after a failed program, whose durability is unknown. */
            return HAR_IO;
        }
    }
    state->next_slot++;
    apply_edit(state, view, edit);
    return HAR_OK;
}

static HarStatus write_checkpoint(HarState *state, const HarView *view, const HarNv *nv,
                                  const HarEdit *snapshot_edit)
{
    if (nv->erase == NULL || state->generation >= ERASED - 1) {
        return HAR_INVALID;
    }
    uint32_t bank = 1u - state->bank;
    if (nv->erase(nv->context, bank) != 0) {
        return HAR_IO;
    }
    uint32_t words[HAR_CHECKPOINT_WORDS];
    int save_receipt = snapshot_edit == NULL ? (view->flags & HAR_DROP_CHECKPOINT_RECEIPT) == 0 :
                       (view->flags & HAR_SNAPSHOT_RECEIPT) != 0;
    unsigned size = save_receipt ? HAR_CHECKPOINT_WORDS : HAR_SNAPSHOT_WORDS;
    words[0] = save_receipt ? CHECKPOINT_TAG : SNAPSHOT_TAG;
    words[1] = view->epoch;
    words[2] = state->generation + 1;
    words[3] = snapshot_edit == NULL ? state->last_id : snapshot_edit->id;
    for (unsigned i = 0; i < HAR_MAX_BLOCKS / 4u; ++i) {
        uint32_t packed = 0;
        for (unsigned byte = 0; byte < 4; ++byte) {
            unsigned index = 4 * i + byte;
            uint8_t label = state->label[index];
            if (snapshot_edit != NULL && index >= snapshot_edit->first && index < snapshot_edit->end) {
                label = snapshot_edit->to;
            }
            packed |= (uint32_t)label << (8 * byte);
        }
        words[4 + i] = packed;
    }
    if (save_receipt) {
        uint32_t *record = words + HAR_SNAPSHOT_WORDS - 3;
        if (snapshot_edit != NULL) {
            encode(view, *snapshot_edit, record);
        } else if (state->last_edit_known != 0) {
            encode(view, state->last_edit, record);
        } else {
            for (unsigned i = 0; i < HAR_RECORD_WORDS; ++i) {
                record[i] = ERASED;
            }
        }
    }
    words[size - 3] = data_crc(words, size - 3);
    words[size - 2] = inverse_words(words, size - 2);
    words[size - 1] = COMMITTED;
    for (unsigned i = 0; i < size; ++i) {
        if (nv->program(nv->context, bank * har_bank_words(nv->slots) + i, words[i]) != 0) {
            return HAR_IO;
        }
    }
    state->bank = bank;
    state->generation++;
    state->next_slot = 0;
    if (snapshot_edit != NULL) {
        apply_edit(state, view, *snapshot_edit);
    }
    if (!save_receipt) {
        state->last_edit_known = 0;
    }
    return HAR_OK;
}

HarStatus har_compact(HarState *state, const HarView *view, const HarNv *nv)
{
    return write_checkpoint(state, view, nv, NULL);
}

HarStatus har_edit_snapshot(HarState *state, const HarView *view, const HarNv *nv, HarEdit edit)
{
    if (edit.id == 0 || edit.id == ERASED) {
        return HAR_INVALID;
    }
    if (edit.id <= state->last_id) {
        return previous_request(state, view, nv, edit);
    }
    if (!valid_edit(state, view, edit)) {
        return HAR_INVALID;
    }
    return write_checkpoint(state, view, nv, &edit);
}
