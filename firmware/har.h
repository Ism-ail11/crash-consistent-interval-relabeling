#ifndef HAR_H
#define HAR_H

#include <stddef.h>
#include <stdint.h>

#define HAR_DIM 24
#define HAR_CLASSES 6
#define HAR_FEATURES_PER_BLOCK 4
#define HAR_WINDOW 100
#define HAR_CHANNELS 6
#define HAR_MAX_BLOCKS 128
#define HAR_RECORD_WORDS 6
#define HAR_MAX_SLOTS 128
#define HAR_FEATURE_LIMIT 2047
#define HAR_PACKED_BYTES 42u
#define HAR_SNAPSHOT_WORDS (7u + HAR_MAX_BLOCKS / 4u)
#define HAR_CHECKPOINT_WORDS (HAR_SNAPSHOT_WORDS + HAR_RECORD_WORDS)

enum { HAR_SUMMARY = 0, HAR_VECTORS = 1, HAR_PACKED = 2 };
enum { HAR_SKIP_CRC_VALIDATION = 1, HAR_DROP_CHECKPOINT_RECEIPT = 2, HAR_SNAPSHOT_RECEIPT = 4 };

typedef enum {
    HAR_OK = 0,
    HAR_DUPLICATE = 1,
    HAR_INVALID = -1,
    HAR_CONFLICT = -2,
    HAR_STALE = -3,
    HAR_FULL = -4,
    HAR_IO = -5,
    HAR_CORRUPT = -6
} HarStatus;

typedef struct {
    int32_t sum[HAR_DIM];
    uint16_t count;
    uint8_t label;
    uint8_t reserved;
} HarBlock;

typedef struct {
    int16_t vector[HAR_FEATURES_PER_BLOCK][HAR_DIM];
    uint16_t count;
    uint8_t label;
    uint8_t reserved;
} HarVectors;

typedef struct {
    uint8_t sum[HAR_PACKED_BYTES];
    uint8_t count;
    uint8_t label;
} HarPacked;

/* An immutable calibration generation. Change epoch when provisioning a new base. */
typedef struct {
    const void *blocks;
    uint32_t epoch;
    uint16_t size;
    uint16_t edit_floor;
    uint8_t layout;
    uint8_t flags;
} HarView;

typedef struct {
    uint32_t (*read)(void *context, uint32_t word);
    int (*program)(void *context, uint32_t word, uint32_t value);
    void *context;
    uint32_t slots;
    int (*erase)(void *context, uint32_t bank);
} HarNv;

/* Half-open, zero-based block range. A single edit moves one source class to one target. */
typedef struct {
    uint32_t id;
    uint16_t first;
    uint16_t end;
    uint8_t from;
    uint8_t to;
} HarEdit;

typedef struct {
    int32_t sum[HAR_CLASSES][HAR_DIM];
    uint32_t count[HAR_CLASSES];
    uint8_t label[HAR_MAX_BLOCKS];
    uint32_t last_id;
    uint32_t next_slot;
    uint32_t generation;
    uint32_t bank;
    uint32_t last_edit_known;
    HarEdit last_edit;
} HarState;

typedef struct {
    int32_t center[HAR_DIM];
    int32_t scale_q12[HAR_DIM];
} HarMetric;

/* Input samples are six signed Q8 channels (accelerometer in g, gyro in rad/s). */
void har_features(const int16_t *samples, int32_t *out);
void har_embed(const int32_t *features, const HarMetric *metric, int16_t *out);
void har_summarize(const HarVectors *vectors, HarBlock *out);
void har_pack_block(const HarBlock *block, HarPacked *out);
void har_unpack_block(const HarPacked *block, HarBlock *out);
HarStatus har_enroll_begin(HarBlock *block, uint8_t label);
HarStatus har_enroll_window(HarBlock *block, const int16_t *samples, const HarMetric *metric);
HarStatus har_open(HarState *state, const HarView *view, const HarNv *nv);
HarStatus har_edit(HarState *state, const HarView *view, const HarNv *nv, HarEdit edit);
HarStatus har_edit_snapshot(HarState *state, const HarView *view, const HarNv *nv, HarEdit edit);
HarStatus har_compact(HarState *state, const HarView *view, const HarNv *nv);
uint32_t har_bank_words(uint32_t slots);
void har_centroids(const HarState *state, const int16_t *prior,
                   uint32_t prior_count, int16_t *out);
uint8_t har_predict(const int16_t *embedding, const int16_t *centroids);

#endif
