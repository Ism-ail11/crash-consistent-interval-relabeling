#include "har.h"

static int32_t magnitude(int32_t x)
{
    return x < 0 ? -x : x;
}

static int32_t nearest_div(int32_t x, int32_t n)
{
    return x < 0 ? -((-x + n / 2) / n) : (x + n / 2) / n;
}

void har_features(const int16_t *samples, int32_t *out)
{
    for (unsigned axis = 0; axis < HAR_CHANNELS; ++axis) {
        int32_t sum = 0;
        int32_t absolute = 0;
        int32_t difference = 0;
        int32_t previous = samples[axis];
        for (unsigned i = 0; i < HAR_WINDOW; ++i) {
            int32_t value = samples[i * HAR_CHANNELS + axis];
            sum += value;
            absolute += magnitude(value);
            difference += magnitude(value - previous);
            previous = value;
        }
        int32_t mean = nearest_div(sum, HAR_WINDOW);
        int32_t deviation = 0;
        for (unsigned i = 0; i < HAR_WINDOW; ++i) {
            deviation += magnitude(samples[i * HAR_CHANNELS + axis] - mean);
        }
        out[axis] = mean;
        out[6 + axis] = nearest_div(deviation, HAR_WINDOW);
        out[12 + axis] = nearest_div(difference, HAR_WINDOW - 1);
        out[18 + axis] = nearest_div(absolute, HAR_WINDOW);
    }
}

void har_embed(const int32_t *features, const HarMetric *metric, int16_t *out)
{
    for (unsigned i = 0; i < HAR_DIM; ++i) {
        /* The host fits centers in the input feature range and clamps scale to 4096. */
        int32_t delta = features[i] - metric->center[i];
        int32_t value = nearest_div(delta * metric->scale_q12[i], 4096);
        if (value > HAR_FEATURE_LIMIT) {
            value = HAR_FEATURE_LIMIT;
        }
        if (value < -HAR_FEATURE_LIMIT) {
            value = -HAR_FEATURE_LIMIT;
        }
        out[i] = (int16_t)value;
    }
}

void har_summarize(const HarVectors *vectors, HarBlock *out)
{
    out->count = vectors->count;
    out->label = vectors->label;
    out->reserved = 0;
    for (unsigned d = 0; d < HAR_DIM; ++d) {
        int32_t value = 0;
        for (unsigned v = 0; v < vectors->count; ++v) {
            value += vectors->vector[v][d];
        }
        out->sum[d] = value;
    }
}

void har_pack_block(const HarBlock *block, HarPacked *out)
{
    for (unsigned group = 0; group < HAR_DIM / 4; ++group) {
        const int32_t *s = block->sum + 4 * group;
        uint32_t a = (uint32_t)s[0] & 0x3FFFu;
        uint32_t b = (uint32_t)s[1] & 0x3FFFu;
        uint32_t c = (uint32_t)s[2] & 0x3FFFu;
        uint32_t d = (uint32_t)s[3] & 0x3FFFu;
        uint8_t *p = out->sum + 7 * group;
        p[0] = (uint8_t)a;
        p[1] = (uint8_t)((a >> 8) | (b << 6));
        p[2] = (uint8_t)(b >> 2);
        p[3] = (uint8_t)((b >> 10) | (c << 4));
        p[4] = (uint8_t)(c >> 4);
        p[5] = (uint8_t)((c >> 12) | (d << 2));
        p[6] = (uint8_t)(d >> 6);
    }
    out->count = (uint8_t)block->count;
    out->label = block->label;
}

static int32_t signed14(uint32_t word)
{
    return (int32_t)(word ^ 0x2000u) - 0x2000;
}

void har_unpack_block(const HarPacked *block, HarBlock *out)
{
    for (unsigned group = 0; group < HAR_DIM / 4; ++group) {
        const uint8_t *p = block->sum + 7 * group;
        int32_t *s = out->sum + 4 * group;
        s[0] = signed14(p[0] | ((p[1] & 63u) << 8));
        s[1] = signed14((p[1] >> 6) | ((uint32_t)p[2] << 2) | ((p[3] & 15u) << 10));
        s[2] = signed14((p[3] >> 4) | ((uint32_t)p[4] << 4) | ((p[5] & 3u) << 12));
        s[3] = signed14((p[5] >> 2) | ((uint32_t)p[6] << 6));
    }
    out->count = block->count;
    out->label = block->label;
    out->reserved = 0;
}

HarStatus har_enroll_begin(HarBlock *block, uint8_t label)
{
    if (label >= HAR_CLASSES) {
        return HAR_INVALID;
    }
    for (unsigned d = 0; d < HAR_DIM; ++d) {
        block->sum[d] = 0;
    }
    block->count = 0;
    block->label = label;
    block->reserved = 0;
    return HAR_OK;
}

HarStatus har_enroll_window(HarBlock *block, const int16_t *samples, const HarMetric *metric)
{
    if (block->count >= HAR_FEATURES_PER_BLOCK || block->label >= HAR_CLASSES) {
        return HAR_INVALID;
    }
    int32_t raw[HAR_DIM];
    int16_t embedded[HAR_DIM];
    har_features(samples, raw);
    har_embed(raw, metric, embedded);
    for (unsigned d = 0; d < HAR_DIM; ++d) {
        block->sum[d] += embedded[d];
    }
    ++block->count;
    return HAR_OK;
}

void har_centroids(const HarState *state, const int16_t *prior,
                   uint32_t prior_count, int16_t *out)
{
    for (unsigned c = 0; c < HAR_CLASSES; ++c) {
        for (unsigned d = 0; d < HAR_DIM; ++d) {
            int16_t anchor = prior[c * HAR_DIM + d];
            uint32_t count = state->count[c] + prior_count;
            out[c * HAR_DIM + d] = count == 0 ? anchor : (int16_t)nearest_div(
                state->sum[c][d] + anchor * (int32_t)prior_count, (int32_t)count);
        }
    }
}

uint8_t har_predict(const int16_t *embedding, const int16_t *centroids)
{
    uint8_t best = 0;
    uint32_t best_distance = UINT32_MAX;
    for (uint8_t c = 0; c < HAR_CLASSES; ++c) {
        uint32_t distance = 0;
        for (unsigned d = 0; d < HAR_DIM; ++d) {
            int32_t diff = embedding[d] - centroids[c * HAR_DIM + d];
            distance += (uint32_t)(diff * diff);
        }
        if (distance < best_distance) {
            best_distance = distance;
            best = c;
        }
    }
    return best;
}
