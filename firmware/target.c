/* STM32F4 experiment harness. Commands and test inputs are supplied by the host. */
#include "har.h"

#define BANK0 UINT32_C(0x08020000)
#define BANK1 UINT32_C(0x08040000)
#define BASE UINT32_C(0x08060000)
#define FLASH_KEYR (*(volatile uint32_t *)UINT32_C(0x40023C04))
#define FLASH_SR (*(volatile uint32_t *)UINT32_C(0x40023C0C))
#define FLASH_CR (*(volatile uint32_t *)UINT32_C(0x40023C10))
#define RCC_APB2ENR (*(volatile uint32_t *)UINT32_C(0x40023844))
#define USART_SR (*(volatile uint32_t *)UINT32_C(0x40011000))
#define USART_DR (*(volatile uint32_t *)UINT32_C(0x40011004))
#define USART_BRR (*(volatile uint32_t *)UINT32_C(0x40011008))
#define USART_CR1 (*(volatile uint32_t *)UINT32_C(0x4001100C))
#define AIRCR (*(volatile uint32_t *)UINT32_C(0xE000ED0C))
#define FLASH_BUSY (1u << 16)
#define FLASH_ERRORS UINT32_C(0xF2)
#define ACTIVE_SLOTS 16u
#define MAILBOX_MAGIC UINT32_C(0x48415234)
enum { PROVISION = 1, OPEN, EDIT, COMPACT, INFER, ENROLL, REBUILD, SNAPSHOT_EDIT, CACHE_BUILD };

typedef struct {
    uint32_t magic;
    uint32_t request;
    uint32_t response;
    uint32_t command;
    int32_t status;
    uint32_t layout;
    uint32_t epoch;
    uint32_t block_count;
    uint32_t edit_floor;
    uint32_t flags;
    uint32_t edit_id;
    uint32_t first;
    uint32_t end;
    uint32_t from;
    uint32_t to;
    int32_t cut;
    uint32_t mask;
    uint32_t operations;
    uint32_t program_bytes;
    uint32_t erase_count;
    uint32_t state_hash;
    uint32_t last_id;
    uint32_t generation;
    uint32_t bank;
    uint32_t next_slot;
    uint32_t prediction;
    uint32_t prior_count;
    uint32_t boot_count;
    uint32_t stage;
    uint32_t scratch_bytes;
} Mailbox;

typedef struct {
    int32_t cut;
    uint32_t mask;
    uint32_t operations;
    uint32_t program_bytes;
    uint32_t erase_count;
} FlashRun;

__attribute__((section(".mailbox"), used)) volatile Mailbox mailbox;
__attribute__((section(".mailbox"), used)) HarMetric input_metric;
__attribute__((section(".mailbox"), used)) int16_t input_prior[HAR_CLASSES][HAR_DIM];
__attribute__((section(".mailbox"), used)) int16_t input_samples[HAR_FEATURES_PER_BLOCK][HAR_WINDOW * HAR_CHANNELS];
__attribute__((section(".mailbox"), used)) uint32_t input_base[HAR_MAX_BLOCKS * sizeof(HarVectors) / 4];
__attribute__((section(".mailbox"), used)) HarBlock output_block;
/* The harness reports the active footprint, not this maximum staging capacity. */
__attribute__((section(".mailbox"), used)) HarBlock vector_cache[HAR_MAX_BLOCKS];

static HarState state;
static int opened;
static int cached_vectors;
static uint32_t provision_layout;
static uint32_t provision_epoch;
static uint32_t provision_count;
static uint32_t provision_floor;
static uint32_t provision_flags;

__attribute__((noinline, used)) void operation_begin(void)
{
    __asm__ volatile("nop" ::: "memory");
}

__attribute__((noinline, used)) void operation_end(void)
{
    __asm__ volatile("nop" ::: "memory");
}

static void flash_wait(void)
{
    while ((FLASH_SR & FLASH_BUSY) != 0) {
    }
}

static void flash_unlock(void)
{
    if ((FLASH_CR & (1u << 31)) != 0) {
        FLASH_KEYR = UINT32_C(0x45670123);
        FLASH_KEYR = UINT32_C(0xCDEF89AB);
    }
}

static int flash_write(uint32_t address, uint32_t value)
{
    volatile uint32_t *pointer = (volatile uint32_t *)(uintptr_t)address;
    if ((*pointer & value) != value) {
        return -1;
    }
    flash_wait();
    flash_unlock();
    FLASH_SR = FLASH_ERRORS | 1u;
    FLASH_CR = (2u << 8) | 1u; /* x32 programming */
    *pointer = value;
    flash_wait();
    FLASH_CR = (1u << 31);
    return (FLASH_SR & FLASH_ERRORS) == 0 && *pointer == value ? 0 : -1;
}

static int flash_erase(uint32_t sector, uint32_t address)
{
    flash_wait();
    flash_unlock();
    FLASH_SR = FLASH_ERRORS | 1u;
    uint32_t command = (2u << 8) | (sector << 3) | 2u;
    FLASH_CR = command | (1u << 16);
    flash_wait();
    FLASH_CR = 1u << 31;
    return (FLASH_SR & FLASH_ERRORS) == 0 &&
           *(volatile uint32_t *)(uintptr_t)address == UINT32_MAX ? 0 : -1;
}

static void reset_now(void)
{
    /* The control mailbox lives in simulated SRAM across SYSRESETREQ.
       Startup clears all firmware-owned BSS and reconstructs it on OPEN. */
    mailbox.response = mailbox.request;
    __asm__ volatile("dsb" ::: "memory");
    AIRCR = UINT32_C(0x05FA0004);
    __asm__ volatile("dsb" ::: "memory");
    for (;;) {
        __asm__ volatile("wfi");
    }
}

static uint32_t bank_address(uint32_t bank)
{
    return bank == 0 ? BANK0 : BANK1;
}

static uint32_t journal_address(uint32_t word)
{
    uint32_t span = har_bank_words(ACTIVE_SLOTS);
    return bank_address(word / span) + (word % span) * 4u;
}

static uint32_t read_word(void *context, uint32_t word)
{
    (void)context;
    return *(volatile uint32_t *)(uintptr_t)journal_address(word);
}

static int program_word(void *context, uint32_t word, uint32_t value)
{
    FlashRun *run = context;
    uint32_t address = journal_address(word);
    if ((int32_t)run->operations == run->cut) {
        uint32_t partial = value | ~run->mask;
        if (run->mask != 0 && flash_write(address, partial) != 0) {
            return -1;
        }
        reset_now();
    }
    if (flash_write(address, value) != 0) {
        return -1;
    }
    ++run->operations;
    run->program_bytes += 4;
    return 0;
}

static int erase_bank(void *context, uint32_t bank)
{
    FlashRun *run = context;
    if ((int32_t)run->operations == run->cut) {
        if (run->mask != 0 && flash_erase(5 + bank, bank_address(bank)) != 0) {
            return -1;
        }
        reset_now();
    }
    if (flash_erase(5 + bank, bank_address(bank)) != 0) {
        return -1;
    }
    ++run->operations;
    ++run->erase_count;
    return 0;
}

#define BASE_TAG UINT32_C(0x48414253)
#define BASE_HEADER_WORDS 7u
#define BASE_MARKER UINT32_C(0xD5D07A55)

static uint32_t hash_word(uint32_t hash, uint32_t word)
{
    return (hash ^ word) * UINT32_C(16777619);
}

static HarStatus load_base(void)
{
    const uint32_t *base = (const uint32_t *)(uintptr_t)BASE;
    if (base[0] != BASE_TAG || base[6] != BASE_MARKER ||
        base[1] > HAR_PACKED || base[3] == 0 || base[3] > HAR_MAX_BLOCKS ||
        base[4] > base[3] ||
        base[5] > (HAR_SKIP_CRC_VALIDATION | HAR_DROP_CHECKPOINT_RECEIPT | HAR_SNAPSHOT_RECEIPT)) {
        return HAR_CORRUPT;
    }
    provision_layout = base[1];
    provision_epoch = base[2];
    provision_count = base[3];
    provision_floor = base[4];
    provision_flags = base[5];
    return HAR_OK;
}

/* Provisioning establishes an immutable base and erases both journal banks. */
static HarStatus provision(FlashRun *run)
{
    uint32_t layout = mailbox.layout;
    uint32_t count = mailbox.block_count;
    if (layout > HAR_PACKED || count == 0 || count > HAR_MAX_BLOCKS ||
        mailbox.edit_floor > count ||
        mailbox.flags > (HAR_SKIP_CRC_VALIDATION | HAR_DROP_CHECKPOINT_RECEIPT | HAR_SNAPSHOT_RECEIPT)) {
        return HAR_INVALID;
    }
    opened = 0;
    if (erase_bank(run, 0) != 0 || erase_bank(run, 1) != 0 || flash_erase(7, BASE) != 0) {
        return HAR_IO;
    }
    ++run->erase_count;
    const uint32_t address = BASE;
    const uint32_t header[6] = {BASE_TAG, layout, mailbox.epoch, count, mailbox.edit_floor, mailbox.flags};
    const uint32_t bytes[] = {sizeof(HarBlock), sizeof(HarVectors), sizeof(HarPacked)};
    uint32_t words = count * bytes[layout] / 4u;
    for (unsigned i = 0; i < 6; ++i) {
        if (flash_write(address + 4 * i, header[i]) != 0) {
            return HAR_IO;
        }
    }
    for (uint32_t i = 0; i < words; ++i) {
        if (flash_write(address + 4 * (BASE_HEADER_WORDS + i), input_base[i]) != 0) {
            return HAR_IO;
        }
    }
    if (flash_write(address + 24, BASE_MARKER) != 0) {
        return HAR_IO;
    }
    run->program_bytes += 4 * (BASE_HEADER_WORDS + words);
    return load_base();
}

static HarView get_view(void)
{
    const void *blocks = (const void *)(uintptr_t)(BASE + 4 * BASE_HEADER_WORDS);
    HarView view = {
        cached_vectors ? (const void *)vector_cache : blocks,
        provision_epoch, (uint16_t)provision_count, (uint16_t)provision_floor,
        (uint8_t)(cached_vectors ? HAR_SUMMARY : provision_layout), (uint8_t)provision_flags
    };
    return view;
}

static HarStatus cache_vectors(void)
{
    const HarVectors *vectors = (const HarVectors *)(uintptr_t)(BASE + 4 * BASE_HEADER_WORDS);
    for (unsigned i = 0; i < provision_count; ++i) {
        if (vectors[i].count == 0 || vectors[i].count > HAR_FEATURES_PER_BLOCK) {
            return HAR_CORRUPT;
        }
        har_summarize(&vectors[i], &vector_cache[i]);
    }
    cached_vectors = 1;
    mailbox.scratch_bytes = provision_count * sizeof(HarBlock);
    return HAR_OK;
}

static uint32_t model_hash(void)
{
    uint32_t hash = UINT32_C(2166136261);
    for (unsigned c = 0; c < HAR_CLASSES; ++c) {
        hash = hash_word(hash, state.count[c]);
        for (unsigned d = 0; d < HAR_DIM; ++d) {
            hash = hash_word(hash, (uint32_t)state.sum[c][d]);
        }
    }
    for (unsigned i = 0; i < provision_count; ++i) {
        hash = hash_word(hash, state.label[i]);
    }
    return hash;
}

static HarStatus command(uint32_t code, FlashRun *run)
{
    HarNv nv = {read_word, program_word, run, ACTIVE_SLOTS, erase_bank};
    if (code == PROVISION) {
        cached_vectors = 0;
        mailbox.scratch_bytes = 0;
        return provision(run);
    }
    if (code == ENROLL) {
        uint32_t stage = mailbox.stage;
        if (mailbox.from >= HAR_CLASSES ||
            (stage != UINT32_MAX && (stage >= HAR_MAX_BLOCKS ||
             (mailbox.layout != HAR_SUMMARY && mailbox.layout != HAR_PACKED)))) {
            return HAR_INVALID;
        }
        HarStatus status = har_enroll_begin(&output_block, (uint8_t)mailbox.from);
        for (unsigned i = 0; i < HAR_FEATURES_PER_BLOCK && status == HAR_OK; ++i) {
            status = har_enroll_window(&output_block, input_samples[i], &input_metric);
        }
        if (status == HAR_OK && stage != UINT32_MAX) {
            const unsigned char *source = (const unsigned char *)&output_block;
            HarPacked compact;
            unsigned size = sizeof(HarBlock);
            if (mailbox.layout == HAR_PACKED) {
                har_pack_block(&output_block, &compact);
                source = (const unsigned char *)&compact;
                size = sizeof(HarPacked);
            }
            unsigned char *target = (unsigned char *)input_base + stage * size;
            for (unsigned i = 0; i < size; ++i) {
                target[i] = source[i];
            }
        }
        return status;
    }
    if (code == OPEN || code == CACHE_BUILD) {
        cached_vectors = 0;
        mailbox.scratch_bytes = 0;
        HarStatus status = load_base();
        if (status != HAR_OK) {
            opened = 0;
            return status;
        }
        if (code == CACHE_BUILD) {
            status = provision_layout == HAR_VECTORS ? cache_vectors() : HAR_INVALID;
            if (status != HAR_OK) {
                opened = 0;
                return status;
            }
        }
        HarView view = get_view();
        status = har_open(&state, &view, &nv);
        opened = status == HAR_OK;
        return status;
    }
    if (!opened) {
        return HAR_INVALID;
    }
    HarView view = get_view();
    if (code == REBUILD) {
        /* Full-vector baseline replays every vector from the current labels after an edit. */
        if (view.layout != HAR_VECTORS) {
            return HAR_INVALID;
        }
        return har_open(&state, &view, &nv);
    }
    if (code == EDIT || code == SNAPSHOT_EDIT) {
        if (mailbox.first > UINT16_MAX || mailbox.end > UINT16_MAX ||
            mailbox.from >= HAR_CLASSES || mailbox.to >= HAR_CLASSES) {
            return HAR_INVALID;
        }
        HarEdit edit = {mailbox.edit_id, (uint16_t)mailbox.first, (uint16_t)mailbox.end,
                        (uint8_t)mailbox.from, (uint8_t)mailbox.to};
        HarStatus status = code == EDIT ? har_edit(&state, &view, &nv, edit) :
                           har_edit_snapshot(&state, &view, &nv, edit);
        if (status == HAR_OK && (uint32_t)run->cut == run->operations) {
            reset_now();
        }
        return status;
    }
    if (code == COMPACT) {
        HarStatus status = har_compact(&state, &view, &nv);
        if (status == HAR_OK && (uint32_t)run->cut == run->operations) {
            reset_now();
        }
        return status;
    }
    if (code == INFER) {
        int32_t raw[HAR_DIM];
        int16_t embedding[HAR_DIM];
        int16_t prototypes[HAR_CLASSES][HAR_DIM];
        if (mailbox.prior_count > 4096) {
            return HAR_INVALID;
        }
        har_features(input_samples[0], raw);
        har_embed(raw, &input_metric, embedding);
        har_centroids(&state, &input_prior[0][0], mailbox.prior_count, &prototypes[0][0]);
        mailbox.prediction = har_predict(embedding, &prototypes[0][0]);
        return HAR_OK;
    }
    return HAR_INVALID;
}

void firmware_main(void)
{
    RCC_APB2ENR |= 1u << 4;
    USART_BRR = 0x008B; /* Nominal 115200 baud with a 16 MHz peripheral clock. */
    USART_CR1 = (1u << 13) | (1u << 3);
    if (mailbox.magic != MAILBOX_MAGIC) {
        mailbox.magic = MAILBOX_MAGIC;
        mailbox.request = 0;
        mailbox.response = 0;
        mailbox.boot_count = 0;
    }
    mailbox.boot_count++;
    mailbox.scratch_bytes = 0;
    USART_DR = 'B';
    for (;;) {
        if (mailbox.request == mailbox.response) {
            continue;
        }
        FlashRun run = {mailbox.cut, mailbox.mask, 0, 0, 0};
        uint32_t sequence = mailbox.request;
        uint32_t code = mailbox.command;
        operation_begin();
        HarStatus result = command(code, &run);
        operation_end();
        if (result == HAR_IO || result == HAR_CORRUPT) {
            opened = 0;
        }
        mailbox.status = result;
        mailbox.operations = run.operations;
        mailbox.program_bytes = run.program_bytes;
        mailbox.erase_count = run.erase_count;
        mailbox.last_id = opened ? state.last_id : 0;
        mailbox.generation = opened ? state.generation : 0;
        mailbox.bank = opened ? state.bank : 0;
        mailbox.next_slot = opened ? state.next_slot : 0;
        mailbox.state_hash = opened ? model_hash() : 0;
        mailbox.response = sequence;
        __asm__ volatile("dsb" ::: "memory");
        USART_DR = 'D';
    }
}
