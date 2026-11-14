CROSS_COMPILE ?= arm-none-eabi-
ARM_CC = $(CROSS_COMPILE)gcc
ARM_SIZE = $(CROSS_COMPILE)size
BUILD ?= build
TARGET = $(BUILD)/har-stm32f4.elf
CORTEX_FLAGS = -mcpu=cortex-m4 -mthumb -std=c11 -Os -g -ffreestanding \
               -fno-builtin -ffunction-sections -fdata-sections -Wall -Wextra -Werror
SOURCES = firmware/startup.c firmware/target.c firmware/har.c firmware/features.c

.PHONY: firmware size test
firmware: $(TARGET)

$(TARGET): $(SOURCES) firmware/har.h firmware/stm32f4.ld Makefile
	mkdir -p $(BUILD)
	$(ARM_CC) $(CORTEX_FLAGS) -nostdlib -Wl,--gc-sections -Wl,-T,firmware/stm32f4.ld \
		-Wl,-Map,$(BUILD)/har-stm32f4.map $(SOURCES) -o $@
	$(ARM_SIZE) $@

size: firmware
	$(ARM_SIZE) -A $(TARGET)

test:
	python -m unittest discover -s tests -v
