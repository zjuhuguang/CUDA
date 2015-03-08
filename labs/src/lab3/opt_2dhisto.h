#ifndef OPT_KERNEL
#define OPT_KERNEL

void opt_2dhisto(uint32_t* D_input, int, int, uint32_t* g_bins, uint8_t* D_bins);

/* Include below the function headers of any other functions that you implement */

void* AllocateDevice(size_t size);

void MemCpyToDevice(void* dst, void* src, size_t size);

void CopyFromDevice(void* D_host, void* D_device, size_t size);

void FreeDevice(void* D_device);

#endif
