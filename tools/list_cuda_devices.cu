#include "smoke/smoke.h"
#include <iostream>

int main() {
    auto devices = legacy_ai::query_all_devices();
    if (devices.empty()) {
        std::cerr << "No CUDA-capable devices found.\n";
        return 1;
    }
    legacy_ai::print_device_table(devices);
    return 0;
}
