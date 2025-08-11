// jpeg_encoder.cpp
#include <turbojpeg.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

extern "C" {
    // Encodes a raw BGR frame to JPEG
    unsigned char* encode_to_jpeg(unsigned char* bgr_data, int width, int height, int quality, unsigned long* jpeg_size) {
        tjhandle compressor = tjInitCompress();
        if (!compressor) {
            std::cerr << "[TurboJPEG] Init failed." << std::endl;
            return nullptr;
        }

        unsigned char* jpeg_buf = nullptr;
        int subsamp = TJSAMP_420;

        int success = tjCompress2(
            compressor,
            bgr_data,
            width,
            0, height,
            TJPF_BGR,
            &jpeg_buf,
            jpeg_size,
            subsamp,
            quality,
            TJFLAG_FASTDCT
        );

        tjDestroy(compressor);

        if (success != 0) {
            std::cerr << "[TurboJPEG] Compress failed: " << tjGetErrorStr() << std::endl;
            return nullptr;
        }

        return jpeg_buf;
    }

    // Frees JPEG buffer after sending to Python
    void free_jpeg_buffer(unsigned char* jpeg_buf) {
        tjFree(jpeg_buf);
    }
}
