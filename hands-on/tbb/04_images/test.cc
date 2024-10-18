#include <chrono>
#include <cassert>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define FMT_HEADER_ONLY
#include "fmt/core.h"
#include "fmt/color.h"

using namespace std::literals;

struct Image {
  unsigned char* data_ = nullptr;
  int width_ = 0;
  int height_ = 0;
  int channels_ = 0;

  Image() {
  }

  Image(std::string const& filename) {
    open(filename);
  }

  Image(int width, int height, int channels) :
    width_(width),
    height_(height),
    channels_(channels)
  {
    size_t size = width_ * height_ * channels_;
    data_ = static_cast<unsigned char*>(stbi__malloc(size));
    std::memset(data_, 0x00, size);
  }

  ~Image() {
    close();
  }

  // copy constructor
  Image(Image const& img) :
    width_(img.width_),
    height_(img.height_),
    channels_(img.channels_)
  {
    size_t size = width_ * height_ * channels_;
    data_ = static_cast<unsigned char*>(stbi__malloc(size));
    std::memcpy(data_, img.data_, size);
  }

  // copy assignment
  Image& operator=(Image const& img) {
    // avoid self-copies
    if (&img == this) {
      return *this;
    }

    // free any existing image data
    close();

    width_ = img.width_;
    height_ = img.height_;
    channels_ = img.channels_;
    size_t size = width_ * height_ * channels_;
    data_ = static_cast<unsigned char*>(stbi__malloc(size));
    std::memcpy(data_, img.data_, size);

    return *this;
  }

  // move constructor
  Image(Image && img) :
    data_(img.data_),
    width_(img.width_),
    height_(img.height_),
    channels_(img.channels_)
  {
    // take owndership of the image data
    img.data_ = nullptr;
  }

  // move assignment
  Image& operator=(Image && img) {
    // avoid self-moves
    if (&img == this) {
      return *this;
    }

    // free any existing image data
    close();

    // copy the image properties
    width_ = img.width_;
    height_ = img.height_;
    channels_ = img.channels_;

    // take owndership of the image data
    data_ = img.data_;
    img.data_ = nullptr;

    return *this;
  }

  void open(std::string const& filename) {
    data_ = stbi_load(filename.c_str(), &width_, &height_, &channels_, 0);
    if (data_ == nullptr) {
        throw std::runtime_error("Failed to load "s + filename);
    }
    std::cout << "Loaded image with " << width_ << " x " << height_ << " pixels and " << channels_ << " channels from " << filename << '\n';
  }

  void write(std::string const& filename) {
    if (filename.ends_with(".png")) {
      int status = stbi_write_png(filename.c_str(), width_, height_, channels_, data_, 0);
      if (status == 0) {
        throw std::runtime_error("Error while writing PNG file "s + filename);
      }
    } else if (filename.ends_with(".jpg") or filename.ends_with(".jpeg")) {
      int status = stbi_write_jpg(filename.c_str(), width_, height_, channels_, data_, 95);
      if (status == 0) {
        throw std::runtime_error("Error while writing JPEG file "s + filename);
      }
    } else {
      throw std::runtime_error("File format "s + filename + "not supported"s);
    }
  }

  void close() {
    if (data_ != nullptr) {
      stbi_image_free(data_);
    }
    data_ = nullptr;
  }

  // show an image on the terminal, using up to max_width columns (with one block per column) and up to max_height lines (with two blocks per line)
  void show(int max_width, int max_height) {
    if (data_ == nullptr) {
      return;
    }

    int width, height;
    if (width_ > height_) {
      width = max_width;
      height = max_height * height_ / width_;
    } else {
      width = max_width * width_ / height_;
      height = max_height;
    }

    // two blocks per line
    for (int j = 0; j < height; j += 2) {
      int y1 = j * height_ / height;
      int y2 = (j + 1) * height_ / height;
      // one block per column
      for (int i = 0; i < width; ++i) {
        int x = i * width_ / width;
        int p = (y1 * width_ + x) * channels_;
        int r = data_[p];
        int g = data_[p + 1];
        int b = data_[p + 2];
        auto style = fmt::fg(fmt::rgb(r, g, b));
        if (y2 < height_) {
          p = (y2 * width_ + x) * channels_;
          r = data_[p];
          g = data_[p + 1];
          b = data_[p + 2];
          style |= fmt::bg(fmt::rgb(r, g, b));
        }
        std::cout << fmt::format(style, "▀");
      }
      std::cout << '\n';
    }
  }

};


int main(int argc, const char* argv[]) {
    std::vector<std::string> files;
    if (argc == 1) {
      // no arguments, use a single default image
      files = { "image.png"s };
    } else {
      files.reserve(argc - 1);
      for (int i = 1; i < argc; ++i) {
        files.emplace_back(argv[i]);
      }
    }

    std::vector<Image> images;
    images.resize(files.size());
    for (unsigned int i = 0; i < files.size(); ++i) {
      auto& img = images[i];
      img.open(files[i]);
      img.show(80, 80);
      img.write(fmt::format("out{:02d}.jpg", i));
    }

    return 0;
}
