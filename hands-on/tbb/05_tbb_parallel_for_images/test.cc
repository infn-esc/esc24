#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <syncstream>
#include <vector>

#ifdef __linux__
#include <sys/ioctl.h>
#include <unistd.h>
#endif

#include <tbb/tbb.h>

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

  Image() {}

  Image(std::string const& filename) { open(filename); }

  Image(int width, int height, int channels) : width_(width), height_(height), channels_(channels) {
    size_t size = width_ * height_ * channels_;
    data_ = static_cast<unsigned char*>(stbi__malloc(size));
    std::memset(data_, 0x00, size);
  }

  ~Image() { close(); }

  // copy constructor
  Image(Image const& img) : width_(img.width_), height_(img.height_), channels_(img.channels_) {
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
  Image(Image&& img) : data_(img.data_), width_(img.width_), height_(img.height_), channels_(img.channels_) {
    // take owndership of the image data
    img.data_ = nullptr;
  }

  // move assignment
  Image& operator=(Image&& img) {
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
    std::osyncstream out(std::cout);

    data_ = stbi_load(filename.c_str(), &width_, &height_, &channels_, 0);
    if (data_ == nullptr) {
      throw std::runtime_error("Failed to load "s + filename);
    }
    out << "Loaded image with " << width_ << " x " << height_ << " pixels and " << channels_ << " channels from "
        << filename << '\n';
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

    // two blocks per line
    max_height = max_height * 2;

    // find the best size given the max width and height and the image aspect ratio
    int width, height;
    if (width_ * max_height > height_ * max_width) {
      width = max_width;
      height = max_width * height_ / width_;
    } else {
      width = max_height * width_ / height_;
      height = max_height;
    }

    std::osyncstream out(std::cout);

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
        out << fmt::format(style, "▀");
      }
      out << '\n';
    }

    // out is streamed to std::cout and flushed
  }
};

bool verbose = false;

// make a scaled copy of an image
Image scale(Image const& src, int width, int height) {
  if (width == src.width_ and height == src.height_) {
    // if the dimensions are the same, return a copy of the image
    return src;
  }

  // create a new image
  Image out(width, height, src.channels_);

  auto start = std::chrono::steady_clock::now();

  for (int y = 0; y < height; ++y) {
    // map the row of the scaled image to the nearest rows of the original image
    float yp = static_cast<float>(y) * src.height_ / height;
    int y0 = std::clamp(static_cast<int>(std::floor(yp)), 0, src.height_ - 1);
    int y1 = std::clamp(static_cast<int>(std::ceil(yp)), 0, src.height_ - 1);

    // interpolate between y0 and y1
    float wy0 = yp - y0;
    float wy1 = y1 - yp;
    // if the new y coorindate maps to an integer coordinate in the original image, use a fake distance from identical values corresponding to it
    if (y0 == y1) {
      wy0 = 1.f;
      wy1 = 1.f;
    }
    float dy = wy0 + wy1;

    for (int x = 0; x < width; ++x) {
      int p = (y * out.width_ + x) * out.channels_;

      // map the column of the scaled image to the nearest columns of the original image
      float xp = static_cast<float>(x) * src.width_ / width;
      int x0 = std::clamp(static_cast<int>(std::floor(xp)), 0, src.width_ - 1);
      int x1 = std::clamp(static_cast<int>(std::ceil(xp)), 0, src.width_ - 1);

      // interpolate between x0 and x1
      float wx0 = xp - x0;
      float wx1 = x1 - xp;
      // if the new x coordinate maps to an integer coordinate in the original image, use a fake distance from identical values corresponding to it
      if (x0 == x1) {
        wx0 = 1.f;
        wx1 = 1.f;
      }
      float dx = wx0 + wx1;

      // bi-linear interpolation of all channels
      int p00 = (y0 * src.width_ + x0) * src.channels_;
      int p10 = (y1 * src.width_ + x0) * src.channels_;
      int p01 = (y0 * src.width_ + x1) * src.channels_;
      int p11 = (y1 * src.width_ + x1) * src.channels_;

      for (int c = 0; c < src.channels_; ++c) {
        out.data_[p + c] =
            static_cast<unsigned char>(std::round((src.data_[p00 + c] * wx1 * wy1 + src.data_[p10 + c] * wx1 * wy0 +
                                                   src.data_[p01 + c] * wx0 * wy1 + src.data_[p11 + c] * wx0 * wy0) /
                                                  (dx * dy)));
      }
    }
  }

  auto finish = std::chrono::steady_clock::now();
  float ms = std::chrono::duration_cast<std::chrono::duration<float>>(finish - start).count() * 1000.f;
  if (verbose) {
    std::cerr << fmt::format("scale:      {:6.2f}", ms) << " ms\n";
  }

  return out;
}

// copy a source image into a target image, cropping any parts that fall outside the target image
void write_to(Image const& src, Image& dst, int x, int y) {
  // copying to an image with a different number of channels is not supported
  assert(src.channels_ == dst.channels_);

  // the whole source image would fall outside of the target image along the X axis
  if ((x + src.width_ < 0) or (x >= dst.width_)) {
    return;
  }

  // the whole source image would fall outside of the target image along the Y axis
  if ((y + src.height_ < 0) or (y >= dst.height_)) {
    return;
  }

  // find the valid range for the overlapping part of the images along the X and Y axes
  int src_x_from = std::max(0, -x);
  int src_x_to = std::min(src.width_, dst.width_ - x);
  int dst_x_from = std::max(0, x);
  //int dst_x_to   = std::min(src.width_ + x, dst.width_);
  int x_width = src_x_to - src_x_from;

  int src_y_from = std::max(0, -y);
  int src_y_to = std::min(src.height_, dst.height_ - y);
  int dst_y_from = std::max(0, y);
  //int dst_y_to   = std::min(src.height_ + y, dst.height_);
  int y_height = src_y_to - src_y_from;

  auto start = std::chrono::steady_clock::now();

  for (int y = 0; y < y_height; ++y) {
    int src_p = ((src_y_from + y) * src.width_ + src_x_from) * src.channels_;
    int dst_p = ((dst_y_from + y) * dst.width_ + dst_x_from) * dst.channels_;
    std::memcpy(dst.data_ + dst_p, src.data_ + src_p, x_width * src.channels_);
  }

  auto finish = std::chrono::steady_clock::now();
  float ms = std::chrono::duration_cast<std::chrono::duration<float>>(finish - start).count() * 1000.f;
  if (verbose) {
    std::cerr << fmt::format("write_to:   {:6.2f}", ms) << " ms\n";
  }
}

// convert an image to grayscale
Image grayscale(Image const& src) {
  // non-RGB images are not supported
  assert(src.channels_ >= 3);

  auto start = std::chrono::steady_clock::now();

  Image dst = src;
  for (int y = 0; y < dst.height_; ++y) {
    for (int x = 0; x < dst.width_; ++x) {
      int p = (y * dst.width_ + x) * dst.channels_;
      int r = dst.data_[p];
      int g = dst.data_[p + 1];
      int b = dst.data_[p + 2];
      // NTSC values for RGB to grayscale conversion
      int y = (299 * r + 587 * g + 114 * b) / 1000;
      dst.data_[p] = y;
      dst.data_[p + 1] = y;
      dst.data_[p + 2] = y;
    }
  }

  auto finish = std::chrono::steady_clock::now();
  float ms = std::chrono::duration_cast<std::chrono::duration<float>>(finish - start).count() * 1000.f;
  if (verbose) {
    std::cerr << fmt::format("grayscale:  {:6.2f}", ms) << " ms\n";
  }

  return dst;
}

// apply an RGB tint to an image
Image tint(Image const& src, int r, int g, int b) {
  // non-RGB images are not supported
  assert(src.channels_ >= 3);

  auto start = std::chrono::steady_clock::now();

  Image dst = src;
  for (int y = 0; y < dst.height_; ++y) {
    for (int x = 0; x < dst.width_; ++x) {
      int p = (y * dst.width_ + x) * dst.channels_;
      int r0 = dst.data_[p];
      int g0 = dst.data_[p + 1];
      int b0 = dst.data_[p + 2];
      dst.data_[p] = r0 * r / 255;
      dst.data_[p + 1] = g0 * g / 255;
      dst.data_[p + 2] = b0 * b / 255;
    }
  }

  auto finish = std::chrono::steady_clock::now();
  float ms = std::chrono::duration_cast<std::chrono::duration<float>>(finish - start).count() * 1000.f;
  if (verbose) {
    std::cerr << fmt::format("tint:       {:6.2f}", ms) << " ms\n";
  }

  return dst;
}

int main(int argc, const char* argv[]) {
  const char* verbose_env = std::getenv("VERBOSE");
  if (verbose_env != nullptr and std::strlen(verbose_env) != 0) {
    verbose = true;
  }

  std::vector<std::string> files;
  if (argc == 1) {
    // no arguments, use a single default image
    files = {"image.png"s};
  } else {
    files.reserve(argc - 1);
    for (int i = 1; i < argc; ++i) {
      files.emplace_back(argv[i]);
    }
  }

  int rows = 80;
  int columns = 80;
#if defined(__linux__) && defined(TIOCGWINSZ)
  if (isatty(STDOUT_FILENO)) {
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    if (w.ws_row > 1 and w.ws_col > 1) {
      rows = w.ws_row - 1;
      columns = w.ws_col - 1;
    }
  }
#endif

  std::vector<Image> images;
  images.resize(files.size());
  tbb::parallel_for<int>(0, files.size(), 1, [&](int i) {
    auto& img = images[i];
    img.open(files[i]);
    img.show(columns, rows);

    Image small = scale(img, img.width_ * 0.5, img.height_ * 0.5);
    Image gray = grayscale(small);
    Image tone1 = tint(gray, 168, 56, 172);  // purple-ish
    Image tone2 = tint(gray, 100, 143, 47);  // green-ish
    Image tone3 = tint(gray, 255, 162, 36);  // gold-ish

    Image out(img.width_, img.height_, img.channels_);
    write_to(tone1, out, 0, 0);
    write_to(tone2, out, img.width_ * 0.5, 0);
    write_to(tone3, out, 0, img.height_ * 0.5);
    write_to(gray, out, img.width_ * 0.5, img.height_ * 0.5);

    std::cout << '\n';
    out.show(columns, rows);
    out.write(fmt::format("out{:02d}.jpg", i));
  });

  return 0;
}
