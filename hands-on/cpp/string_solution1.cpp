#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <cstring>

class String {
  char* s_ = nullptr; // nullptr or null-terminated
 public:
  String() = default;
  String(char const* s)
  {
    size_t size = std::strlen(s) + 1;
    s_ = new char[size];
    std::memcpy(s_, s, size);
  }
  ~String()
  {
    delete [] s_;
  }
  String(String const& other)
  {
    size_t size = std::strlen(other.s_) + 1;
    s_ = new char[size];
    std::memcpy(s_, other.s_, size);
  }
  String(String&& tmp) // noexcept
      : s_(tmp.s_)
  {
    tmp.s_ = nullptr;
  }
  String& operator=(String const& other)
  {
    // use the copy-and-swap idiom. Not terribly efficient but safe and doesn't
    // require a self-assignment test
    String tmp(other);
    std::swap(s_, tmp.s_);
    return *this;
  }
  String& operator=(String&& tmp)
  {
    // do not check for self-assignment. situations like
    // x = std::move(x)
    // are rare
    std::swap(s_, tmp.s_);
    return *this;
  }
  std::size_t size() const
  {
    return s_ ? std::strlen(s_) : 0;
  }
  char const* c_str() const
  {
    return s_;
  }
  char& operator[](std::size_t n)
  {
    return s_[n];
  }
  char const& operator[](std::size_t n) const
  {
    return s_[n];
  }
};

String get_string()
{
  return String{"Consectetur adipiscing elit"};
}

std::chrono::duration<float> test()
{
  auto s = get_string();
  std::vector<String> v(10000000, s);
  auto start = std::chrono::high_resolution_clock::now();
  v.push_back(s);
  return std::chrono::high_resolution_clock::now() - start;
}

int main()
{
  String const s1("Lorem ipsum dolor sit amet");

  String s2 = get_string();

  String s3;
  s3 = s1;

  String s4;
  s4 = std::move(s2);

  char& c1 = s4[4];
  char const& c2 = s1[4];

  std::cout << s3.c_str() << '\n';

  // auto d = test();
  // std::cout << d.count() << " \n";
}
