#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <cstring>

class String {
  char* s_ = nullptr; // nullptr or null-terminated
 public:
  String() = default;
  String(char const* s) {
    size_t size = std::strlen(s) + 1;
    s_ = new char[size];
    std::memcpy(s_, s, size);
  }
  ~String() { delete [] s_; }
  String(String const& other) {
    size_t size = std::strlen(other.s_) + 1;
    s_ = new char[size];
    std::memcpy(s_, other.s_, size);
  }
  String(String&& tmp)
      : s_(tmp.s_) {
    tmp.s_ = nullptr;
  }
  String& operator=(String const& other)
  {
    // to be implemented
  }
  String& operator=(String&& other)
  {
    // to be implemented
  }
  std::size_t size() const {
    return s_ ? strlen(s_) : 0;
  }
  char const* c_str() const
  {
    // to be implemented;
  }
  char& operator[](std::size_t n)
  {
    // to be implemented
  }
  char const& operator[](std::size_t n) const
  {
    // to be implemented
  }
};

String get_string()
{
  return String{"Consectetur adipiscing elit"};
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
}
