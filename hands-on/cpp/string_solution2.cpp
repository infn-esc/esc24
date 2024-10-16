#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <cstring>
#include <cassert>
#include <memory>
#include <type_traits>

class String {
  std::unique_ptr<char[]> s_; // nullptr or null-terminated
 public:
  String() = default;
  String(char const* s)
  {
    assert(s);
    auto size = std::strlen(s) + 1;
    s_.reset(new char[size]);
    memcpy(s_.get(), s, size);
  }
  ~String() = default;
  String(String const& other)
  {
    assert(other.s_);
    auto size = std::strlen(other.s_.get()) + 1;
    s_.reset(new char[size]);
    memcpy(s_.get(), other.s_.get(), size);
  }
  String& operator=(String const& other)
  {
    String tmp(other);
    std::swap(s_, tmp.s_);
    return *this;
  }

  String(String&& tmp) = default;
  // : s_(std::move(tmp.s_)) {}

  // if you want to measure the effect of noexcept replace the above with
  // String(String&& tmp)
  //     : s_(std::move(tmp.s_))
  // {
  // }  

  String& operator=(String&& tmp) = default;
  // { s_ = std::move(tmp.s_); return *this; }

  std::size_t size() const
  {
    return s_ ? std::strlen(s_.get()) : 0;
  }
  char const* c_str() const
  {
    return s_.get();
  }
  char& operator[](std::size_t n)
  {
    assert(s_ && n < size());
    return s_[n];
  }
  char const& operator[](std::size_t n) const
  {
    assert(s_ && n < size());
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

  auto d = test();
  std::cout << d.count() << " \n";
}

static_assert(
    std::is_default_constructible<String>::value,
    "String is not default constructible"
);
static_assert(
    std::is_copy_constructible<String>::value,
    "String is not copy constructible"
);
static_assert(
    std::is_copy_assignable<String>::value,
    "String is not copy assignable"
);
static_assert(
    std::is_move_constructible<String>::value,
    "String is not move constructible"
);
static_assert(
    std::is_move_assignable<String>::value,
    "String is not move assignable"
);
static_assert(
    std::is_destructible<String>::value,
    "String is not destructible"
);
static_assert(
    std::is_nothrow_default_constructible<String>::value,
    "String is not noexcept default constructible"
);
static_assert(
    !std::is_nothrow_copy_constructible<String>::value,
    "String cannot be noexcept copy constructible"
);
static_assert(
    !std::is_nothrow_copy_assignable<String>::value,
    "String cannot be noexcept copy assignable"
);
static_assert(
    std::is_nothrow_move_constructible<String>::value,
    "String is not noexcept move constructible"
);
static_assert(
    std::is_nothrow_move_assignable<String>::value,
    "String is not noexcept move assignable"
);
static_assert(
    std::is_nothrow_destructible<String>::value,
    "String is not noexecpt destructible"
);
