#ifndef TRACKING_HPP
#define TRACKING_HPP

#include <cstdio>

struct Tracking
{
  Tracking() { std::puts(__PRETTY_FUNCTION__); }
  Tracking(Tracking const&) { std::puts(__PRETTY_FUNCTION__); }
  Tracking& operator=(Tracking const&) { std::puts(__PRETTY_FUNCTION__); return *this; }
  Tracking(Tracking&&) { std::puts(__PRETTY_FUNCTION__); }
  Tracking& operator=(Tracking&&) { std::puts(__PRETTY_FUNCTION__); return *this; }
  ~Tracking() { std::puts(__PRETTY_FUNCTION__); }
};

#endif
