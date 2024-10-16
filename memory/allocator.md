---
title: Allocating Memory 
layout: main 
category: memory 
---

# Allocating Memory 

### Exercise0 : Single Allocation 
in hands-on/memory/allocators
Modify the file single_bufferSoA.cc

Let's write a different version of our GoodParticleSoA datastructure:
- Each member is a pointer, that will point to the correct memory location where you are reserving space
- Should take as constructor agument and integer `N` (number of particles)
	- It needs an additional member, representing a memory buffer (`void* buffer`/`char* buffer`)
	- Instead of allocating memory separately for each SoA column (member) you should allocate all the required memory in one single operation. Assigning the pointer to this location to `buffer`.
	- Now you have to assign to your members the correct memory address within this buffer.
	- The destructor should free the `buffer` memory

### Exercise1: Caching Allocator 
Have a look at the slides with the illustration on how the caching allocator works
hands-on/memory/allocators/caching_allocator, modify CachingAllocator.h
- Write a class `CachingAllocator`. This class should have three methods:
	- `allocate(std::size_t requiredMemory);`
		- should allocate a new block of memory if not already cached and returning the pointer to this memory location
	- `deallocate(T* ptr)`; where ptr has been allocated with `malloc`/`calloc`/`aligned_alloc`
		- should put the allocated block in the container of cached blocks
	- `free()` releases all the memory allocated.

Write a new SoA datastrcture with some members and try to allocate memory for each member, and then deallocate the memory. Do this operation many times and try to compare it with "standard" allocation (e.g. using `std::vector` in your SoA).

Note: Add debug info in your caching allocator while you are designing it, it will be extremely helpful to understand what is going on with your allocator
