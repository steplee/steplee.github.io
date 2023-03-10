```meta
title: Expanding files with fallocate
date: 2023/01/21
tags: programming, os, mmap
```
# Expanding Files with `fallocate`
I discovered a new trick the other week. You can insert wholes into files using `fallocate` with the `falloc_fl_insert_range` mode. The amazing catch is that it even works with `mmap`ed files, and no programmer synchronization nor book-keeping needs to be done!

I decided to exploit this trick to make my large tiled image library/format simpler. Previously it used LMDB and stored georefenced imagery on one base level, plus optionally overviews in seperate LMDB `db`s (but in the same file). The 'overviews' are downsampled versions of the base image support (these are like disk-based mip-maps -- if we want to access a large area of the base image, we'd need to sample huge numbers of pixels and then subsample them). Using LMDB worked pretty well, but I when creating the files I had to use multiple transactions so that (I assume) LMDB could rebalance the B+ tree. And I always like to try to implement a simpler approach when I can.

To that effect, I wanted to just have a sorted array of keys and a binary blob of values that the keys know how to index into. Everything is flat and contiguous here. But we do not know the number of tiles or the required size of the binary blob until we visit all tiles and encode them, and we using a two-phase approach is unattractive.
We can do something like a standard expanding array (`std::vector`) and double the capacity of our arrays when we out grow them. Doing this in memory requires copying the old elements a newly allocated, larger array. Doing this on disk would be way too slow and use require 2x the free space.
Here is where the use of `fallocate` comes in.
If we need to expand the `keys` buffer, we use `fallocate` to insert a range at the end of it, just before the beginning of the `vals` buffer.
If we need to expand the `vals` buffer, we just extend the file.
We just have to fix the pointers and lengths, and that's it!

In actuality, the `keys` buffer is actually also storing `k2vs` data, which tells the key the offset into the `values` array that it's value is located.
This is valid because the writer application can do the writes in order. Unordered writes would not work -- the best bet would be a B+ tree. I actually implemented a B+ tree to start off this project, but then I discovered the trick and decided to go with that.

The relevant code is [here](https://github.com/steplee/frast/blob/v2.0/frast2/flat/flat_env.cc#L198)
