# Polymorphic Tree in C
I was reading some kernel code today to try and see how [BCC](https://github.com/iovisor/bcc) works with USDTs/uprobes and the ftrace/perf kernel subsystems.

I came across an interesting idiom in C for having allowing a balanced binary tree implementation to be used with different types.
`uprobes.c` contains the following line `struct uprobe *u = rb_entry(n, struct uprobe, rb_node);`, which expands to `struct uprobe *u = container_of(n, struct uprobe, rb_node);`
This `container_of` macro is defined in `include/linux/kernel.h`, and so is probably widely used. It's defined as
```C
/**
 * container_of - cast a member of a structure out to the containing structure
 * @ptr:	the pointer to the member.
 * @type:	the type of the container struct this is embedded in.
 * @member:	the name of the member within the struct.
 *
 */
#define container_of(ptr, type, member) ({			\
	const typeof(((type *)0)->member) * __mptr = (ptr);	\
	(type *)((char *)__mptr - offsetof(type, member)); })
```

This takes a minute to parse, but it is just finding the pointer of a struct from a certain member of the struct.
So for example the uprobe struct has a member `rb_node`, and the above `rb_entry` macro finds the pointer to the base struct `uprobe` given a pointer to the inner `rb_node`.

This is kind of the reverse to how you'd normally do it in an object-oriented container based way.
In languages with generics you'd have a parameterized class, where the tree nodes contain pointers to your custom type.
In this approach, your custom type has a pointer to the tree node.
