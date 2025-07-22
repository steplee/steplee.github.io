```meta
title: The Best Function I've Ever Written
date: 2025/07/21
tags: programming, python, pytorch
```

This was used to implement hard negative mining for a model I was working on. I needed to layout a tensor so that all of the vectors assigned to KMeans centroids were grouped together, while randomizing the order of the clusters.

```python
# Here is a test snippet that demonstrates the desired behaviour:
#
#     a = torch.arange(3).repeat_interleave(3) # Cluster assignments per sample.
#                                              # so a = [0, 0, 0, 1, 1, 1, 2, 2, 2]

#     randomize_order_but_sequence_by_cluster_id(a) => [5, 3, 4, 6, 8, 7, 1, 2, 0]
#     randomize_order_but_sequence_by_cluster_id(a) => [0, 2, 1, 6, 8, 7, 3, 5, 4]
#     randomize_order_but_sequence_by_cluster_id(a) => [3, 4, 5, 0, 2, 1, 7, 8, 6]
#
# i.e. keep all samples within the same cluster together, but randomize there order in the output sequence,
#      while at the same time randomizing the outer order of all clusters.
#
def randomize_order_but_sequence_by_cluster_id(a):
    M = a.max().item()+1             # num clusters
    i = torch.randperm(len(a))       # randomize intra-cluster order
    o = torch.randperm(M)            # randomize inter-cluster order
    return i[o[a][i].sort().indices]
```
