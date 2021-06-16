import itertools
import time
import struct
import sys
import os, psutil

from bitarray import bitarray, util

sys.setrecursionlimit(999999)


def linf_dist(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def l1_dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class CachedIndependencePolynomial:
    def __init__(self, N, dist):
        self.N = N

        # List all the points + dists in the graph.
        pts = []
        dists = []
        for x in range(-N, N + 1):
            for y in range(-N, N + 1):
                if dist((0, 0), (x, y)) <= N:
                    pts.append((x, y))
                    dists.append(l1_dist((-N, -N), (x, y)))

        # Sort the points on the given distsances.
        self.dists, self.pts = zip(*sorted(zip(dists, pts)))

        # Calculate the mapping (r,c) -> index in the pts array
        pt2idx = {}
        for i, pt in enumerate(self.pts):
            pt2idx[pt] = i
        self.center = pt2idx[(0, 0)]

        # Store index mapping for mirrored (x=y) graph.
        self.idx_mirror = []
        for i, pt in enumerate(self.pts):
            self.idx_mirror.append(pt2idx[pt[1], pt[0]])

        # Calculate nbrs for each point: L, U, R, D.
        self.nbrs = []
        for pt in self.pts:
            x, y = pt
            nbrs_v = []
            for dx, dy in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                if (x + dx, y + dy) in pt2idx:
                    nbrs_v.append(pt2idx[(x + dx, y + dy)])
            self.nbrs.append(nbrs_v)

        # For each dist we store a cache.
        self.cache = [{} for _ in range(self.dists[-1] + 1)]

    def plot(self):
        import matplotlib.pyplot as plt
        import numpy as np

        G_n = np.array(self.pts)
        plt.scatter(G_n[:, 0], G_n[:, 1])
        for v, pt in enumerate(self.pts):
            for n_v in self.nbrs[v]:
                plt.plot([G_n[v][0], G_n[n_v][0]], [G_n[v][1], G_n[n_v][1]],
                         c='b')

        plt.show()

    # Calculate the polynomial for the given sub-graph.
    def calculate_loop(self, mask, mask_m=None, store_depth=0):
        if mask.count() == 0: return [1]
        if mask.count() == 1: return [1, 1]

        # Find the last vertex in the subgraph, the vertex with largest dist.
        v = util.rindex(mask)
        dist = self.dists[v]

        # If this result is cached, return from cache.
        mask_b = mask.tobytes()
        if mask_b in self.cache[dist]:
            return self.cache[dist][mask_b]

        # If the mirrored (in x=y) result is cached, return from cache.
        mask_m_b = mask_m.tobytes()
        if mask_m_b in self.cache[dist]:
            return self.cache[dist][mask_m_b]

        # The result is not cached, recurse on v.

        # Subgraph (and its mirror) without v.
        maskv = mask.copy()
        maskv[v] = False
        maskv_m = mask_m.copy()
        maskv_m[self.idx_mirror[v]] = False

        # Subgraph (and its mirror) without N[v].
        maskNv = maskv.copy()
        for nbr in self.nbrs[v]:
            maskNv[nbr] = False
        maskNv_m = maskv_m.copy()
        for nbr in self.nbrs[v]:
            maskNv_m[self.idx_mirror[nbr]] = False

        # Recurse.
        resultv = self.calculate_loop(maskv, maskv_m, max(0, store_depth - 1))
        resultNv = [0] + self.calculate_loop(maskNv, maskNv_m, 2)
        result = [
            x + y
            for x, y in itertools.zip_longest(resultv, resultNv, fillvalue=0)
        ]

        if store_depth <= 0:  # Store this result in cache.
            self.cache[dist][mask_b] = result
        return result

    def clear_cache(self):
        for d in self.cache:
            d.clear()

    # Calculate the independence polynomial for the given graph G.
    # We do this generating independence polynomials for the
    # subgraphs of G having max_distance <= 1, 2, \ldots.
    def calculate(self, mask):
        mask_n = bitarray(len(self.pts))
        mask_n.setall(False)
        max_dist = 4
        for v, dist in enumerate(self.dists):
            # mask_n will contain all pts having dist <= max_dist.
            if (dist > max_dist):
                max_dist = dist
                mask_n &= mask

                # We can clear the cache of subgraphs with max_dist <= dist - 5.
                self.cache[dist - 5].clear()

                # Calculate the independence polynomial for mask_n.
                self.calculate_loop(mask_n, mask_n)
                assert len(self.cache[dist - 5]) == 0
            mask_n[v] = True

        # Finally, return the independence polynomial of the required graph.
        return self.calculate_loop(mask, mask)


for N in range(99):
    print('--- Consider G_N graph with l1 dist for N = {} ---\n'.format(N))
    independence_poly = CachedIndependencePolynomial(N, l1_dist)
    #independence_poly.plot()

    # Calculate independence poly for graph without center.
    mask = bitarray(len(independence_poly.pts))
    mask.setall(True)
    mask[independence_poly.center] = False

    print('Independence polynomial for graph without center.')
    time_start = time.time()
    resultv = independence_poly.calculate(mask)
    print(resultv)
    print('Took {}s. Mem usage: {}mb.\n'.format(
        time.time() - time_start,
        psutil.Process(os.getpid()).memory_info().rss / 1024**2),
          flush=True)

    independence_poly.clear_cache()  # Reset cache, memory savings.
    for nbr in independence_poly.nbrs[independence_poly.center]:
        mask[nbr] = False

    print('Independence polynomial for graph without N[center].')
    time_start = time.time()
    resultNv = [0] + independence_poly.calculate(mask)
    print(resultNv)
    print('Took {}s. Mem usage: {}mb.\n'.format(
        time.time() - time_start,
        psutil.Process(os.getpid()).memory_info().rss / 1024**2),
          flush=True)

    print('Independence polynomial for entire graph.')
    print([
        x + y for x, y in itertools.zip_longest(resultv, resultNv, fillvalue=0)
    ])

    print('')
