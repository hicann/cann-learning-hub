/* ==========================================================================
 * transpose_ext.c -- Extension: turning the lesson into an optimization.
 *
 * The Traversal Order Experiment showed that a column-major traversal is
 * slow. A matrix transpose is the case where that cannot simply be avoided:
 * whichever way the loops are written, one of the two matrices is walked
 * down a column.
 *
 *     for (i) for (j)  b[j * n + i] = a[i * n + j];
 *              ^ reads a row of A          ^ writes a column of B
 *
 * The way out is not to change the order of the elements but the SIZE of the
 * region worked on: transpose one small tile at a time, small enough that the
 * tile of A and the tile of B both stay in cache while they are being used.
 * Every cache line brought in is then fully consumed before it is evicted.
 *
 * This is the same idea that Chapter 3 will apply to matrix multiplication.
 * ========================================================================== */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NTIMES 3

double get_time_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

// ---------------------------------------------------------
// The straightforward transpose, and the reference result.
// ---------------------------------------------------------
void transpose_naive(const int32_t *a, int32_t *b, long n) {
  for (long i = 0; i < n; i++)
    for (long j = 0; j < n; j++) b[j * n + i] = a[i * n + j];
}

// ---------------------------------------------------------
// The blocked transpose. Same elements, same assignments, different order.
// ---------------------------------------------------------
void transpose_blocked(const int32_t *a, int32_t *b, long n, long t) {
  for (long ii = 0; ii < n; ii += t) {
    for (long jj = 0; jj < n; jj += t) {
      // The last tile of a row or column is smaller when n is not a multiple
      // of t. Clamping here is what makes the code work for any n.
      long imax = (ii + t < n) ? ii + t : n;
      long jmax = (jj + t < n) ? jj + t : n;
      for (long i = ii; i < imax; i++)
        for (long j = jj; j < jmax; j++) b[j * n + i] = a[i * n + j];
    }
  }
}

int main(int argc, char **argv) {
  long n = (argc > 1) ? atol(argv[1]) : 4096;
  if (n < 8) n = 4096;

  long tiles[] = {8, 16, 32, 64, 128};
  int ntiles = (int)(sizeof(tiles) / sizeof(tiles[0]));

  size_t bytes = (size_t)n * (size_t)n * sizeof(int32_t);
  int32_t *a = (int32_t *)malloc(bytes);
  int32_t *ref = (int32_t *)malloc(bytes);
  int32_t *b = (int32_t *)malloc(bytes);
  if (!a || !ref || !b) {
    printf("Alloc failed\n");
    return 1;
  }
  for (long i = 0; i < n * n; i++) a[i] = (int32_t)(i * 2654435761u);

  printf("============================================================\n");
  printf(" Blocked Transpose Experiment\n");
  printf(" Matrix : %ld x %ld int32  (%.1f MiB per matrix)\n", n, n,
         (double)bytes / (1024.0 * 1024.0));
  printf(" A tile of %ld x %ld int32 occupies %ld B; two of them must fit\n",
         tiles[0], tiles[0], tiles[0] * tiles[0] * 4);
  printf(" comfortably in L1 for the blocking to pay off.\n");
  printf("============================================================\n");

  // Golden reference
  transpose_naive(a, ref, n);

  printf("\n------------------------------------------------------\n");
  printf("| %-16s | %9s | %7s | %5s |\n", "Method", "Time (ms)", "Speedup",
         "Check");
  printf("|------------------|-----------|---------|-------|\n");

  memset(b, 0, bytes);
  double t0 = get_time_ms();
  for (int r = 0; r < NTIMES; r++) transpose_naive(a, b, n);
  double t_base = (get_time_ms() - t0) / NTIMES;
  printf("| %-16s | %9.3f | %5.2f x | %5s |\n", "Naive", t_base, 1.0,
         memcmp(ref, b, bytes) == 0 ? "PASS" : "FAIL");

  double best_t = t_base;
  long best_tile = 0;
  for (int k = 0; k < ntiles; k++) {
    // Clear: otherwise a correct result left by the previous variant would
    // mask a variant that writes nothing at all.
    memset(b, 0, bytes);
    t0 = get_time_ms();
    for (int r = 0; r < NTIMES; r++) transpose_blocked(a, b, n, tiles[k]);
    double t = (get_time_ms() - t0) / NTIMES;

    int ok = (memcmp(ref, b, bytes) == 0);
    char name[32];
    snprintf(name, sizeof(name), "Blocked %ld", tiles[k]);
    printf("| %-16s | %9.3f | %5.2f x | %5s |\n", name, t, t_base / t,
           ok ? "PASS" : "FAIL");
    // Only a variant that produced the right matrix may claim to be fastest.
    if (ok && t < best_t) {
      best_t = t;
      best_tile = tiles[k];
    }
  }
  printf("------------------------------------------------------\n");
  if (best_tile)
    printf("  Best tile: %ld x %ld  (%.2f x faster than the naive version)\n",
           best_tile, best_tile, t_base / best_t);
  else
    printf("  No tile size beat the naive version on this machine.\n");

  free(a);
  free(ref);
  free(b);
  return 0;
}
