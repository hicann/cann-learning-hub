/* threadpool2.c -- reference implementation of the extension. */
#include "threadpool2.h"

#include <pthread.h>
#include <stdlib.h>

typedef struct tp2_task {
  tp2_task_fn fn;
  void *arg;
  struct tp2_task *next;
} tp2_task_t;

struct tp2_pool {
  pthread_mutex_t lock;
  pthread_cond_t not_empty;  /* a task is available                          */
  pthread_cond_t not_full;   /* a queue slot became free                     */
  pthread_cond_t all_done;   /* queued == 0 and running == 0                 */

  tp2_task_t *head, *tail;
  long queued;               /* tasks waiting in the queue                   */
  long running;              /* tasks currently being executed by a worker   */
  long peak;                 /* high-water mark of queued                    */
  int capacity;              /* 0 = unbounded                                */
  int shutdown;

  int nthreads;
  pthread_t *threads;
};

static __thread int tp2_tls_id = -1;

int tp2_worker_id(void) { return tp2_tls_id; }

typedef struct {
  tp2_pool_t *pool;
  int id;
} tp2_arg_t;

static void *tp2_worker(void *raw) {
  tp2_arg_t *wa = (tp2_arg_t *)raw;
  tp2_pool_t *p = wa->pool;
  tp2_tls_id = wa->id;
  free(wa);

  for (;;) {
    pthread_mutex_lock(&p->lock);

    while (p->queued == 0 && !p->shutdown)
      pthread_cond_wait(&p->not_empty, &p->lock);

    if (p->queued == 0 && p->shutdown) {
      pthread_mutex_unlock(&p->lock);
      break;
    }

    tp2_task_t *t = p->head;
    p->head = t->next;
    if (p->head == NULL) p->tail = NULL;
    p->queued--;
    p->running++;   /* the task has left the queue but is NOT finished yet */

    /* One slot has just become free, so at most one blocked producer can
     * proceed: signal, not broadcast. */
    pthread_cond_signal(&p->not_full);
    pthread_mutex_unlock(&p->lock);

    t->fn(t->arg);
    free(t);

    pthread_mutex_lock(&p->lock);
    p->running--;
    /* The pool is idle only when nothing is queued AND nothing is running.
     * Testing p->queued alone would let tp2_wait() return while the last
     * tasks were still executing. broadcast, because several threads may be
     * waiting in tp2_wait(). */
    if (p->queued == 0 && p->running == 0)
      pthread_cond_broadcast(&p->all_done);
    pthread_mutex_unlock(&p->lock);
  }
  return NULL;
}

tp2_pool_t *tp2_create(int nthreads, int capacity) {
  if (nthreads <= 0) return NULL;
  tp2_pool_t *p = (tp2_pool_t *)calloc(1, sizeof(tp2_pool_t));
  if (!p) return NULL;

  p->nthreads = nthreads;
  p->capacity = capacity > 0 ? capacity : 0;
  p->threads = (pthread_t *)calloc((size_t)nthreads, sizeof(pthread_t));
  if (!p->threads) {
    free(p);
    return NULL;
  }

  pthread_mutex_init(&p->lock, NULL);
  pthread_cond_init(&p->not_empty, NULL);
  pthread_cond_init(&p->not_full, NULL);
  pthread_cond_init(&p->all_done, NULL);

  for (int i = 0; i < nthreads; i++) {
    tp2_arg_t *wa = (tp2_arg_t *)malloc(sizeof(tp2_arg_t));
    wa->pool = p;
    wa->id = i;
    if (pthread_create(&p->threads[i], NULL, tp2_worker, wa) != 0) {
      free(wa);
      p->nthreads = i;
      tp2_destroy(p);
      return NULL;
    }
  }
  return p;
}

int tp2_submit(tp2_pool_t *pool, tp2_task_fn fn, void *arg) {
  if (!pool || !fn) return -1;

  tp2_task_t *t = (tp2_task_t *)malloc(sizeof(tp2_task_t));
  if (!t) return -1;
  t->fn = fn;
  t->arg = arg;
  t->next = NULL;

  pthread_mutex_lock(&pool->lock);

  /* Backpressure: block until a slot is free. WHILE, not IF -- another
   * producer may take the slot between the signal and this thread's wake-up. */
  while (pool->capacity > 0 && pool->queued >= pool->capacity &&
         !pool->shutdown)
    pthread_cond_wait(&pool->not_full, &pool->lock);

  if (pool->shutdown) {
    pthread_mutex_unlock(&pool->lock);
    free(t);
    return -1;
  }

  if (pool->tail) pool->tail->next = t;
  else pool->head = t;
  pool->tail = t;
  pool->queued++;
  if (pool->queued > pool->peak) pool->peak = pool->queued;

  pthread_cond_signal(&pool->not_empty);
  pthread_mutex_unlock(&pool->lock);
  return 0;
}

void tp2_wait(tp2_pool_t *pool) {
  if (!pool) return;
  pthread_mutex_lock(&pool->lock);
  while (pool->queued > 0 || pool->running > 0)
    pthread_cond_wait(&pool->all_done, &pool->lock);
  pthread_mutex_unlock(&pool->lock);
}

void tp2_destroy(tp2_pool_t *pool) {
  if (!pool) return;

  pthread_mutex_lock(&pool->lock);
  pool->shutdown = 1;
  pthread_cond_broadcast(&pool->not_empty);
  pthread_cond_broadcast(&pool->not_full);   /* release blocked producers */
  pthread_mutex_unlock(&pool->lock);

  for (int i = 0; i < pool->nthreads; i++)
    pthread_join(pool->threads[i], NULL);

  pthread_mutex_destroy(&pool->lock);
  pthread_cond_destroy(&pool->not_empty);
  pthread_cond_destroy(&pool->not_full);
  pthread_cond_destroy(&pool->all_done);
  free(pool->threads);
  free(pool);
}

long tp2_peak_queue(const tp2_pool_t *pool) { return pool ? pool->peak : 0; }
