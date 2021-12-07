/*
   Copyright (c) 2020, James Davis http://people.cs.vt.edu/davisjam/
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "Python.h"

#include "sre.h"
#include "sre_rle.h"
#include "avl_tree.h"

//#include <assert.h>
//#include <stdlib.h>
//#include <stdio.h>

#define BITS_PER_LONG       (sizeof(long) * 8)
#define BIT_WORD(nr)        ((nr) / BITS_PER_LONG)
#define BIT_MASK(nr)        (1UL << ((nr) % BITS_PER_LONG))
#define BITMAP_LAST_WORD_MASK(nbits) (~0UL >> (-(nbits) & (BITS_PER_LONG - 1)))
#define _DIV_ROUND_UP(n, d) (((n) + (d) - 1) / (d))
#define BITS_TO_LONGS(nr)   _DIV_ROUND_UP(nr, BITS_PER_LONG)

static inline void
set_bit(unsigned int nr, volatile unsigned long *addr)
{
    unsigned long mask = BIT_MASK(nr);
    unsigned long *p = ((unsigned long *)addr) + BIT_WORD(nr);

    *p  |= mask;
}

static inline int
test_bit(unsigned int nr, const volatile unsigned long *addr)
{
    return 1UL & (addr[BIT_WORD(nr)] >> (nr & (BITS_PER_LONG-1)));
}

static inline void bitmap_zero(unsigned long *dst, unsigned int nbits)
{
	unsigned int len = BITS_TO_LONGS(nbits) * sizeof(unsigned long);
	memset(dst, 0, len);
}

static inline void bitmap_copy(unsigned long *dst, const unsigned long *src,
                               unsigned int nbits)
{
    unsigned int len = BITS_TO_LONGS(nbits) * sizeof(unsigned long);
    memcpy(dst, src, len);
}

static int bitmap_equal(const unsigned long *bitmap1,
                        const unsigned long *bitmap2, unsigned int bits)
{
    unsigned int k, lim = bits/BITS_PER_LONG;
    for (k = 0; k < lim; ++k)
        if (bitmap1[k] != bitmap2[k])
            return 0;

    if (bits % BITS_PER_LONG)
        if ((bitmap1[k] ^ bitmap2[k]) & BITMAP_LAST_WORD_MASK(bits))
            return 0;

    return 1;
}

/* Offset within a k-bit run. [0, bitsPerRun). */
#define RUN_OFFSET(ix, bitsPerRun) ( (ix) % (bitsPerRun) )
/* Run number within a repeating sequence of k-bit runs. [0, nRuns). */
#define RUN_NUMBER(ix, rleStart, bitsPerRun) ( ( (ix) - (rleStart) ) / (bitsPerRun) )

/* Internal API: RLENode */
typedef struct RLENode RLENode;

static void _RLEVector_validate(RLEVector *vec);
static void _RLEVector_addRun(RLEVector *vec, RLENode *node);
static void _RLEVector_removeRun(RLEVector *vec, RLENode *node);

/* RLE Run -- Implemented as an element of an AVL tree. */
struct RLENode
{
    int offset; /* Key */
    int nRuns;

    /* A bit representation of the run sequence.
     * We look at bits 0, 1, 2, 3, ... (right-to-left).  */
    unsigned long *run;
    int nBitsInRun; /* How many bits to look at */

    struct avl_tree_node node;
};

static int
RLENode_nBits(RLENode *node)
{
    return node->nRuns * node->nBitsInRun;
}

/* First index not captured in this run */
static int
RLENode_end(RLENode *node)
{
    return node->offset + RLENode_nBits(node);
}

static int
RLENode_contains(RLENode *node, int ix)
{
    return node->offset <= ix && ix < RLENode_end(node);
}

static int
RLENode_canMerge(RLENode *l, RLENode *r)
{
    assert(l->nBitsInRun == r->nBitsInRun);
    return bitmap_equal(l->run, r->run, r->nBitsInRun) &&
           RLENode_end(l) == r->offset;
}

/* Returns 0 if target's offset lies within curr. */
int
RLENode_avl_tree_cmp(const struct avl_tree_node *target, const struct avl_tree_node *curr)
{
    RLENode *_target = avl_tree_entry(target, RLENode, node);
    RLENode *_curr = avl_tree_entry(curr, RLENode, node);

    // printf("tree_cmp: target %d curr (%d, %d)\n", _target->offset, _curr->offset, _curr->nRuns);
    if (_target->offset < _curr->offset) {
        return -1;
    } else if (RLENode_contains(_curr, _target->offset)) {
        /* _target falls within _curr */
        return 0;
    } else {
        /* _target falls after _curr */
        return 1;
    }
}

static RLENode *
RLENode_create(int offset, int nRuns, unsigned long *run, int nBitsInRun, int copy_run)
{
    RLENode *node = PyObject_Malloc(sizeof *node);
    node->offset = offset;
    node->nRuns = nRuns;
    node->nBitsInRun = nBitsInRun;

    if (!copy_run)
        node->run = run;
    else {
        node->run = PyObject_Malloc(
            BITS_TO_LONGS(nBitsInRun) * sizeof(unsigned long));
        bitmap_copy(node->run, run, nBitsInRun);
    }

    return node;
}

static void
RLENode_destroy(RLENode *node)
{
    PyObject_Free(node->run);
    PyObject_Free(node);
}

/* External API: RLEVector */

struct RLEVector
{
    struct avl_tree_node *root;
    int currNEntries;
    int mostNEntries; /* High water mark */
    int nBitsInRun; /* Length of the runs we encode */
    int autoValidate; /* Validate after every API usage. This can be wildly expensive. */
};

RLEVector *
RLEVector_create(int runLength, int autoValidate)
{
    RLEVector *vec = PyObject_Malloc(sizeof *vec);
    vec->root = NULL;
    vec->currNEntries = 0;
    vec->mostNEntries = 0;
    vec->nBitsInRun = runLength;

    vec->autoValidate = autoValidate;

    logMsg(LOG_VERBOSE, "RLEVector_create: vec %p nBitsInRun %d, autoValidate %d", vec, vec->nBitsInRun, vec->autoValidate);

    return vec;
}

/* Performs a full walk of the tree looking for fishy business. O(n) steps. */
static void
_RLEVector_validate(RLEVector *vec)
{
    RLENode *prev = NULL, *curr = NULL;
    int nNodes = 0;

    assert(vec != NULL);
    logMsg(LOG_DEBUG, "  _RLEVector_validate: Validating vec %p (size %d, runs of length %d)", vec, vec->currNEntries, vec->nBitsInRun);

    if (vec->currNEntries == 0) {
        return;
    }

    prev = avl_tree_entry(avl_tree_first_in_order(vec->root), RLENode, node);
    if (prev != NULL) {
        curr = avl_tree_entry(avl_tree_next_in_order(&prev->node), RLENode, node);

        if (prev != NULL) {
            nNodes++;
        }

        while (prev != NULL && curr != NULL) {
            logMsg(LOG_DEBUG, "rleVector_validate: prev (%d,%d,%lu) curr (%d,%d,%lu)", prev->offset, prev->nRuns, prev->run[0], curr->offset, curr->nRuns, curr->run[0]);
            assert(prev->offset < curr->offset); /* In-order */
            if (RLENode_end(prev) == curr->offset) {
                assert(prev->nBitsInRun == curr->nBitsInRun);
                assert(vec->nBitsInRun == curr->nBitsInRun);
                if (bitmap_equal(prev->run, curr->run, curr->nBitsInRun)){
                    /* Adjacent identical runs should have been merged */
                    assert(!"rleVector_validate: Adjacent identical runs are not merged");
                }
            }
            prev = curr;
            curr = avl_tree_entry(avl_tree_next_in_order(&curr->node), RLENode, node);
            nNodes++;
        }
    }

    logMsg(LOG_DEBUG, "rleVector_validate: nNodes %d currNEntries %d", nNodes, vec->currNEntries);
    assert(vec->currNEntries == nNodes);
}

typedef struct RLENodeNeighbors RLENodeNeighbors;
struct RLENodeNeighbors
{
    RLENode *a; /* Predecessor -- first before */
    RLENode *b; /* Current, if it exists */
    RLENode *c; /* Successor -- first after */
};

static RLENodeNeighbors
RLEVector_getNeighbors(RLEVector *vec, int ix)
{
    RLENode target, *node;
    RLENodeNeighbors rnn;
    rnn.a = NULL;
    rnn.b = NULL;
    rnn.c = NULL;

    target.offset = ix;
    target.nRuns = -1;

    node = avl_tree_entry(avl_tree_lookup_node_pred(vec->root, &target.node, RLENode_avl_tree_cmp), RLENode, node);
    if (node != NULL) {
        /* node is the largest run beginning <= ix */
        if (RLENode_contains(node, ix)) {
            /* = */
            rnn.b = node;
            rnn.a = avl_tree_entry(avl_tree_prev_in_order(&rnn.b->node), RLENode, node);
            rnn.c = avl_tree_entry(avl_tree_next_in_order(&rnn.b->node), RLENode, node);
        } else {
            /* < */
            rnn.a = node;
            rnn.b = NULL;
            rnn.c = avl_tree_entry(avl_tree_next_in_order(&rnn.a->node), RLENode, node);
        }
    } else {
        /* There is no run <= ix */
        rnn.a = NULL;
        rnn.b = NULL;
        rnn.c = avl_tree_entry(avl_tree_first_in_order(vec->root), RLENode, node);
    }

    logMsg(LOG_DEBUG, "rnn: a %p b %p c %p\n", rnn.a, rnn.b, rnn.c);
    return rnn;
}

/* Given a populated RNN, merge a-b and b-c if possible. */
static void
_RLEVector_mergeNeighbors(RLEVector *vec, RLENodeNeighbors rnn)
{
    logMsg(LOG_DEBUG, "mergeNeighbors: begins");

    int nBefore = vec->currNEntries;

    /* Because rnn are adjacent, we can directly manipulate offsets without
     * breaking the BST property. */
    if (rnn.a != NULL && rnn.b != NULL && RLENode_canMerge(rnn.a, rnn.b)) {
        _RLEVector_removeRun(vec, rnn.b);

        rnn.a->nRuns += rnn.b->nRuns;

        logMsg(LOG_DEBUG, "merge: Removed (%d,%d), merged with now-(%d,%d,%lu)", rnn.b->offset, rnn.b->nRuns, rnn.a->offset, rnn.a->nRuns, rnn.a->run[0]);
        RLENode_destroy(rnn.b);

        /* Set b to a, so that the next logic will work. */
        rnn.b = rnn.a;
    }
    if (rnn.b != NULL && rnn.c != NULL && RLENode_canMerge(rnn.b, rnn.c)) {
        _RLEVector_removeRun(vec, rnn.c);

        rnn.b->nRuns += rnn.c->nRuns;
        logMsg(LOG_DEBUG, "merge: Removed (%d,%d), merged with now-(%d,%d,%lu)", rnn.c->offset, rnn.c->nRuns, rnn.b->offset, rnn.b->nRuns, rnn.b->run[0]);

        RLENode_destroy(rnn.c);
    }

    logMsg(LOG_DEBUG, "mergeNeighbors: before %d after %d", nBefore, vec->currNEntries);

    if (vec->autoValidate)
        _RLEVector_validate(vec);
}

int
RLEVector_runSize(RLEVector *vec)
{
    return vec->nBitsInRun;
}

/* Set the bit at ix.
 * Invariant: always returns with vec fully merged; validate() should pass.
 */
void
RLEVector_set(RLEVector *vec, int ix)
{
    RLENodeNeighbors rnn;
    RLENode *newRun = NULL;
    unsigned long *oldRunKernel, *newRunKernel;
    int roundedIx = ix - RUN_OFFSET(ix, vec->nBitsInRun);

    logMsg(LOG_VERBOSE, "RLEVector_set: %d", ix);

    if (vec->autoValidate)
        _RLEVector_validate(vec);

    assert(RLEVector_get(vec, ix) == 0); /* Shouldn't be set already */

    newRunKernel = PyObject_Malloc(
            BITS_TO_LONGS(vec->nBitsInRun) * sizeof(unsigned long));

    rnn = RLEVector_getNeighbors(vec, ix);

    /* Handle the "new" and "split" cases.
     * Update rnn.{a,b,c} as we go. */
    if (rnn.b == NULL) {
        /* Case: creates a run */
        logMsg(LOG_DEBUG, "%d: Creating a run", ix);

        bitmap_zero(newRunKernel, vec->nBitsInRun);
        set_bit(RUN_OFFSET(ix, vec->nBitsInRun), newRunKernel);
        newRun = RLENode_create(roundedIx, 1, newRunKernel, vec->nBitsInRun, 0);

        _RLEVector_addRun(vec, newRun);
        rnn.b = newRun;
    } else {
        /* Case: splits a run */
        RLENode *prefixRun = NULL, *oldRun = NULL, *suffixRun = NULL;
        int ixRunNumber = 0, nRunsInPrefix = 0, nRunsInSuffix = 0;

        logMsg(LOG_DEBUG, "%d: Splitting the run (%d,%d,%lu)", ix, rnn.b->offset, rnn.b->nRuns, rnn.b->run[0]);

        /* Calculate the run kernels */
        oldRun = rnn.b;
        oldRunKernel = oldRun->run;
        bitmap_copy(newRunKernel, oldRunKernel, vec->nBitsInRun);
        set_bit(RUN_OFFSET(ix, vec->nBitsInRun), newRunKernel);

        /* Remove the affected run */
        _RLEVector_removeRun(vec, oldRun);

        /* Insert the new run */
        logMsg(LOG_DEBUG, "adding intercalary run");
        newRun = RLENode_create(roundedIx, 1, newRunKernel, vec->nBitsInRun, 0);
        _RLEVector_addRun(vec, newRun);
        rnn.b = newRun;

        /* Insert prefix and suffix */
        ixRunNumber = RUN_NUMBER(ix, oldRun->offset, oldRun->nBitsInRun);
        nRunsInPrefix = ixRunNumber;
        nRunsInSuffix = oldRun->nRuns - (ixRunNumber + 1);

        if (nRunsInPrefix > 0) {
            logMsg(LOG_DEBUG, "adding prefix");
            prefixRun = RLENode_create(oldRun->offset, nRunsInPrefix,
                                       oldRunKernel, vec->nBitsInRun, 1);
            _RLEVector_addRun(vec, prefixRun);

            rnn.a = prefixRun;
        }
        if (nRunsInSuffix > 0) {
            logMsg(LOG_DEBUG, "adding suffix");
            suffixRun = RLENode_create(roundedIx + vec->nBitsInRun, nRunsInSuffix,
                                       oldRunKernel, vec->nBitsInRun, 1);
            _RLEVector_addRun(vec, suffixRun);

            rnn.c = suffixRun;
        }

        /* Clean up */
        RLENode_destroy(oldRun);
    }

    logMsg(LOG_DEBUG, "Before merge: run is (%d,%d,%lu)", rnn.b->offset, rnn.b->nRuns, rnn.b->run[0]);

    _RLEVector_mergeNeighbors(vec, rnn);
    /* After merging, rnn.{a,b,c} is untrustworthy. */

    if (vec->autoValidate)
        _RLEVector_validate(vec);

    return;
}

int
RLEVector_get(RLEVector *vec, int ix)
{
    RLENode target;
    RLENode *match = NULL;

    logMsg(LOG_DEBUG, "RLEVector_get: %d", ix);

    if (vec->autoValidate)
        _RLEVector_validate(vec);

    target.offset = ix;
    target.nRuns = -1;
    match = avl_tree_entry(
            avl_tree_lookup_node(vec->root, &target.node, RLENode_avl_tree_cmp),
            RLENode, node);

    if (match == NULL) {
        return 0;
    }

    logMsg(LOG_DEBUG, "match: %p", match);
    return test_bit(RUN_OFFSET(ix, match->nBitsInRun), match->run);
}

int
RLEVector_currSize(RLEVector *vec)
{
    return vec->currNEntries;
}

int
RLEVector_maxObservedSize(RLEVector *vec)
{
    return vec->mostNEntries;
}

int
RLEVector_maxBytes(RLEVector *vec)
{
    return sizeof(RLEVector) /* Internal overhead */ \
        + (sizeof(RLENode) + BITS_TO_LONGS(vec->nBitsInRun) * sizeof(unsigned long)) *
          RLEVector_maxObservedSize(vec) /* Cost per node */ \
        ;
}

void
RLEVector_destroy(RLEVector *vec)
{
    RLENode *node = NULL;
    avl_tree_for_each_in_postorder(node, vec->root, RLENode, node)
        RLENode_destroy(node);
    PyObject_Free(vec);

    return;
}

static void _RLEVector_addRun(RLEVector *vec, RLENode *node)
{
    logMsg(LOG_DEBUG, "Adding run (%d,%d,%lu)", node->offset, node->nRuns, node->run[0]);

    struct avl_tree_node *insert =
        avl_tree_insert(&vec->root, &node->node, RLENode_avl_tree_cmp);
    assert(insert == NULL);
    (void)insert; // make release build happy

    vec->currNEntries++;
    if (vec->mostNEntries < vec->currNEntries) {
        vec->mostNEntries = vec->currNEntries;
    }
}

static void _RLEVector_removeRun(RLEVector *vec, RLENode *node)
{
    logMsg(LOG_DEBUG, "Removing run (%d,%d,%lu)", node->offset, node->nRuns, node->run[0]);

    avl_tree_remove(&vec->root, &node->node);
    vec->currNEntries--;

    if (vec->autoValidate)
        _RLEVector_validate(vec);
}
