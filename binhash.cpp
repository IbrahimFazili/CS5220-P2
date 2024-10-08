#include <string.h>

#include "zmorton.hpp"
#include "binhash.hpp"

/*@q
 * ====================================================================
 */

/*@T
 * \subsection{Spatial hashing implementation}
 * 
 * In the current implementation, we assume [[HASH_DIM]] is $2^b$,
 * so that computing a bitwise of an integer with [[HASH_DIM]] extracts
 * the $b$ lowest-order bits.  We could make [[HASH_DIM]] be something
 * other than a power of two, but we would then need to compute an integer
 * modulus or something of that sort.
 * 
 *@c*/

#define HASH_MASK (HASH_DIM-1)

unsigned particle_bucket(particle_t* p, float h)
{
    unsigned ix = p->x[0]/h;
    unsigned iy = p->x[1]/h;
    unsigned iz = p->x[2]/h;

    return zm_encode(ix & HASH_MASK, iy & HASH_MASK, iz & HASH_MASK);
}

unsigned particle_neighborhood(unsigned* buckets, particle_t* p, float h)
{
    /* BEGIN TASK */

    unsigned px = p->x[0]/h;
    unsigned py = p->x[1]/h;
    unsigned pz = p->x[2]/h;

    // make le buckets
    unsigned neighbour_buckets[27]; // if we stick to using h as is
    int index = 0;

    // @NOTE: can we parallelize this?
    // #pragma omp parallel for collapse(3) shared(neighbor_buckets) private(index)
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                // find for any possible direction the bin could have a neighbour in
                unsigned new_px = px + dx;
                unsigned new_py = py + dy;
                unsigned new_pz = pz + dz;

                // #pragma omp critical {
                    int bucket_hash = zm_encode(new_px & HASH_MASK, new_py & HASH_MASK, new_pz & HASH_MASK);
                    neighbour_buckets[index] = bucket_hash;
                    index++;
                // }
            }
        }
    }

    // now we need to copy it over
    // memcpy(buckets, neighbour_buckets, sizeof(neighbour_buckets));
    memcpy(buckets, neighbour_buckets, index * sizeof(unsigned));
    
    // tells how many were actually found
    return index; 
    /* END TASK */
}

void hash_particles(sim_state_t* s, float h)
{
    /* BEGIN TASK */

    // initialize the table to NULL
    memset(s->hash, 0, HASH_SIZE * sizeof(particle_t*));

    // assign each particle to a bucket
    // my attempt at parallelizing it

    // #pragma omp parallel for
    for (int i = 0; i < s->n; ++i) {
        particle_t* particle = &s->part[i];

        // find the bucket it belongs to
        unsigned bucket = particle_bucket(particle, h);

        // update the pointers
        // #pragma omp critcal 
        // {
            particle->next = s->hash[bucket];
            s->hash[bucket] = particle;
        // }
    }
    /* END TASK */
}
