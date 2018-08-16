#include <immintrin.h>
#include <stdio.h>

static void 
inspect(__m256d v)
{
    double d[4];
    _mm256_storeu_pd(d, v);
    printf( "[%f,%f,%f,%f]\n", d[0], d[1], d[2], d[3] );
}


inline void
transpose(__m256d *v0, __m256d *v1, __m256d *v2, __m256d *v3){

    __m256d t0, t1, t2, t3;
    
    t0 = _mm256_unpacklo_pd( *v0 , *v1 );
    t1 = _mm256_unpackhi_pd( *v0 , *v1 );
    t2 = _mm256_unpacklo_pd( *v2 , *v3 );
    t3 = _mm256_unpackhi_pd( *v2 , *v3 );
    
    *v0 = _mm256_permute2f128_pd( t0, t2, 0x20);
    *v1 = _mm256_permute2f128_pd( t1, t3, 0x20);
    *v2 = _mm256_permute2f128_pd( t0, t2, 0x31);
    *v3 = _mm256_permute2f128_pd( t1, t3, 0x31);

}

int main(){

    double *d = (double *)malloc(16 * sizeof(double));

    for(int i = 1; i <= 16; ++i) d[i-1] = (double)i;

    __m256d t0, t1, t2, t3;
    t0 = _mm256_load_pd( d      ); 
    t1 = _mm256_load_pd( d + 4  ); 
    t2 = _mm256_load_pd( d + 8  ); 
    t3 = _mm256_load_pd( d + 12 ); 

    transpose(&t0, &t1, &t2, &t3);
}
