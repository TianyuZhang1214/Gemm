#include <immintrin.h>
#include <stdio.h>

static void 
inspect(__m256d v)
{
    double d[4];
    _mm256_storeu_pd(d, v);
    printf( "[%f,%f,%f,%f]\n", d[0], d[1], d[2], d[3] );
}

static void inline
inspect8(__m256d v1, __m256d v2)
{
    double d[8];
    _mm256_storeu_pd( d    , v1 );
    _mm256_storeu_pd( d + 4, v2 );
    printf( "[%f,%f,%f,%f] [%f,%f,%f,%f]\n", 
	d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7] );
}
static void inline
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

    double *d = (double *)malloc(64 * sizeof(double));

    for(int i = 1; i <= 64; ++i) d[i-1] = (double)i;

    __m256d t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf;

    t0 = _mm256_load_pd( d        ); 
    t1 = _mm256_load_pd( d + 4*1  ); 
    t2 = _mm256_load_pd( d + 4*2  ); 
    t3 = _mm256_load_pd( d + 4*3  ); 
    t4 = _mm256_load_pd( d + 4*4  ); 
    t5 = _mm256_load_pd( d + 4*5  ); 
    t6 = _mm256_load_pd( d + 4*6  ); 
    t7 = _mm256_load_pd( d + 4*7  ); 
    t8 = _mm256_load_pd( d + 4*8  ); 
    t9 = _mm256_load_pd( d + 4*9  ); 
    ta = _mm256_load_pd( d + 4*10 ); 
    tb = _mm256_load_pd( d + 4*11 ); 
    tc = _mm256_load_pd( d + 4*12 ); 
    td = _mm256_load_pd( d + 4*13 ); 
    te = _mm256_load_pd( d + 4*14 ); 
    tf = _mm256_load_pd( d + 4*15 );

    inspect8( t0, t1 );
    inspect8( t2, t3 );
    inspect8( t4, t5 );
    inspect8( t6, t7 );
    inspect8( t8, t9 );
    inspect8( ta, tb );
    inspect8( tc, td );
    inspect8( te, tf );
    puts("-----------------------");

	transpose( &t0, &t2, &t4, &t6 );
	transpose( &t1, &t3, &t5, &t7 );
	transpose( &t8, &ta, &tc, &te );
	transpose( &t9, &tb, &td, &tf );
    
	inspect8( t0, t1 );
    inspect8( t2, t3 );
    inspect8( t4, t5 );
    inspect8( t6, t7 );
    inspect8( t8, t9 );
    inspect8( ta, tb );
    inspect8( tc, td );
    inspect8( te, tf );

}
