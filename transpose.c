#include <immintrin.h>
#include <stdio.h>

static void 
inspect(__m256d v)
{
    double d[4];
    _mm256_storeu_pd(d, v);
    printf( "[%f,%f,%f,%f]\n", d[0], d[1], d[2], d[3] );
}


//inline void
//transpose(__m256d *v0, __m256d *v1, __m256d *v2, __m256d *v3){
//
//    __m256d t0, t1, t2, t3;
//    
//    t0 = _mm256_unpacklo_pd( *v0 , *v1 );
//    t1 = _mm256_unpackhi_pd( *v0 , *v1 );
//    t2 = _mm256_unpacklo_pd( *v2 , *v3 );
//    t3 = _mm256_unpackhi_pd( *v2 , *v3 );
//    
//    *v0 = _mm256_permute2f128_pd( t0, t2, 0x20);
//    *v1 = _mm256_permute2f128_pd( t1, t3, 0x20);
//    *v2 = _mm256_permute2f128_pd( t0, t2, 0x31);
//    *v3 = _mm256_permute2f128_pd( t1, t3, 0x31);
//
//}

inline void
transpose8(__m256d *r0, __m256d *r1, __m256d *r2, __m256d *r3, __m256d *r4, __m256d *r5, __m256d *r6, __m256d *r7,__m256d *r8, __m256d *r9, __m256d *ra, __m256d *rb, __m256d *rc, __m256d *rd, __m256d *re, __m256d *rf){
    
    __m256d t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf;

    t0 = _mm256_unpacklo_pd( *r0 , *r1 );
    t1 = _mm256_unpackhi_pd( *r0 , *r1 );
    t2 = _mm256_unpacklo_pd( *r2 , *r3 );
    t3 = _mm256_unpackhi_pd( *r2 , *r3 );
    t4 = _mm256_unpacklo_pd( *r4 , *r5 );
    t5 = _mm256_unpackhi_pd( *r4 , *r5 );
    t6 = _mm256_unpacklo_pd( *r6 , *r7 );
    t7 = _mm256_unpackhi_pd( *r6 , *r7 );
    t8 = _mm256_unpacklo_pd( *r8 , *r9 );
    t9 = _mm256_unpackhi_pd( *r8 , *r9 );
    ta = _mm256_unpacklo_pd( *ra , *rb );
    tb = _mm256_unpackhi_pd( *ra , *rb );
    tc = _mm256_unpacklo_pd( *rc , *rd );
    td = _mm256_unpackhi_pd( *rc , *rd );
    te = _mm256_unpacklo_pd( *re , *rf );
    tf = _mm256_unpackhi_pd( *re , *rf );

    *r0 = _mm256_permute2f128_pd( t0, t2, 0x20);
    *r1 = _mm256_permute2f128_pd( t1, t3, 0x20);
    *r2 = _mm256_permute2f128_pd( t0, t2, 0x31);
    *r3 = _mm256_permute2f128_pd( t1, t3, 0x31);
    *r4 = _mm256_permute2f128_pd( t4, t6, 0x20);
    *r5 = _mm256_permute2f128_pd( t5, t7, 0x20);
    *r6 = _mm256_permute2f128_pd( t4, t6, 0x31);
    *r7 = _mm256_permute2f128_pd( t5, t7, 0x31);
    *r8 = _mm256_permute2f128_pd( t8, ta, 0x20);
    *r9 = _mm256_permute2f128_pd( t9, tb, 0x20);
    *ra = _mm256_permute2f128_pd( t8, ta, 0x31);
    *rb = _mm256_permute2f128_pd( t9, tb, 0x31);
    *rc = _mm256_permute2f128_pd( tc, te, 0x20);
    *rd = _mm256_permute2f128_pd( tc, te, 0x20);
    *re = _mm256_permute2f128_pd( td, tf, 0x31);
    *rf = _mm256_permute2f128_pd( td, tf, 0x31);
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

    inspect( t0 );
    inspect( t1 );
    inspect( t2 );
    inspect( t3 );
    inspect( t4 );
    inspect( t5 );
    inspect( t6 );
    inspect( t7 );
    inspect( t8 );
    inspect( t9 );
    inspect( ta );
    inspect( tb );
    inspect( tc );
    inspect( td );
    inspect( te );
    inspect( tf );
    puts("-------------------------------------------------------------------");
    transpose8(&t0, &t1, &t2, &t3, &t4, &t5, &t6, &t7, &t8, &t9, &ta, &tb, &tc, &td, &te, &tf);
}
