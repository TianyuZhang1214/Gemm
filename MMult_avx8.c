/* Create macros so that the matrices are stored in column-major order */

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

/* Block sizes */
#define mc 512
#define kc 256
#define nb 1000

#define min( i, j ) ( (i)<(j) ? (i): (j) )

/* Routine for computing C = A * B + C */

void AddDot8x8_avx( int, double *, int, double *, int, double *, int );
void PackMatrixA( int, double *, int, double * );
void PackMatrixB( int, double *, int, double * );
void InnerKernel( int, int, int, double *, int, double *, 
int, double *, int , int );
void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
	int i, p, pb, ib;

/* This time, we compute a mc x n block of C by a call to the InnerKernel */

	double *tmp = (double*)malloc(4*sizeof(double));
	
    for ( p=0; p<k; p+=kc ){
		pb = min( k-p, kc );
		for ( i=0; i<m; i+=mc ){
			ib = min( m-i, mc );
			InnerKernel( ib, n, pb, &A( i,p ), lda, &B(p, 0 ), 
            ldb, &C( i,0 ), ldc, i==0);
		}
	}
}

void InnerKernel( int m, int n, int k, double *a, int lda, 
                                       double *b, int ldb,
                                       double *c, int ldc, int first_time
                                       )
{
	int i, j;
	double 
		packedA[ m * k ];
	static double 
		packedB[ kc*nb ];    /* Note: using a static buffer is not thread safe... */
	
	
    for ( j=0; j<n; j+=8 ){        /* Loop over the columns of C, unrolled by 4 */
		if ( first_time )
			PackMatrixB( k, &B( 0, j ), ldb, &packedB[ j*k ] );
		for ( i=0; i<m; i+=8 ){        /* Loop over the rows of C */
	    /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
	   one routine (four inner products) */
			if ( j == 0 ) 
				PackMatrixA( k, &A( i, 0 ), lda, &packedA[ i*k ] );
			AddDot8x8_avx( k, &packedA[ i*k ], 8, 
			&packedB[ j*k ], k, &C( i,j ), ldc);
		}
	}
}

void PackMatrixA( int k, double *a, int lda, double *a_to )
{
	int j;
	
	for( j=0; j<k; j++){  /* loop over columns of A */
		double 
			*a_ij_pntr = &A( 0, j );
		
		*a_to     = *a_ij_pntr;
		*(a_to+1) = *(a_ij_pntr+1);
		*(a_to+2) = *(a_ij_pntr+2);
		*(a_to+3) = *(a_ij_pntr+3);
		
		a_to += 4;
	}
}

void PackMatrixB( int k, double *b, int ldb, double *b_to )
{
	int i;
	double 
		*b_i0_pntr = &B( 0, 0 ), *b_i1_pntr = &B( 0, 1 ),
		*b_i2_pntr = &B( 0, 2 ), *b_i3_pntr = &B( 0, 3 );
	
	for( i=0; i<k; i++){  /* loop over rows of B */
		*b_to++ = *b_i0_pntr++;
		*b_to++ = *b_i1_pntr++;
		*b_to++ = *b_i2_pntr++;
		*b_to++ = *b_i3_pntr++;
	}
}

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3
#include <immintrin.h>

typedef union
{
  __m128d v;
  double d[2];
} v2df_t;

typedef union
{
  __m256d v;
  double d[4];
} v4df_t;

inspect(__m256d v)
{
    double d[4];
    _mm256_storeu_pd(d, v);
    printf( "[%lf, %lf, %lf, %lf]\n", d[0], d[1], d[2], d[3]);
}


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

void AddDot8x8_avx( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc )
{
	int p;
	v4df_t
		c_00_c_30_vreg,    c_01_c_31_vreg,    c_02_c_32_vreg,    c_03_c_33_vreg,
		c_04_c_34_vreg,    c_05_c_35_vreg,    c_06_c_36_vreg,    c_07_c_37_vreg,

		c_40_c_70_vreg,    c_41_c_71_vreg,    c_42_c_72_vreg,    c_43_c_73_vreg,
		c_44_c_74_vreg,    c_45_c_75_vreg,    c_46_c_76_vreg,    c_47_c_77_vreg,
		
        a_0p_a_3p_vreg,    a_4p_a_7p_vreg,
		b_p0_vreg,         b_p1_vreg,         b_p2_vreg,         b_p3_vreg,
		b_p4_vreg,         b_p5_vreg,         b_p6_vreg,         b_p7_vreg; 
	
	c_00_c_30_vreg.v = _mm256_setzero_pd();
	c_01_c_31_vreg.v = _mm256_setzero_pd();
	c_02_c_32_vreg.v = _mm256_setzero_pd(); 
	c_03_c_33_vreg.v = _mm256_setzero_pd(); 
	c_04_c_34_vreg.v = _mm256_setzero_pd();
	c_05_c_35_vreg.v = _mm256_setzero_pd();
	c_06_c_36_vreg.v = _mm256_setzero_pd(); 
	c_07_c_37_vreg.v = _mm256_setzero_pd(); 
	
    c_40_c_70_vreg.v = _mm256_setzero_pd(); 
    c_41_c_71_vreg.v = _mm256_setzero_pd(); 
    c_42_c_72_vreg.v = _mm256_setzero_pd(); 
    c_43_c_73_vreg.v = _mm256_setzero_pd(); 
    c_44_c_74_vreg.v = _mm256_setzero_pd(); 
    c_45_c_75_vreg.v = _mm256_setzero_pd(); 
    c_46_c_76_vreg.v = _mm256_setzero_pd(); 
    c_47_c_77_vreg.v = _mm256_setzero_pd(); 
	
	for ( p=0; p<k; p++ ){
		a_0p_a_3p_vreg.v = _mm256_load_pd( (double *) a );
		a_4p_a_7p_vreg.v = _mm256_load_pd( (double *) (a+4) );
		
		a += 8;
		
		b_p0_vreg.v = _mm256_set1_pd( *  b    );   /* load and duplicate */
		b_p1_vreg.v = _mm256_set1_pd( * (b+1) );   /* load and duplicate */
		b_p2_vreg.v = _mm256_set1_pd( * (b+2) );   /* load and duplicate */
		b_p3_vreg.v = _mm256_set1_pd( * (b+3) );   /* load and duplicate */
		b_p4_vreg.v = _mm256_set1_pd( * (b+4) );   /* load and duplicate */
		b_p5_vreg.v = _mm256_set1_pd( * (b+5) );   /* load and duplicate */
		b_p6_vreg.v = _mm256_set1_pd( * (b+6) );   /* load and duplicate */
		b_p7_vreg.v = _mm256_set1_pd( * (b+7) );   /* load and duplicate */
		
		b += 8;
		
		c_00_c_30_vreg.v = _mm256_add_pd( c_00_c_30_vreg.v,
						   _mm256_mul_pd( a_0p_a_3p_vreg.v, b_p0_vreg.v));
		c_01_c_31_vreg.v = _mm256_add_pd( c_01_c_31_vreg.v,
						   _mm256_mul_pd( a_0p_a_3p_vreg.v, b_p1_vreg.v));
		c_02_c_32_vreg.v = _mm256_add_pd( c_02_c_32_vreg.v,
						   _mm256_mul_pd( a_0p_a_3p_vreg.v, b_p2_vreg.v));
		c_03_c_33_vreg.v = _mm256_add_pd( c_03_c_33_vreg.v,
		                   _mm256_mul_pd( a_0p_a_3p_vreg.v, b_p3_vreg.v));

        c_04_c_34_vreg.v = _mm256_add_pd( c_04_c_34_vreg.v,
                           _mm256_mul_pd( a_0p_a_3p_vreg.v, b_p4_vreg.v));
        c_05_c_35_vreg.v = _mm256_add_pd( c_05_c_35_vreg.v,
                           _mm256_mul_pd( a_0p_a_3p_vreg.v, b_p5_vreg.v));
        c_06_c_36_vreg.v = _mm256_add_pd( c_06_c_36_vreg.v,
                           _mm256_mul_pd( a_0p_a_3p_vreg.v, b_p6_vreg.v));
        c_07_c_37_vreg.v = _mm256_add_pd( c_07_c_37_vreg.v,
                           _mm256_mul_pd( a_0p_a_3p_vreg.v, b_p7_vreg.v));

        c_40_c_70_vreg.v = _mm256_add_pd( c_40_c_70_vreg.v,
                           _mm256_mul_pd( a_4p_a_7p_vreg.v, b_p0_vreg.v));
        c_41_c_71_vreg.v = _mm256_add_pd( c_41_c_71_vreg.v,
                           _mm256_mul_pd( a_4p_a_7p_vreg.v, b_p1_vreg.v));
        c_42_c_72_vreg.v = _mm256_add_pd( c_42_c_72_vreg.v,
                           _mm256_mul_pd( a_4p_a_7p_vreg.v, b_p2_vreg.v));
        c_43_c_73_vreg.v = _mm256_add_pd( c_43_c_73_vreg.v,
                           _mm256_mul_pd( a_4p_a_7p_vreg.v, b_p3_vreg.v));
        
        c_44_c_74_vreg.v = _mm256_add_pd( c_44_c_74_vreg.v,
                           _mm256_mul_pd( a_4p_a_7p_vreg.v, b_p4_vreg.v));
        c_45_c_75_vreg.v = _mm256_add_pd( c_45_c_75_vreg.v,
                           _mm256_mul_pd( a_4p_a_7p_vreg.v, b_p5_vreg.v));
        c_46_c_76_vreg.v = _mm256_add_pd( c_46_c_76_vreg.v,
                           _mm256_mul_pd( a_4p_a_7p_vreg.v, b_p6_vreg.v));
        c_47_c_77_vreg.v = _mm256_add_pd( c_47_c_77_vreg.v,
                           _mm256_mul_pd( a_4p_a_7p_vreg.v, b_p7_vreg.v));

	}









/*
	C( 0, 0 ) += c_00_c_30_vreg.d[0];  C( 0, 1 ) += c_01_c_31_vreg.d[0];
	C( 0, 2 ) += c_02_c_32_vreg.d[0];  C( 0, 3 ) += c_03_c_33_vreg.d[0];
	C( 0, 4 ) += c_04_c_34_vreg.d[0];  C( 0, 5 ) += c_05_c_35_vreg.d[0];
	C( 0, 6 ) += c_06_c_36_vreg.d[0];  C( 0, 7 ) += c_07_c_37_vreg.d[0];
    
    C( 1, 0 ) += c_00_c_30_vreg.d[1];  C( 1, 1 ) += c_01_c_31_vreg.d[1]; 
	C( 1, 2 ) += c_02_c_32_vreg.d[1];  C( 1, 3 ) += c_03_c_33_vreg.d[1];
    C( 1, 4 ) += c_04_c_34_vreg.d[1];  C( 1, 5 ) += c_05_c_35_vreg.d[1]; 
	C( 1, 6 ) += c_06_c_36_vreg.d[1];  C( 1, 7 ) += c_07_c_37_vreg.d[1];

    C( 2, 0 ) += c_00_c_30_vreg.d[2];  C( 2, 1 ) += c_01_c_31_vreg.d[2]; 
	C( 2, 2 ) += c_02_c_32_vreg.d[2];  C( 2, 3 ) += c_03_c_33_vreg.d[2];
    C( 2, 4 ) += c_04_c_34_vreg.d[2];  C( 2, 5 ) += c_05_c_35_vreg.d[2]; 
	C( 2, 6 ) += c_06_c_36_vreg.d[2];  C( 2, 7 ) += c_07_c_37_vreg.d[2];

    C( 3, 0 ) += c_00_c_30_vreg.d[3];  C( 3, 1 ) += c_01_c_31_vreg.d[3]; 
	C( 3, 2 ) += c_02_c_32_vreg.d[3];  C( 3, 3 ) += c_03_c_33_vreg.d[3];
    C( 3, 4 ) += c_04_c_34_vreg.d[3];  C( 3, 5 ) += c_05_c_35_vreg.d[3]; 
	C( 3, 6 ) += c_06_c_36_vreg.d[3];  C( 3, 7 ) += c_07_c_37_vreg.d[3];

	C( 4, 0 ) += c_40_c_70_vreg.d[0];  C( 4, 1 ) += c_41_c_71_vreg.d[0];
	C( 4, 2 ) += c_42_c_72_vreg.d[0];  C( 4, 3 ) += c_43_c_73_vreg.d[0];
	C( 4, 4 ) += c_44_c_74_vreg.d[0];  C( 4, 5 ) += c_45_c_75_vreg.d[0];
	C( 4, 6 ) += c_46_c_76_vreg.d[0];  C( 4, 7 ) += c_47_c_77_vreg.d[0];
    
    C( 5, 0 ) += c_40_c_70_vreg.d[1];  C( 5, 1 ) += c_41_c_71_vreg.d[1]; 
	C( 5, 2 ) += c_42_c_72_vreg.d[1];  C( 5, 3 ) += c_43_c_73_vreg.d[1];
    C( 5, 4 ) += c_44_c_74_vreg.d[1];  C( 5, 5 ) += c_45_c_75_vreg.d[1]; 
	C( 5, 6 ) += c_46_c_76_vreg.d[1];  C( 5, 7 ) += c_47_c_77_vreg.d[1];

    C( 6, 0 ) += c_40_c_70_vreg.d[2];  C( 6, 1 ) += c_41_c_71_vreg.d[2]; 
	C( 6, 2 ) += c_42_c_72_vreg.d[2];  C( 6, 3 ) += c_43_c_73_vreg.d[2];
    C( 6, 4 ) += c_44_c_74_vreg.d[2];  C( 6, 5 ) += c_45_c_75_vreg.d[2]; 
	C( 6, 6 ) += c_46_c_76_vreg.d[2];  C( 6, 7 ) += c_47_c_77_vreg.d[2];

    C( 7, 0 ) += c_40_c_70_vreg.d[3];  C( 7, 1 ) += c_41_c_71_vreg.d[3]; 
	C( 7, 2 ) += c_42_c_72_vreg.d[3];  C( 7, 3 ) += c_43_c_73_vreg.d[3];
    C( 7, 4 ) += c_44_c_74_vreg.d[3];  C( 7, 5 ) += c_45_c_75_vreg.d[3]; 
	C( 7, 6 ) += c_46_c_76_vreg.d[3];  C( 7, 7 ) += c_47_c_77_vreg.d[3];
*/
   
    inspect( c_00_c_30_vreg.v );
    inspect( c_01_c_31_vreg.v );
    inspect( c_02_c_32_vreg.v );
    inspect( c_03_c_33_vreg.v );
    inspect( c_04_c_34_vreg.v );
    inspect( c_05_c_35_vreg.v );
    inspect( c_06_c_36_vreg.v );
    inspect( c_07_c_37_vreg.v );
    inspect( c_40_c_70_vreg.v );
    inspect( c_41_c_71_vreg.v );
    inspect( c_42_c_72_vreg.v );
    inspect( c_43_c_73_vreg.v );
    inspect( c_44_c_74_vreg.v );
    inspect( c_45_c_75_vreg.v );
    inspect( c_46_c_76_vreg.v );
    inspect( c_47_c_77_vreg.v );

    transpose8( &( c_00_c_30_vreg.v ), &( c_40_c_70_vreg.v ),
                &( c_01_c_31_vreg.v ), &( c_41_c_71_vreg.v ),
                &( c_02_c_32_vreg.v ), &( c_42_c_72_vreg.v ),
                &( c_03_c_33_vreg.v ), &( c_43_c_73_vreg.v ),
                &( c_04_c_34_vreg.v ), &( c_44_c_74_vreg.v ),
                &( c_05_c_35_vreg.v ), &( c_45_c_75_vreg.v ),
                &( c_06_c_36_vreg.v ), &( c_46_c_76_vreg.v ),
                &( c_07_c_37_vreg.v ), &( c_47_c_77_vreg.v ) );

    inspect( c_00_c_30_vreg.v );
    inspect( c_01_c_31_vreg.v );
    inspect( c_02_c_32_vreg.v );
    inspect( c_03_c_33_vreg.v );
    inspect( c_04_c_34_vreg.v );
    inspect( c_05_c_35_vreg.v );
    inspect( c_06_c_36_vreg.v );
    inspect( c_07_c_37_vreg.v );
    inspect( c_40_c_70_vreg.v );
    inspect( c_41_c_71_vreg.v );
    inspect( c_42_c_72_vreg.v );
    inspect( c_43_c_73_vreg.v );
    inspect( c_44_c_74_vreg.v );
    inspect( c_45_c_75_vreg.v );
    inspect( c_46_c_76_vreg.v );
    inspect( c_47_c_77_vreg.v );
    exit(1);

    _mm256_storeu_pd( & C( 0, 0 ) ,  
					_mm256_add_pd( _mm256_load_pd( & C( 0, 0 ) ) , c_00_c_30_vreg.v ) );
    _mm256_storeu_pd( & C( 1, 0 ) , 
					_mm256_add_pd( _mm256_load_pd( & C( 1, 0 ) ) , c_01_c_31_vreg.v ) );
    _mm256_storeu_pd( & C( 2, 0 ) , 
					_mm256_add_pd( _mm256_load_pd( & C( 2, 0 ) ) , c_02_c_32_vreg.v ) );
    _mm256_storeu_pd( & C( 3, 0 ) , 
					_mm256_add_pd( _mm256_load_pd( & C( 3, 0 ) ) , c_03_c_33_vreg.v ) );


}

