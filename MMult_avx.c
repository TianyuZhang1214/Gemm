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

void AddDot4x4( int, double *, int, double *, int, double *, int );
void AddDot4x4_avx( int, double *, int, double *, int, double *, int );
void PackMatrixA( int, double *, int, double * );
void PackMatrixB( int, double *, int, double * );
void InnerKernel( int, int, int, double *, int, double *, 
int, double *, int , int );
static inline void duplicate4(double *, double *);

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
	
	
    for ( j=0; j<n; j+=4 ){        /* Loop over the columns of C, unrolled by 4 */
		if ( first_time )
			PackMatrixB( k, &B( 0, j ), ldb, &packedB[ j*k ] );
		for ( i=0; i<m; i+=4 ){        /* Loop over the rows of C */
	    /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
	   one routine (four inner products) */
			if ( j == 0 ) 
				PackMatrixA( k, &A( i, 0 ), lda, &packedA[ i*k ] );
//			AddDot4x4( k, &packedA[ i*k ], 4, &packedB[ j*k ], k, &C( i,j ), ldc );
			AddDot4x4_avx( k, &packedA[ i*k ], 4, 
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

inspect(__m128d v)
{
    double d[2];
    _mm_storeu_pd(d, v);
    printf( "[%lf, %lf]\n", d[0], d[1] );
}

inspect256(__m256d v)
{
    double d[4];
    _mm256_storeu_pd(d, v);
    printf( "[%lf, %lf, %lf, %lf]\n", d[0], d[1], d[2], d[3]);
}

inline void
transpose4(__m256d *v0, __m256d *v1, __m256d *v2, __m256d *v3){

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


void AddDot4x4_avx( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc 
)
{
	int p;
	v4df_t
		c_00_c_30_vreg,    c_01_c_31_vreg,    c_02_c_32_vreg,    c_03_c_33_vreg,
		a_0p_a_3p_vreg,
		b_p0_vreg, b_p1_vreg, b_p2_vreg, b_p3_vreg; 

	
	c_00_c_30_vreg.v = _mm256_setzero_pd();   
	c_01_c_31_vreg.v = _mm256_setzero_pd();
	c_02_c_32_vreg.v = _mm256_setzero_pd(); 
	c_03_c_33_vreg.v = _mm256_setzero_pd(); 
	
	for ( p=0; p<k; p++ ){
		a_0p_a_3p_vreg.v = _mm256_load_pd( (double *) a );
		
		a += 4;
		
		b_p0_vreg.v = _mm256_set1_pd( * b );       /* load and duplicate */
		b_p1_vreg.v = _mm256_set1_pd( * (b+1) );   /* load and duplicate */
		b_p2_vreg.v = _mm256_set1_pd( * (b+2) );   /* load and duplicate */
		b_p3_vreg.v = _mm256_set1_pd( * (b+3) );   /* load and duplicate */
		
		b += 4;
		
		c_00_c_30_vreg.v =  _mm256_add_pd( c_00_c_30_vreg.v,
						    _mm256_mul_pd( a_0p_a_3p_vreg.v, b_p0_vreg.v));
		c_01_c_31_vreg.v =  _mm256_add_pd( c_01_c_31_vreg.v,
						    _mm256_mul_pd( a_0p_a_3p_vreg.v, b_p1_vreg.v));
		c_02_c_32_vreg.v =  _mm256_add_pd( c_02_c_32_vreg.v,
						    _mm256_mul_pd( a_0p_a_3p_vreg.v, b_p2_vreg.v));
		c_03_c_33_vreg.v =  _mm256_add_pd( c_03_c_33_vreg.v,
		                    _mm256_mul_pd( a_0p_a_3p_vreg.v, b_p3_vreg.v));
	}


	C( 0, 0 ) += c_00_c_30_vreg.d[0];  C( 0, 1 ) += c_01_c_31_vreg.d[0];  
	C( 0, 2 ) += c_02_c_32_vreg.d[0];  C( 0, 3 ) += c_03_c_33_vreg.d[0]; 
	C( 1, 0 ) += c_00_c_30_vreg.d[1];  C( 1, 1 ) += c_01_c_31_vreg.d[1];  
	C( 1, 2 ) += c_02_c_32_vreg.d[1];  C( 1, 3 ) += c_03_c_33_vreg.d[1]; 
	C( 2, 0 ) += c_00_c_30_vreg.d[2];  C( 2, 1 ) += c_01_c_31_vreg.d[2];  
	C( 2, 2 ) += c_02_c_32_vreg.d[2];  C( 2, 3 ) += c_03_c_33_vreg.d[2]; 
	C( 3, 0 ) += c_00_c_30_vreg.d[3];  C( 3, 1 ) += c_01_c_31_vreg.d[3];  
	C( 3, 2 ) += c_02_c_32_vreg.d[3];  C( 3, 3 ) += c_03_c_33_vreg.d[3]; 

/*    Transpose seems doesn't help anymore      */

//    transpose4( &( c_00_c_30_vreg.v ), &( c_01_c_31_vreg.v ), 
//				&( c_02_c_32_vreg.v ), &( c_03_c_33_vreg.v ) );
//
//	
//    _mm256_storeu_pd( & C( 0, 0 ) ,  
//					_mm256_add_pd( _mm256_load_pd( & C( 0, 0 ) ) , c_00_c_30_vreg.v ) );
//    _mm256_storeu_pd( & C( 1, 0 ) , 
//					_mm256_add_pd( _mm256_load_pd( & C( 1, 0 ) ) , c_01_c_31_vreg.v ) );
//    _mm256_storeu_pd( & C( 2, 0 ) , 
//					_mm256_add_pd( _mm256_load_pd( & C( 2, 0 ) ) , c_02_c_32_vreg.v ) );
//    _mm256_storeu_pd( & C( 3, 0 ) , 
//					_mm256_add_pd( _mm256_load_pd( & C( 3, 0 ) ) , c_03_c_33_vreg.v ) );
}

