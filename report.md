海东哥，我这几天把gemm搞了一下，大概就是跟着那篇论文还有张先轶的一堂课把整个过程学了一遍，感觉讲的真的很细很多问题以前没考虑过，当年排序没什么数据重用和数据预热，这次真是学到了新知识。然后对于接下来的优化主要尝试了下面三种方式（两种都没啥用...）：  
主要优化的就是这个循环最内层的`AddDot4x4()`，这是原kernel，我也不知道这个该咋给你汇报一下子，就用这种思路+代码的方式吧。

```
//Original kernel implemented by SSE
void AddDot4x4( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc )
{
  /* So, this routine computes a 4x4 block of matrix A

           C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ).  
           C( 1, 0 ), C( 1, 1 ), C( 1, 2 ), C( 1, 3 ).  
           C( 2, 0 ), C( 2, 1 ), C( 2, 2 ), C( 2, 3 ).  
           C( 3, 0 ), C( 3, 1 ), C( 3, 2 ), C( 3, 3 ).  

     Notice that this routine is called with c = C( i, j ) in the
     previous routine, so these are actually the elements 

           C( i  , j ), C( i  , j+1 ), C( i  , j+2 ), C( i  , j+3 ) 
           C( i+1, j ), C( i+1, j+1 ), C( i+1, j+2 ), C( i+1, j+3 ) 
           C( i+2, j ), C( i+2, j+1 ), C( i+2, j+2 ), C( i+2, j+3 ) 
           C( i+3, j ), C( i+3, j+1 ), C( i+3, j+2 ), C( i+3, j+3 ) 
	  
     in the original matrix C 

     And now we use vector registers and instructions */

  int p;
  v2df_t
    c_00_c_10_vreg,    c_01_c_11_vreg,    c_02_c_12_vreg,    c_03_c_13_vreg,
    c_20_c_30_vreg,    c_21_c_31_vreg,    c_22_c_32_vreg,    c_23_c_33_vreg,
    a_0p_a_1p_vreg,
    a_2p_a_3p_vreg,
    b_p0_vreg, b_p1_vreg, b_p2_vreg, b_p3_vreg; 

  c_00_c_10_vreg.v = _mm_setzero_pd();   
  c_01_c_11_vreg.v = _mm_setzero_pd();
  c_02_c_12_vreg.v = _mm_setzero_pd(); 
  c_03_c_13_vreg.v = _mm_setzero_pd(); 
  c_20_c_30_vreg.v = _mm_setzero_pd();   
  c_21_c_31_vreg.v = _mm_setzero_pd();  
  c_22_c_32_vreg.v = _mm_setzero_pd();   
  c_23_c_33_vreg.v = _mm_setzero_pd(); 

  for ( p=0; p<k; p++ ){
    a_0p_a_1p_vreg.v = _mm_load_pd( (double *) a );
    a_2p_a_3p_vreg.v = _mm_load_pd( (double *) ( a+2 ) );
    a += 4;

    b_p0_vreg.v = _mm_loaddup_pd( (double *) b );       /* load and duplicate */
    b_p1_vreg.v = _mm_loaddup_pd( (double *) (b+1) );   /* load and duplicate */
    b_p2_vreg.v = _mm_loaddup_pd( (double *) (b+2) );   /* load and duplicate */
    b_p3_vreg.v = _mm_loaddup_pd( (double *) (b+3) );   /* load and duplicate */

    b += 4;

    /* First row and second rows */
    c_00_c_10_vreg.v += a_0p_a_1p_vreg.v * b_p0_vreg.v;
    c_01_c_11_vreg.v += a_0p_a_1p_vreg.v * b_p1_vreg.v;
    c_02_c_12_vreg.v += a_0p_a_1p_vreg.v * b_p2_vreg.v;
    c_03_c_13_vreg.v += a_0p_a_1p_vreg.v * b_p3_vreg.v;

    /* Third and fourth rows */
    c_20_c_30_vreg.v += a_2p_a_3p_vreg.v * b_p0_vreg.v;
    c_21_c_31_vreg.v += a_2p_a_3p_vreg.v * b_p1_vreg.v;
    c_22_c_32_vreg.v += a_2p_a_3p_vreg.v * b_p2_vreg.v;
    c_23_c_33_vreg.v += a_2p_a_3p_vreg.v * b_p3_vreg.v;
  }

  C( 0, 0 ) += c_00_c_10_vreg.d[0];  C( 0, 1 ) += c_01_c_11_vreg.d[0];  
  C( 0, 2 ) += c_02_c_12_vreg.d[0];  C( 0, 3 ) += c_03_c_13_vreg.d[0]; 

  C( 1, 0 ) += c_00_c_10_vreg.d[1];  C( 1, 1 ) += c_01_c_11_vreg.d[1];  
  C( 1, 2 ) += c_02_c_12_vreg.d[1];  C( 1, 3 ) += c_03_c_13_vreg.d[1]; 

  C( 2, 0 ) += c_20_c_30_vreg.d[0];  C( 2, 1 ) += c_21_c_31_vreg.d[0];  
  C( 2, 2 ) += c_22_c_32_vreg.d[0];  C( 2, 3 ) += c_23_c_33_vreg.d[0]; 

  C( 3, 0 ) += c_20_c_30_vreg.d[1];  C( 3, 1 ) += c_21_c_31_vreg.d[1];  
  C( 3, 2 ) += c_22_c_32_vreg.d[1];  C( 3, 3 ) += c_23_c_33_vreg.d[1]; 
}
```

1.在我的笔记本上，调整过分块大小之后，他那个SSE最终版的性能差不多是10 GFlops，然后我先把SSE版改成了AVX版，性能大概整体提升了50%，到了 15～16 GFlops（其实已经到头了）....

```
//Modified kernel implemented by AVX2
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

}

```





2.我把矩阵C的 4 * 4 子矩阵由流水加法的过程改成了向量加法、矩阵转置的过程，发现性能并没有什么提升......

3.把循环展开由4 * 4变为8 * 8，每次计算矩阵C的8 * 8子矩阵，计算过程为向量加法、矩阵转置，性能依旧没什么提升......

2和3就把3的代码贴出来吧，思路是一样的。

```
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
		
	transpose( &( c_00_c_30_vreg.v ), &( c_01_c_31_vreg.v ),
			   &( c_02_c_32_vreg.v ), &( c_03_c_33_vreg.v ) );

	transpose( &( c_40_c_70_vreg.v ), &( c_41_c_71_vreg.v ),
			   &( c_42_c_72_vreg.v ), &( c_43_c_73_vreg.v ) );
	
	transpose( &( c_04_c_34_vreg.v ), &( c_05_c_35_vreg.v ),
			   &( c_06_c_36_vreg.v ), &( c_07_c_37_vreg.v ) );

	transpose( &( c_44_c_74_vreg.v ), &( c_45_c_75_vreg.v ),
			   &( c_46_c_76_vreg.v ), &( c_47_c_77_vreg.v ) );
	
    _mm256_storeu_pd( & C( 0, 0 ) ,  
					_mm256_add_pd( _mm256_load_pd( & C( 0, 0 ) ) , c_00_c_30_vreg.v ) );
    _mm256_storeu_pd( & C( 0, 4 ) ,  
					_mm256_add_pd( _mm256_load_pd( & C( 0, 4 ) ) , c_04_c_34_vreg.v ) );

    _mm256_storeu_pd( & C( 1, 0 ) , 
					_mm256_add_pd( _mm256_load_pd( & C( 1, 0 ) ) , c_01_c_31_vreg.v ) );
    _mm256_storeu_pd( & C( 1, 4 ) , 
					_mm256_add_pd( _mm256_load_pd( & C( 1, 4 ) ) , c_05_c_35_vreg.v ) );
    
	_mm256_storeu_pd( & C( 2, 0 ) , 
					_mm256_add_pd( _mm256_load_pd( & C( 2, 0 ) ) , c_02_c_32_vreg.v ) );
	_mm256_storeu_pd( & C( 2, 4 ) , 
					_mm256_add_pd( _mm256_load_pd( & C( 2, 4 ) ) , c_06_c_36_vreg.v ) );
    
	_mm256_storeu_pd( & C( 3, 0 ) , 
					_mm256_add_pd( _mm256_load_pd( & C( 3, 0 ) ) , c_03_c_33_vreg.v ) );
	_mm256_storeu_pd( & C( 3, 4 ) , 
					_mm256_add_pd( _mm256_load_pd( & C( 3, 4 ) ) , c_07_c_37_vreg.v ) );

	_mm256_storeu_pd( & C( 4, 0 ) ,  
					_mm256_add_pd( _mm256_load_pd( & C( 4, 0 ) ) , c_40_c_70_vreg.v ) );
    _mm256_storeu_pd( & C( 4, 4 ) ,  
					_mm256_add_pd( _mm256_load_pd( & C( 4, 4 ) ) , c_44_c_74_vreg.v ) );

    _mm256_storeu_pd( & C( 5, 0 ) , 
					_mm256_add_pd( _mm256_load_pd( & C( 5, 0 ) ) , c_41_c_71_vreg.v ) );
    _mm256_storeu_pd( & C( 5, 4 ) , 
					_mm256_add_pd( _mm256_load_pd( & C( 5, 4 ) ) , c_45_c_75_vreg.v ) );
    
	_mm256_storeu_pd( & C( 6, 0 ) , 
					_mm256_add_pd( _mm256_load_pd( & C( 6, 0 ) ) , c_42_c_72_vreg.v ) );
	_mm256_storeu_pd( & C( 6, 4 ) , 
					_mm256_add_pd( _mm256_load_pd( & C( 6, 4 ) ) , c_46_c_76_vreg.v ) );
    
	_mm256_storeu_pd( & C( 7, 0 ) , 
					_mm256_add_pd( _mm256_load_pd( & C( 7, 0 ) ) , c_43_c_73_vreg.v ) );
	_mm256_storeu_pd( & C( 7, 4 ) , 
					_mm256_add_pd( _mm256_load_pd( & C( 7, 4 ) ) , c_47_c_77_vreg.v ) );
}
```

我印象里在KNL上面做排序的时候按 8 展开了循环依然是有提升的呀，这个跟pipeline深度还是很有关系的....

我的笔记本和mac mini都是core i5 2.6Ghz的，Haswell架构，大概`DP per second = 8`，那么单核的浮点运算的峰值性能是 `2.6Ghz * 8 = 20.8 GFlops`，那现在差不多能跑到峰值性能的70%～75%？ 

今天上午找高萍讨论了一下，顺便把优化的过程跟她说了一下，她也没啥更好的想法，她也感觉这个论文讲的一些东西她们以前也没考虑过。  

这个从访存和数据局部性的考虑来看，我会的他都做了，我不会的他也做了，我也没什么新的想法，所以关于访存这部分，还有油水可以榨嘛？

然后关于计算的话，做了一些尝试都没什么用，再深入考虑内嵌汇编，重排指令流水有必要嘛？问了高萍，她们也并没有嵌过instrisinc指令集的汇编。

我把这一版的代码放在github上了：
[Github](https://github.com/TianyuZhang1214/Gemm.git)







