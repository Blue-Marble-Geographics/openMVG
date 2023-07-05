/** @file imopv_sse2.c
 ** @brief Vectorized image operations - SSE2 - Definition
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "../../../P2PUtils.h"

#if ! defined(VL_DISABLE_SSE2) & ! defined(__SSE2__)
#error "Compiling with SSE2 enabled, but no __SSE2__ defined"
#endif

#if ! defined(VL_DISABLE_SSE2)

#ifndef VL_IMOPV_SSE2_INSTANTIATING

#include <emmintrin.h>
#include "imopv.h"
#include "imopv_sse2.h"

#define FLT VL_TYPE_FLOAT
#define VL_IMOPV_SSE2_INSTANTIATING
#include "imopv_sse2.c"

#if 0
#define FLT VL_TYPE_DOUBLE
#define VL_IMOPV_SSE2_INSTANTIATING
#include "imopv_sse2.c"
#endif

/* ---------------------------------------------------------------- */
/* VL_IMOPV_SSE2_INSTANTIATING */
#else

#include "float.th"

/* Perform MULT SSE2 fetches simultaneously. */
#if (VSIZE==4)
#define MULT (4)
#elif (VSIZE==8)
#define MULT (2)
#else
#pragma error "Unsupported"
#endif

/* ---------------------------------------------------------------- */
void
VL_XCAT3(_vl_imconvcol_v, SFX, _sse2)
(T* dst, vl_size dst_stride,
 T const* src,
 vl_size src_width, vl_size src_height, vl_size src_stride,
 T const* filt, vl_index filt_begin, vl_index filt_end,
 int step, unsigned int flags)
{
  vl_index x = 0 ;
  vl_index y ;
  vl_index dheight = (src_height - 1) / step + 1 ;
  vl_bool use_simd  = VALIGNED(src_stride) ;
  vl_bool transp    = flags & VL_TRANSPOSE ;
  vl_bool zeropad   = (flags & VL_PAD_MASK) == VL_PAD_BY_ZERO ;
  double totcol = 0 ;
  double simdcol = 0 ;

  if (zeropad || (!transp)) {
      return; // "Unsupported.";
  }

  /* let filt point to the last sample of the filter */
  filt += filt_end - filt_begin ;

  while (x < (signed)src_width) {
    /* Calculate dest[x,y] = sum_p image[x,p] filt[y - p]
     * where supp(filt) = [filt_begin, filt_end] = [fb,fe].
     *
     * CHUNK_A: y - fe <= p < 0
     *          completes VL_MAX(fe - y, 0) samples
     * CHUNK_B: VL_MAX(y - fe, 0) <= p < VL_MIN(y - fb, height - 1)
     *          completes fe - VL_MAX(fb, height - y) + 1 samples
     * CHUNK_C: completes all samples
     */

    T const *filti ;
    vl_index stop ;

    /* Always allow SIMD processing. */
    if (((x + VSIZE*MULT) < (signed)src_width) 
       /*& VALIGNED(src + x) & use_simd*/)
    {
      /* ----------------------------------------------  Vectorized */
      for (y = 0 ; y < (signed)src_height ; y += step)  {
        union {VTYPE v ; T x [VSIZE] ; } acc[MULT];
        VTYPE v[MULT], c ;
        VTYPE v2[MULT], c2;
        VTYPE tmp[MULT];
        VTYPE tmp2[MULT];
        T const *srci ;
        /* srci2 is always srci + 1 */

        for (int i = 0; i != MULT; ++i) {
          acc[i].v = v[i] = VSTZ() ;
        }

        filti = filt ;
        stop = filt_end - y ;
        srci = src + x - stop * src_stride ;

        if (stop > 0) {
          for (int i = 0; i != MULT; ++i) {
            v[i] =_Load(src + x + VSIZE*i) ;
          }
          }
          while (filti > filt - stop) {
            c = VLD1 (filti--) ;
          for (int i = 0; i != MULT; ++i) {
            tmp[i] = VMUL(v[i], c);
          }
          for (int i = 0; i != MULT; ++i) {
            acc[i].v = VADD (acc[i].v, tmp[i]) ;
          }
            srci += src_stride ;
          }

        stop = filt_end - VL_MAX(filt_begin, y - (signed)src_height + 1) + 1 ;

        /* Rework loop (and below) to reduce add latency. */
        while (filti > filt - stop) {
          for (int i = 0; i != MULT; ++i) {
            v[i] =_Load(srci + VSIZE*i) ;
          }
          c = VLD1 (filti--) ;
          for (int i = 0; i != MULT; ++i) {
            tmp[i] = VMUL(v[i], c);
          }
          for (int i = 0; i != MULT; ++i) {
            acc[i].v = VADD (acc[i].v,  tmp[i]) ;
          }
          srci += src_stride ;
        }

        stop = filt_end - filt_begin + 1;
        while (filti > filt - stop) {
          c = VLD1 (filti--) ;
            for (int i = 0; i != MULT; ++i) {
              tmp[i] = VMUL(v[i], c);
            }
            for (int i = 0; i != MULT; ++i) {
              acc[i].v = VADD (acc[i].v, tmp[i]) ;
            }
        }

        for (int i = 0; i != MULT; ++i) {
          *dst = acc[i].x[0] ; dst += dst_stride ;
          *dst = acc[i].x[1] ; dst += dst_stride ;
#if(VSIZE == 4)
          *dst = acc[i].x[2] ; dst += dst_stride ;
          *dst = acc[i].x[3] ; dst += dst_stride ;
#endif
        }
        dst += 1 * 1 - VSIZE*MULT * dst_stride ;
      } /* next y */

      dst += VSIZE*MULT * dst_stride - dheight * 1 ;
      x       += VSIZE*MULT ;
      simdcol += VSIZE*MULT ;
      totcol  += VSIZE*MULT ;
    } else {
      /* -------------------------------------------------  Vanilla */
      for (y = 0 ; y < (signed)src_height ; y += step) {
        T acc = 0 ;
        T v = 0, c ;
        T const* srci ;

        filti = filt ;
        stop = filt_end - y ;
        srci = src + x - stop * src_stride ;

        if (stop > 0) {
            v = *(src + x) ;
          }
          while (filti > filt - stop) {
            c = *filti-- ;
            acc += v * c ;
            srci += src_stride ;
          }

        stop = filt_end - VL_MAX(filt_begin, y - (signed)src_height + 1) + 1 ;
        while (filti > filt - (signed)stop) {
          v = *srci ;
          c = *filti-- ;
          acc += v * c ;
          srci += src_stride ;
        }

        stop = filt_end - filt_begin + 1 ;
        while (filti > filt - stop) {
          c = *filti-- ;
          acc += v * c ;
        }

          *dst = acc ; dst += 1 ;
      } /* next y */
        dst += 1 * dst_stride - dheight * 1 ;
      x      += 1 ;
      totcol += 1 ;
    } /* next x */
  }
}

/* ---------------------------------------------------------------- */
#if 0
void
VL_XCAT(_vl_imconvcoltri_v, SFX, sse2)
(T* dst, int dst_stride,
 T const* src,
 int src_width, int src_height, int src_stride,
 int filt_size,
 int step, unsigned int flags)
{
  int x = 0 ;
  int y ;
  int dheight = (src_height - 1) / step + 1 ;
  vl_bool use_simd  = ((src_stride & ALIGNSTRIDE) == 0) &&
  (! (flags & VL_NO_SIMD)) ;
  vl_bool transp = flags & VL_TRANSPOSE ;
  vl_bool zeropad = (flags & VL_PAD_MASK) == VL_PAD_BY_ZERO ;

  T * buff = vl_malloc(sizeof(T) * (src_height + filt_size)) ;
#define fa (1.0 / (double) (filt_size + 1))
  T scale = fa*fa*fa*fa ;
  buff += filt_size ;

  while (x < src_width) {
    T const *srci ;

    use_simd = 0 ;
    if ((x + VSIZE < src_width) &
        (((vl_ptrint)(src + x) & ALIGNPTR) == 0) &
        use_simd)
    {

    } else {
      int stridex = transp ? dst_stride : 1 ;
      int stridey = transp ? 1 : dst_stride ;
      srci = src + x + src_stride * (src_height - 1) ;

      /* integrate backward the column */
      buff [src_height - 1] = *srci ;
      for (y = src_height-2 ; y >=  0 ; --y) {
        srci -= src_stride ;
        buff [y] = buff [y+1] + *srci ;
      }
      if (zeropad) {
        for ( ; y >= - filt_size ; --y) {
          buff [y] = buff [y+1] ;
        }
      } else {
        for ( ; y >= - filt_size ; --y) {
          buff [y] = buff[y+1] + *srci ;
        }
      }

      /* compute the filter forward */
      for (y = - filt_size ; y < src_height - filt_size ; ++y) {
        buff [y] = buff [y] - buff [y + filt_size] ;
      }
      if (! zeropad) {
        for (y = src_height - filt_size ; y < src_height ; ++y) {
          buff [y] = buff [y] - buff [src_height-1]  *
          (src_height - filt_size - y) ;
        }
      }

      /* integrate forward the column */
      for (y = - filt_size + 1 ; y < src_height ; ++y) {
        buff [y] += buff [y - 1] ;
      }

      /* compute the filter backward */
      for (y = src_height - 1 ; y >= 0 ; --y) {
        dst [x*stridex + y*stridey]
        = scale * (buff [y] - buff [y - filt_size]) ;
      }
    } /* next y */
    x += 1 ;
  }
  vl_free (buff - filt_size) ;
}
#endif

#undef FLT
#undef VL_IMOPV_SSE2_INSTANTIATING
#endif

/* ! VL_DISABLE_SSE2 */
#endif
