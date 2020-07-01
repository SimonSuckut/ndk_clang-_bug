#include <arm_neon.h>

class  __attribute__((packed, aligned(16))) Matrix {

public:
    // store the matrix in column major order to allow efficient multiplications
    // This is the same format as glsl expects
    union{
        struct{
            const float m11;
            const float m21;
            const float m31;
            const float m41;

            const float m12;
            const float m22;
            const float m32;
            const float m42;

            const float m13;
            const float m23;
            const float m33;
            const float m43;

            const float m14;
            const float m24;
            const float m34;
            const float m44;
        };
        const float m[4][4];
        #if defined(__arm__) || defined(__aarch64__)
        float32x4x4_t smid_data;
        #elif defined(__i386__) || defined(__amd64__)
        __m128 smid_data[4];
        #endif
    };

    inline Matrix(): m11(1.0F), m12(0.0F), m13(0.0F), m14(0.0F),
              m21(0.0F), m22(1.0F), m23(0.0F), m24(0.0F),
              m31(0.0F), m32(0.0F), m33(1.0F), m34(0.0F),
              m41(0.0F), m42(0.0F), m43(0.0F), m44(1.0F) {
        float32_t __attribute__((aligned(32))) input[16] = { 1.0F, 0.0F, 0.0F, 0.0F,
                                                             0.0F, 1.0F, 0.0F, 0.0F,
                                                             0.0F, 0.0F, 1.0F, 0.0F,
                                                             0.0F, 0.0F, 0.0F, 1.0F };
        #if defined(__arm__) || defined(__aarch64__)
            this->smid_data = vld1q_f32_x4(input);
        #elif defined(__i386__) || defined(__amd64__)
            this->smid_data[0] = _mm_load_ps(&input[0]);
            this->smid_data[1] = _mm_load_ps(&input[4]);
            this->smid_data[2] = _mm_load_ps(&input[8]);
            this->smid_data[3] = _mm_load_ps(&input[12]);
        #endif
    }
    
    inline Matrix(float _m11, float _m12, float _m13, float _m14,
           float _m21, float _m22, float _m23, float _m24,
           float _m31, float _m32, float _m33, float _m34,
           float _m41, float _m42, float _m43, float _m44)
           : m11(1.0F), m12(0.0F), m13(0.0F), m14(0.0F),
             m21(0.0F), m22(1.0F), m23(0.0F), m24(0.0F),
             m31(0.0F), m32(0.0F), m33(1.0F), m34(0.0F),
             m41(0.0F), m42(0.0F), m43(0.0F), m44(1.0F) {
        float32_t __attribute__((aligned(32))) input[16] = { _m11, _m21, _m31, _m41,
                                                            _m12, _m22, _m32, _m42,
                                                            _m13, _m23, _m33, _m43,
                                                            _m14, _m24, _m34, _m44 };
        #if defined(__arm__) || defined(__aarch64__)
            this->smid_data = vld1q_f32_x4(input);
        #elif defined(__i386__) || defined(__amd64__)
            this->smid_data[0] = _mm_load_ps(&input[0]);
            this->smid_data[1] = _mm_load_ps(&input[4]);
            this->smid_data[2] = _mm_load_ps(&input[8]);
            this->smid_data[3] = _mm_load_ps(&input[12]);
        #endif

    }
};

inline Matrix operator * (const Matrix &m1, const Matrix &m2) {

    Matrix result;

#if defined(__arm__)
    result.smid_data.val[0] = vmulq_lane_f32(m1.smid_data.val[0], vget_low_f32(m2.smid_data.val[0]), 0);
    result.smid_data.val[0] = vmlaq_lane_f32(result.smid_data.val[0], m1.smid_data.val[1], vget_low_f32(m2.smid_data.val[0]), 1);
    result.smid_data.val[0] = vmlaq_lane_f32(result.smid_data.val[0], m1.smid_data.val[2], vget_high_f32(m2.smid_data.val[0]), 0);
    result.smid_data.val[0] = vmlaq_lane_f32(result.smid_data.val[0], m1.smid_data.val[3], vget_high_f32(m2.smid_data.val[0]), 2);

    result.smid_data.val[1] = vmulq_lane_f32(m1.smid_data.val[0], vget_low_f32(m2.smid_data.val[1]), 0);
    result.smid_data.val[1] = vmlaq_lane_f32(result.smid_data.val[1], m1.smid_data.val[1], vget_low_f32(m2.smid_data.val[1]), 1);
    result.smid_data.val[1] = vmlaq_lane_f32(result.smid_data.val[1], m1.smid_data.val[2], vget_high_f32(m2.smid_data.val[1]), 0);
    result.smid_data.val[1] = vmlaq_lane_f32(result.smid_data.val[1], m1.smid_data.val[3], vget_high_f32(m2.smid_data.val[1]), 1);

    result.smid_data.val[2] = vmulq_lane_f32(m1.smid_data.val[0], vget_low_f32(m2.smid_data.val[2]), 0);
    result.smid_data.val[2] = vmlaq_lane_f32(result.smid_data.val[2], m1.smid_data.val[1], vget_low_f32(m2.smid_data.val[2]), 1);
    result.smid_data.val[2] = vmlaq_lane_f32(result.smid_data.val[2], m1.smid_data.val[2], vget_high_f32(m2.smid_data.val[2]), 0);
    result.smid_data.val[2] = vmlaq_lane_f32(result.smid_data.val[2], m1.smid_data.val[3], vget_high_f32(m2.smid_data.val[2]), 1);

    result.smid_data.val[3] = vmulq_lane_f32(m1.smid_data.val[0], vget_low_f32(m2.smid_data.val[3]), 0);
    result.smid_data.val[3] = vmlaq_lane_f32(result.smid_data.val[3], m1.smid_data.val[1], vget_low_f32(m2.smid_data.val[3]), 1);
    result.smid_data.val[3] = vmlaq_lane_f32(result.smid_data.val[3], m1.smid_data.val[2], vget_high_f32(m2.smid_data.val[3]), 0);
    result.smid_data.val[3] = vmlaq_lane_f32(result.smid_data.val[3], m1.smid_data.val[3], vget_high_f32(m2.smid_data.val[3]), 1);


#elif defined(__aarch64__)
    result.smid_data.val[0] = vmulq_laneq_f32(                         m1.smid_data.val[0], m2.smid_data.val[0], 0);
    result.smid_data.val[0] = vmlaq_laneq_f32(result.smid_data.val[0], m1.smid_data.val[1], m2.smid_data.val[0], 1);
    result.smid_data.val[0] = vmlaq_laneq_f32(result.smid_data.val[0], m1.smid_data.val[2], m2.smid_data.val[0], 2);
    result.smid_data.val[0] = vmlaq_laneq_f32(result.smid_data.val[0], m1.smid_data.val[3], m2.smid_data.val[0], 3);

    result.smid_data.val[1] = vmulq_laneq_f32(                         m1.smid_data.val[0], m2.smid_data.val[1], 0);
    result.smid_data.val[1] = vmlaq_laneq_f32(result.smid_data.val[1], m1.smid_data.val[1], m2.smid_data.val[1], 1);
    result.smid_data.val[1] = vmlaq_laneq_f32(result.smid_data.val[1], m1.smid_data.val[2], m2.smid_data.val[1], 2);
    result.smid_data.val[1] = vmlaq_laneq_f32(result.smid_data.val[1], m1.smid_data.val[3], m2.smid_data.val[1], 3);

    result.smid_data.val[2] = vmulq_laneq_f32(                         m1.smid_data.val[0], m2.smid_data.val[2], 0);
    result.smid_data.val[2] = vmlaq_laneq_f32(result.smid_data.val[2], m1.smid_data.val[1], m2.smid_data.val[2], 1);
    result.smid_data.val[2] = vmlaq_laneq_f32(result.smid_data.val[2], m1.smid_data.val[2], m2.smid_data.val[2], 2);
    result.smid_data.val[2] = vmlaq_laneq_f32(result.smid_data.val[2], m1.smid_data.val[3], m2.smid_data.val[2], 3);

    result.smid_data.val[3] = vmulq_laneq_f32(                         m1.smid_data.val[0], m2.smid_data.val[3], 0);
    result.smid_data.val[3] = vmlaq_laneq_f32(result.smid_data.val[3], m1.smid_data.val[1], m2.smid_data.val[3], 1);
    result.smid_data.val[3] = vmlaq_laneq_f32(result.smid_data.val[3], m1.smid_data.val[2], m2.smid_data.val[3], 2);
    result.smid_data.val[3] = vmlaq_laneq_f32(result.smid_data.val[3], m1.smid_data.val[3], m2.smid_data.val[3], 3);
    
#elif defined(__i386__) || defined(__amd64__)

    result.smid_data[0] = _mm_mul_ps(                                m1.smid_data[0], _mm_set1_ps(m2.m[0][0]));
    result.smid_data[0] = _mm_add_ps(result.smid_data[0], _mm_mul_ps(m1.smid_data[1], _mm_set1_ps(m2.m[0][1])));
    result.smid_data[0] = _mm_add_ps(result.smid_data[0], _mm_mul_ps(m1.smid_data[2], _mm_set1_ps(m2.m[0][2])));
    result.smid_data[0] = _mm_add_ps(result.smid_data[0], _mm_mul_ps(m1.smid_data[3], _mm_set1_ps(m2.m[0][3])));

    result.smid_data[1] = _mm_mul_ps(                                m1.smid_data[0], _mm_set1_ps(m2.m[1][0]));
    result.smid_data[1] = _mm_add_ps(result.smid_data[1], _mm_mul_ps(m1.smid_data[1], _mm_set1_ps(m2.m[1][1])));
    result.smid_data[1] = _mm_add_ps(result.smid_data[1], _mm_mul_ps(m1.smid_data[2], _mm_set1_ps(m2.m[1][2])));
    result.smid_data[1] = _mm_add_ps(result.smid_data[1], _mm_mul_ps(m1.smid_data[3], _mm_set1_ps(m2.m[1][3])));

    result.smid_data[2] = _mm_mul_ps(                                m1.smid_data[0], _mm_set1_ps(m2.m[2][0]));
    result.smid_data[2] = _mm_add_ps(result.smid_data[2], _mm_mul_ps(m1.smid_data[1], _mm_set1_ps(m2.m[2][1])));
    result.smid_data[2] = _mm_add_ps(result.smid_data[2], _mm_mul_ps(m1.smid_data[2], _mm_set1_ps(m2.m[2][2])));
    result.smid_data[2] = _mm_add_ps(result.smid_data[2], _mm_mul_ps(m1.smid_data[3], _mm_set1_ps(m2.m[2][3])));

    result.smid_data[3] = _mm_mul_ps(                                m1.smid_data[0], _mm_set1_ps(m2.m[3][0]));
    result.smid_data[3] = _mm_add_ps(result.smid_data[3], _mm_mul_ps(m1.smid_data[1], _mm_set1_ps(m2.m[3][1])));
    result.smid_data[3] = _mm_add_ps(result.smid_data[3], _mm_mul_ps(m1.smid_data[2], _mm_set1_ps(m2.m[3][2])));
    result.smid_data[3] = _mm_add_ps(result.smid_data[3], _mm_mul_ps(m1.smid_data[3], _mm_set1_ps(m2.m[3][3])));

#else

    #pragma GCC error "None ASM implementation for Matrix x Matrix used!"
    result = Matrix((m1.m11 * m2.m11) + (m1.m12 * m2.m21) + (m1.m13 * m2.m31) + (m1.m14 * m2.m41),
                    (m1.m11 * m2.m12) + (m1.m12 * m2.m22) + (m1.m13 * m2.m32) + (m1.m14 * m2.m42),
                    (m1.m11 * m2.m13) + (m1.m12 * m2.m23) + (m1.m13 * m2.m33) + (m1.m14 * m2.m43),
                    (m1.m11 * m2.m14) + (m1.m12 * m2.m24) + (m1.m13 * m2.m34) + (m1.m14 * m2.m44),

                    (m1.m21 * m2.m11) + (m1.m22 * m2.m21) + (m1.m23 * m2.m31) + (m1.m24 * m2.m41),
                    (m1.m21 * m2.m12) + (m1.m22 * m2.m22) + (m1.m23 * m2.m32) + (m1.m24 * m2.m42),
                    (m1.m21 * m2.m13) + (m1.m22 * m2.m23) + (m1.m23 * m2.m33) + (m1.m24 * m2.m43),
                    (m1.m21 * m2.m14) + (m1.m22 * m2.m24) + (m1.m23 * m2.m34) + (m1.m24 * m2.m44),

                    (m1.m31 * m2.m11) + (m1.m32 * m2.m21) + (m1.m33 * m2.m31) + (m1.m34 * m2.m41),
                    (m1.m31 * m2.m12) + (m1.m32 * m2.m22) + (m1.m33 * m2.m32) + (m1.m34 * m2.m42),
                    (m1.m31 * m2.m13) + (m1.m32 * m2.m23) + (m1.m33 * m2.m33) + (m1.m34 * m2.m43),
                    (m1.m31 * m2.m14) + (m1.m32 * m2.m24) + (m1.m33 * m2.m34) + (m1.m34 * m2.m44),

                    (m1.m41 * m2.m11) + (m1.m42 * m2.m21) + (m1.m43 * m2.m31) + (m1.m44 * m2.m41),
                    (m1.m41 * m2.m12) + (m1.m42 * m2.m22) + (m1.m43 * m2.m32) + (m1.m44 * m2.m42),
                    (m1.m41 * m2.m13) + (m1.m42 * m2.m23) + (m1.m43 * m2.m33) + (m1.m44 * m2.m43),
                    (m1.m41 * m2.m14) + (m1.m42 * m2.m24) + (m1.m43 * m2.m34) + (m1.m44 * m2.m44) );
#endif

    return result;
};



static Matrix m = 	Matrix(  1.0F, 0.0F, 0.0F, 0.0F,
							 0.0F, 1.0F, 0.0F, 0.0F,
							 0.0F, 0.0F, 1.0F, 0.0F,
							 0.0F, 0.0F, 1.0F, 0.0F) *
						Matrix( 1.0F, 0.0F, 0.0F, 0.0F,
							 0.0F, 1.0F, 0.0F, 0.0F,
							 0.0F, 0.0F, 1.0F, 1.0F,
							 0.0F, 0.0F, 0.0F, 1.0F);
