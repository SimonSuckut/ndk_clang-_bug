#include <arm_neon.h>

class  __attribute__((packed, aligned(16))) Matrix {

public:
    // store the matrix in column major order to allow efficient multiplications
    // This is the same format as glsl expects
	float32x4x4_t smid_data;


    inline Matrix() {
        float32_t __attribute__((aligned(32))) input[16] = { 1.0F, 0.0F, 0.0F, 0.0F,
                                                             0.0F, 1.0F, 0.0F, 0.0F,
                                                             0.0F, 0.0F, 1.0F, 0.0F,
                                                             0.0F, 0.0F, 0.0F, 1.0F };
		this->smid_data = vld1q_f32_x4(input);
    }
    
    inline Matrix(float _m11, float _m12, float _m13, float _m14,
           float _m21, float _m22, float _m23, float _m24,
           float _m31, float _m32, float _m33, float _m34,
           float _m41, float _m42, float _m43, float _m44) {
        float32_t __attribute__((aligned(32))) input[16] = { _m11, _m21, _m31, _m41,
                                                            _m12, _m22, _m32, _m42,
                                                            _m13, _m23, _m33, _m43,
                                                            _m14, _m24, _m34, _m44 };
		this->smid_data = vld1q_f32_x4(input);
    }
};

inline Matrix operator * (const Matrix &m1, const Matrix &m2) {

    Matrix result;

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
