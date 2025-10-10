#ifndef BOX_HPP_
#define BOX_HPP_

#include <cuda_runtime.h>
#include "helper_math.h"
#include <iostream>

class Box3f {
public:
    inline __host__ __device__ Box3f& extend(const float3 other)
    {
        m_lower = fminf(m_lower, other);
        m_upper = fmaxf(m_upper, other);
        return *this;
    }

    inline __host__ __device__ float3 getCenter() const 
    {
        if (!isValid()) {
            std::cerr << "Box is invalid before calling getCenter()" << std::endl;
        }
        return (m_lower + m_upper) / 2.0f;
    }

    inline __host__ __device__ float3 getSpan() const
    {
        if (!isValid()) {
            std::cerr << "Box is invalid before calling getSpan()" << std::endl;
        }
        return m_upper - m_lower;
    }

    inline __host__ __device__ float3 getMin() const
    {
        if (!isValid()) {
            std::cerr << "Box is invalid before calling getMin()" << std::endl;
        }
        return m_lower;
    }

protected:

    inline __host__ __device__ bool isValid() const 
    {
        return m_lower.x <= m_upper.x && m_lower.y <= m_upper.y && m_lower.z <= m_upper.z;
    }
    
    float3 m_lower    {make_float3( FLT_MAX,  FLT_MAX,  FLT_MAX)};
    float3 m_upper    {make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX)};
};


#endif // BOX_HPP_