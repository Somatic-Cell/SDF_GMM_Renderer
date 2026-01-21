#ifndef SHADER_COMMON_CUH_
#define SHADER_COMMON_CUH_

#include <cuda_runtime.h>
#include "../../utils/helper_math.h"
#include "../../utils/my_math.hpp"
#define _USE_MATH_DEFINES
#include <math.h>

struct LensRay
{
    float3 org;
    float3 dir;
    float weight    {1.0f};
};

struct IntersectedData
{
    float3 baseColor;
    float ior;
    float roughness;
    float metallic;
    float3 wiLocal;
    float3 normal;
};

struct LightSample
{
    float3  position;
    float3  direction;
    float distance      {1e7f};
    float3  emission    {make_float3(0.0f)};
    float   pdf         {0.0f};
};

inline __device__ __host__ void rotate(const float3 _u, const float r, float3& vx, float3& vy, float3& vz)
{
    float3 u = normalize(_u);
    float s = sin(r);
    float c = cos(r);
    const float3 rx = make_float3(u.x * u.x + (1.f - u.x * u.x) * c,   u.x * u.y * (1.f - c) - u.z * s,    u.x * u.z * (1.f - c) + u.y * s);
    const float3 ry = make_float3(u.x * u.y * (1.f - c) + u.z * s,     u.y * u.y + (1.f - u.y * u.y) * c,  u.y * u.z * (1.f - c) - u.x * s);
    const float3 rz = make_float3(u.x * u.z * (1.f - c) * u.y * s,     u.y * u.z * (1.f - c) + u.x * s,    u.z * u.z + (1.f - u.z * u.z) * c);

    // | rx.x  ry.x  rz.x |   | vx.x  vy.x  vz.x |
    // | rx.y  ry.y  rz.y | * | vx.y  vy.y  vz.y |
    // | rx.z  ry.z  rz.z |   | vx.z  vy.z  vz.z |
    //
    const float3 tmpx = make_float3(rx.x * vx.x + ry.x * vx.y + rz.x * vx.z, rx.x * vy.x + ry.x * vy.y + rz.x * vy.z, rx.x * vz.x + ry.x * vz.y + rz.x * vz.z);
    const float3 tmpy = make_float3(rx.y * vx.x + ry.y * vx.y + rz.y * vx.z, rx.y * vy.x + ry.y * vy.y + rz.y * vy.z, rx.y * vz.x + ry.y * vz.y + rz.y * vz.z);
    const float3 tmpz = make_float3(rx.z * vx.x + ry.z * vx.y + rz.z * vx.z, rx.z * vy.x + ry.z * vy.y + rz.z * vy.z, rx.z * vz.x + ry.z * vz.y + rz.z * vz.z);

    vx = tmpx;
    vy = tmpy;
    vz = tmpz;
}

inline __device__ __host__ float3 random_unit_hemisphere(float rnd1, float rnd2)
{
    float a = rnd1 * 2.f * M_PI;
    float z = rnd2;
    float r = fmaxf(0.0f, sqrtf(1.f - z * z));
    return make_float3(r * cosf(a), z, r * sinf(a));
}

inline __device__ __host__ float3 random_unit_sphere(float rnd1, float rnd2)
{
    float3 p;
    float costheta, phi;
    float sintheta;

    // 球上の一点をサンプリング（極座標表現）
    costheta = 2.f * rnd1 - 1.f; // 0 < theta < pi
    phi = 2.f * M_PI * rnd2;

    sintheta = sqrtf(1.f - costheta * costheta);

    p.x = sintheta * cosf(phi);
    p.y = sintheta * sinf(phi);
    p.z = costheta;

    return p;
}

inline __device__ __host__ float2 random_unit_disk(float rnd1, float rnd2)
{
    float theta = 2.0f * float(M_PI) * rnd1;
    float r = sqrtf(rnd2);
    float2 p = r * make_float2(cosf(theta), sinf(theta));
    return p;
}

inline __device__ __host__ float3 randomCosineHemisphere(float rnd1, float rnd2)
{
    const float r = sqrtf(rnd1);
    const float phi = rnd2 * 2.f * M_PI;
    const float x = r * cosf(phi);
    const float z = r * sinf(phi);
    const float y = fmaxf(0.0f, sqrtf(1.f - x * x - z * z));
    return make_float3(x, y, z);
}

// 基底変換
inline __device__ __host__ float3 localToWorld(const float3 v, const float3 localX, const float3 localY, const float3 localZ) {
    return normalize(v.x * localX + v.y * localY + v.z * localZ);
}

inline __device__ __host__ float3 worldToLocal(const float3 v, const float3 localX, const float3 localY, const float3 localZ) {
    return normalize(make_float3(dot(v, localX), dot(v, localY), dot(v, localZ)));
}

inline __device__ __host__ float wrap01(float x) {
    x -= floorf(x);          // [0,1) に折り返し
    return x;
}

// 球面座標
inline __device__ __host__ void orthogonalToUVCoord(const float3 dir, float* u, float* v) {
    // const float3 dir = normalize(_dir);
    float phi = atan2f(dir.z, dir.x);
    if(phi < 0.0f) phi += 2.0f * M_PI;
    float theta = acosf(fminf(fmaxf(dir.y, -1.0f), 1.0f));
    float uu = phi / (2.f * M_PI);
    *u = wrap01(uu + 0.5f);
    *v = theta / M_PI;
}


inline __device__ __host__ float3 sphericalToOrthogonalCoord(const float theta, const float phi) {
    float sinT = sinf(theta);
    return make_float3(cosf(phi) * sinT, cosf(theta), sinf(phi) * sinT);
}


// wi を返す
// inline __device__ float3 cosineSampling(const float u, const float v)
// {
//     const float theta = 0.5f * acosf(1.0f - 2.0f * u);
//     const float phi = 2.0f * M_PI * v;

//     const float sinTheta    = sinf(theta);
//     const float cosTheta    = cosf(theta);
//     const float sinPhi      = sinf(phi);
//     const float cosPhi      = cosf(phi);
//     return make_float3(sinTheta * cosPhi, cosTheta, sinTheta * sinPhi);
// }

// wm を返す
inline __device__ float3 visibleNormalSampling( const float alpha, 
                                                const float3 wo, 
                                                const float rnd1, 
                                                const float rnd2)
{
    float3 Vh = normalize(make_float3(alpha * wo.x, wo.y, alpha * wo.z));

    float3 normal = make_float3(0.f, 1.f, 0.f);
    if (Vh.y > 0.99f) {
        normal = make_float3(0.f, 0.f, -1.f);
    }

    float3 T1 = normalize(cross(Vh, normal));
    float3 T2 = cross(T1, Vh);

    float r = sqrtf(fmaxf(rnd1, 0.f));
    float phi = 2.f * M_PI * rnd2;
    float t1 = r * cosf(phi);
    float t2 = r * sinf(phi);
    float s = 0.5f * (1.f + Vh.y);
    t2 = (1.f - s) * sqrtf(fmaxf(1.f - t1 * t1, 0.f)) + s * t2;

    float3 Nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(1e-7f, 1.f - t1 * t1 - t2 * t2)) * Vh;
    float3 Ne = normalize(make_float3(alpha * Nh.x, Nh.y, alpha * Nh.z));
    return Ne;
}

inline __device__ float3 evalLambertBRDF(const float3 baseColor, const float3 wo, const float3 wi)
{
    return baseColor / M_PI;
}

inline __device__ float getLambertPdf(const float3 wo, const float3 wi){
    return wi.y / M_PI ;
}

inline __device__ float ggx_D(const float alpha, const float3 wm)
{
    const float t = wm.y * wm.y + (wm.x * wm.x + wm.z * wm.z) / (alpha * alpha);
    return 1.f / (M_PI * alpha * alpha * t * t);
}

inline __device__ float lambda(const float alpha, const float3 w)
{
    float vy = fmaxf(w.y, 1e-5f);
    float t = alpha * alpha * (w.x * w.x + w.z * w.z) / (vy * vy);
    return (-1.f + sqrtf(1.f + t)) / 2.f;
}

inline __device__ float smith_G1(const float alpha, const float3 w)
{
    return 1.0f / (1.0f + lambda(alpha, w));
}

inline __device__ float smith_G2(const float alpha, const float3 wo, const float3 wi)
{
    return 1.0f / (1.0f + lambda(alpha, wo) + lambda(alpha, wi));
}

inline __device__ float3 schlick(const float3 wo, const float3 n, const float3 F0){
    return F0 + (make_float3(1.0f) - F0) * powf(1.f - dot(wo, n), 5);
}

inline __device__ float schlick(const float3 wo, const float3 n, const float F0){
    return F0 + (1.0f - F0) * powf(1.f - fabsf(dot(wo, n)), 5);
}

inline __device__ float3 evalSpecularBRDF( const float alpha, 
                                        const float3 F0, 
                                        const float3& wo, 
                                        const float3& wi)
{
    float3 wm = normalize(wo + wi);

    float D = ggx_D(alpha, wm);
    float G = smith_G2(alpha, wo, wi);
    float3 F = schlick(wo, wm, F0);

    float in = fmaxf(wi.y, 1e-4f);
    float on = fmaxf(wo.y, 1e-4f);

    float3 brdf = F * G * D / (4.f * in * on + 1e-5f);

    return brdf;
}

inline __device__ float getGGXPdf(const float alpha, const float3 wo, const float3 wi){
    float3 wm = normalize(wo + wi);
    return ggx_D(alpha, wm) * smith_G1(alpha, wo) * fmaxf(dot(wo, wm), 1e-7f) / (fmaxf(wo.y, 1e-7f) * 4.0f * fmaxf(dot(wi, wm), 1e-7f));
}

__forceinline__ __device__ float balanceHeuristicWeight(const int num_a, const float pdf_a, const int num_b, const float pdf_b){
    float pa = (pdf_a > 0.f) ? pdf_a : 0.0f;
    float pb = (pdf_b > 0.f) ? pdf_b : 0.0f;

    const float np_a = (float)num_a * pa;
    const float np_b = (float)num_b * pb;
    const float s = np_a + np_b;
    return (s > 0.f) ? np_a / s : 0.0f;
}

static __forceinline__ __device__ int lowerBoundCDF(const float* __restrict__ cdf, int n, float u)
{
    int low = 0, high = n - 1;
    while (low < high) {
        int mid = (low + high) >> 1;
        if (u <= cdf[mid])  high = mid;
        else                low = mid + 1;
    }
    return low;
}

static __forceinline__ __device__ void thetaPhiFromPatch(
    const int row, 
    const int col,
    const int wp,
    const int hp,
    float* theta0,
    float* theta1,
    float* phi0,
    float* phi1
){
    float u0 = float(col)       / float(wp);
    float u1 = float(col + 1)   / float(wp);
    float v0 = float(row)       / float(hp);
    float v1 = float(row + 1)   / float(hp);
    
    *phi0 = 2.0f * M_PI * u0;
    *phi1 = 2.0f * M_PI * u1;
    *theta0 = M_PI * v0;
    *theta1 = M_PI * v1;
}

static __forceinline__ __device__ float2 envUVFromSpherical(
    const float theta,
    const float phi
)
{
    float u = phi / (2.0f *  M_PI);
    float v = theta / M_PI;
    u = u - floorf(u);
    v = fminf(fmaxf(v, 0.0f), 1.0f);
    return make_float2(u, v);
}

static __forceinline__ __device__ uint32_t hash_u32(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}


static __forceinline__ __device__ float u32_to_unit_float(uint32_t x)
{
    // 下位24bitを使って [0,1) に正規化（24bit精度）
    return (x & 0x00FFFFFFu) * (1.0f / 16777216.0f); // 2^24
}

static __forceinline__ __device__ float3 hsv2rgb(float h, float s, float v)
{
    float c = v * s;
    float hp = h * 6.0f;
    float x = c * (1.0f - fabsf(fmodf(hp, 2.0f) - 1.0f));
    float3 rgb;
    if      (hp < 1) rgb = make_float3(c, x, 0);
    else if (hp < 2) rgb = make_float3(x, c, 0);
    else if (hp < 3) rgb = make_float3(0, c, x);
    else if (hp < 4) rgb = make_float3(0, x, c);
    else if (hp < 5) rgb = make_float3(x, 0, c);
    else             rgb = make_float3(c, 0, x);
    float m = v - c;
    return rgb + make_float3(m);
}

static __forceinline__ __device__ float3 instanceIdToRGB(uint32_t instanceID, uint32_t seed = 0u)
{
    // seed を変えると配色パターンを変えられます
    uint32_t x = instanceID ^ (seed * 0x9e3779b9u);
    uint32_t h = hash_u32(x);

    float hue = ((h >> 8) & 0x00FFFFFFu) * (1.0f / 16777216.0f);
    // uint32_t h1 = hash_u32(h0 ^ 0xA511E9B3u);
    // uint32_t h2 = hash_u32(h1 ^ 0x63D83595u);

    return hsv2rgb(hue, 0.85f, 0.95f);
}

#endif // SHADER_COMMON_CUH_