#ifndef LAUNCH_PARAMS_H_
#define LAUNCH_PARAMS_H_

#include "optix8.hpp"
#include "../utils/my_math.hpp"
#include <cuda_runtime.h>

// ray type
enum {
    RADIANCE_RAY_TYPE=0, 
    SHADOW_RAY_TYPE, 
    RAY_TYPE_COUNT
};

// material 
enum {
    MATERIAL_TYPE_DIFFUSE=0, 
    MATERIAL_TYPE_PRINCIPLED_BRDF, 
    MATERIAL_TYPE_GLASS, 
    MATERIAL_TYPE_LIGHT, 
    NUM_MATERIAL_TYPE
};


// camera model
enum {
    LENS_TYPE_PINHOLE=0, 
    LENS_TYPE_THIN_LENS, 
    NUM_LENS_TYPE
};

// BRDF, BSDF
enum {
    BXDF_TYPE_DIFFUSE_SAMPLE=0,
    BXDF_TYPE_DIFFUSE_EVAL,
    BXDF_TYPE_PRINCIPLED_SAMPLE,
    BXDF_TYPE_PRINCIPLED_EVAL,
    BXDF_TYPE_GLASS_SAMPLE,
    BXDF_TYPE_GLASS_EVAL,
    NUM_BXDF
};

// light
enum {
    LIGHT_TYPE_ENV_SPHERE=0, 
    LIGHT_TYPE_TRIANGLE, 
    NUM_LIGHT_TYPE
};


struct TextureSlot {
    bool hasTexture             {false};
    cudaTextureObject_t texture;
};


struct TriangleMeshSBTData {
    float3* vertex;
    float3* normal;
    float4* tangent;
    float2* diffuseTexcoord;
    float2* normalTexcoord;
    float2* emissiveTexcoord;
    uint3*  index;

    bool hasTangent;
    bool hasNormal;

    unsigned int instanceID;

    unsigned int    materialType;
    float3  color;
    float   roughness;
    float   metallic;
    float3  emissive;

    TextureSlot diffuseTexture;
    TextureSlot normalTexture;
    TextureSlot rmTexture;
    TextureSlot emissiveTexture;
};

struct LightDefinition{
    int lightType;
    uint32_t lightIndexInType; // 三角形のインデックス or 環境マップのインデックス
};

struct TriangleLightData {
    float3 v0, v1, v2;
    float3 normal;
    float area;
    float3 constantEmission;

    // emissive テクスチャ用
    float2 uv0, uv1, uv2;
    TextureSlot emissiveTexture;
};

struct LaunchParams {
    struct {
        float4* colorBuffer;
        float4* normalBuffer;
        float4* albedoBuffer;

        mymath::matrix3x4* objectMatrixBuffer;
        mymath::matrix3x3* normalMatrixBuffer;

        int2    size            {make_int2(1920, 1080)};
        int     accumID         {0};
        int     numPixelSamples {256};
        int     maxBounce       {16};
        int     frameID         {0};
    } frame;

    struct {
        // Extrinsic
        float3 position;
        float3 direction;
        float3 horizontal;
        float3 vertical;

        // Intrinsics
        float focalLength       {50.0f};
        float fValue            {2.8f};
        float pintDist          {1.0f};
        float sensitivity       {1.0f};
        float fov               {20.f};
        
        // camera mode
        int cameraMode          {LENS_TYPE_PINHOLE};
    } camera;

    struct {
        LightDefinition*    lightDefinition;
        TriangleLightData*  triangleLightData;
        int numLights                   {0};
        float lightIntensityFactor      {1.0f};
    } light;

    struct {
        bool    hasEnvMap       {false};
        float*  coarseMarginal;
        float*  coarseConditional;
        float*  patchWeight;
        float   totalWeight;
        int2    patchSize;
    } envMapInfo;
    
    struct {
        // RIS
        int risSampleNum        {8};
    } samplingStrategy;

    OptixTraversableHandle traversable;
    cudaTextureObject_t envMap;
};

#endif // LAUNCH_PARAMS_H_