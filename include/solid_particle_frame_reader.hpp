#ifndef SOLID_PATRICLE_FRAME_READER_HPP_
#define SOLID_PATRICLE_FRAME_READER_HPP_

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <array>

struct ParticleInstance
{
    int meshID = -1;
    std::array<float, 3> pos{};
    std::array<float, 9> R{};
    float scale = 1.0f;
};

static inline void mulAffine3x4(const float A[12], const float B[12], float C[12])
{
    // C = A * B （Bを先に適用してからA）
    // 3x3
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            C[i*4 + j] =
                A[i*4 + 0] * B[0*4 + j] +
                A[i*4 + 1] * B[1*4 + j] +
                A[i*4 + 2] * B[2*4 + j];
        }
    }
    // translation: C.t = A.R * B.t + A.t
    for (int i = 0; i < 3; ++i) {
        C[i*4 + 3] =
            A[i*4 + 0] * B[0*4 + 3] +
            A[i*4 + 1] * B[1*4 + 3] +
            A[i*4 + 2] * B[2*4 + 3] +
            A[i*4 + 3];
    }
}

static inline void makePivotRotationZ_RowMajor3x4(float out[12], float theta, float px, float py, float pz)
{
    const float c = cosf(theta);
    const float s = sinf(theta);

    // R_z
    const float R00 =  c, R01 = -s, R02 = 0.f;
    const float R10 =  s, R11 =  c, R12 = 0.f;
    const float R20 = 0.f, R21 = 0.f, R22 = 1.f;

    // t = p - R p
    const float rpx = R00*px + R01*py + R02*pz;
    const float rpy = R10*px + R11*py + R12*pz;
    const float rpz = R20*px + R21*py + R22*pz;

    const float tx = px - rpx;
    const float ty = py - rpy;
    const float tz = pz - rpz;

    out[0]=R00; out[1]=R01; out[2]=R02; out[3]=tx;
    out[4]=R10; out[5]=R11; out[6]=R12; out[7]=ty;
    out[8]=R20; out[9]=R21; out[10]=R22; out[11]=tz;
}

class SolidParticlesFrameReader
{
public:
    bool loadFromFile(
        std::string& path,
        std::vector<ParticleInstance>& out,
        std::string* err = nullptr
    ) const
    {
        std::ifstream ifs(path);
        if(!ifs){
            if(err) *err = "Falied to open file:" + path;
            return false;
        }

        std::string line;

        // read count line
        int count = -1;
        while(std::getline(ifs, line)){
            const char* s = line.c_str();
            if(tryParseInt(s, count)) break; // 最初に int を検出した行を粒子数とみなす
        }
        if(count < 0) {
            if (err) * err = "Failed to read particle count.";
            return false;
        }

        // skip header line
        // 次の1行は読み捨て
        std::streampos posBeforeHeader = ifs.tellg();
        if(std::getline(ifs, line)){
            // ヘッダでなさそうなら巻き戻す
            const char* s = line.c_str();
            int dummy = 0;
            if(tryParseInt(s, dummy)) {
                ifs.clear();
                ifs.seekg(posBeforeHeader);
            }
        }
        
        // インスタンス分確保
        out.clear();
        out.reserve(static_cast<size_t>(count));

        // particle のインスタンス情報の読み取り
        int lineNo = 0;
        while((int)out.size() < count && std::getline(ifs, line)) {
            ++lineNo;
            const char* s = line.c_str();

            // 空行っぽい行をスキップ
            if(isSkippableLine(s)) continue;

            ParticleInstance inst;
            if(!tryParseInstanceLine(s, inst)){
                if(err) {
                    *err = "Parse error at particle line (after header), lineNo=" + std::to_string(lineNo) + "\nLine: " + line;
                } 
                return false;
            }
            out.push_back(inst);
        }

        if((int)out.size() != count){
            if(err) {
                *err = "Particle count mismatch. expected= " + std::to_string(count)
                + ", got= " + std::to_string(out.size());
            }

            return false;
        }
        return true;
    }

    // ParticleInstance から Optix instance transform matrix を作る関数
    static void makeOptixTransformRowMajor3x4(const ParticleInstance& inst, float out12[12])
    {
        auto R = [&](int i, int j) -> float {return inst.R[j* 3 + i]; };
        const float s = inst.scale;

        out12[0] =  R(0, 0) * s; out12[1] =  R(0, 2) * s; out12[2]  = -R(0, 1) * s; out12[3]  = inst.pos[0];
        out12[4] =  R(2, 0) * s; out12[5] =  R(2, 2) * s; out12[6]  = -R(2, 1) * s; out12[7]  = inst.pos[2] - 1.0/100.0f;
        out12[8] = -R(1, 0) * s; out12[9] = -R(1, 2) * s; out12[10] =  R(1, 1) * s; out12[11] = 1.0f - inst.pos[1];

    }

    static void makeOptixTransformRowMajor3x4_RotateAroundAxisXZ(
        const ParticleInstance& inst,
        float out12[12],
        float theta /* 毎フレーム更新 */
    ){
        // 1) まず base を作る（元の処理）
        float base[12];
        auto R = [&](int i, int j) -> float { return inst.R[j*3 + i]; };
        const float s = inst.scale;

        base[0]  =  R(0,0)*s; base[1]  =  R(0,2)*s; base[2]  = -R(0,1)*s; base[3]  = inst.pos[0];
        base[4]  =  R(2,0)*s; base[5]  =  R(2,2)*s; base[6]  = -R(2,1)*s; base[7]  = inst.pos[2] - 0.0f/100.0f + 0.22f;
        base[8]  = -R(1,0)*s; base[9]  = -R(1,2)*s; base[10] =  R(1,1)*s; base[11] = 1.0f - inst.pos[1];

        // 2) ピボット p（x=0.5, z=0.5）※この (x,z) は「OptiX空間での」x,z です
        //    回転軸がY方向の場合、pyは結果に効きにくいので 0 でOK（必要ならシーンに合わせて）
        const float px = 0.5f;
        const float py = 0.71f;
        const float pz = 0.5f;

        float pivotRot[12];
        makePivotRotationZ_RowMajor3x4(pivotRot, theta, px, py, pz);

        // 3) 合成：M_final = M_pivot * M_base
        mulAffine3x4(pivotRot, base, out12);
    }

private:
    static bool isSkippableLine(const char* s)
    {
        while(*s){
            unsigned char c = static_cast<unsigned char>(*s);
            if(!(std::isspace(c) || *s == ',')) break;
            ++s;
        }
        // コメント
        if(*s == '\0') return true;
        if(*s == '#')  return true;
        return false;
    }

    static void skipSep(const char*& s)
    {
        while(*s){
            unsigned char c = static_cast<unsigned char>(*s);
            if (std::isspace(c) || *s == ',') { ++s; continue; }
            break;
        }
    }

    static bool tryParseInt(const char*& s, int& out)
    {
        skipSep(s);
        if(!*s) return false;

        char* end = nullptr;
        errno = 0;
        long v = std::strtol(s, &end, 10);
        if(end == s) return false;
        if(errno != 0) return false;

        out = static_cast<int>(v);
        s = end;

        return true;
    }

    static bool tryParseFloat(const char*& s, float& out)
    {
        skipSep(s);
        if(!*s) return false;

        char* end = nullptr;
        errno = 0;
        float v = std::strtof(s, &end);
        if(end == s) return false;
        if(errno != 0) return false;

        out = v;
        s = end;
        return true;
    }

    static bool tryParseInstanceLine(const char*& s, ParticleInstance& inst)
    {
        if (!tryParseInt(s, inst.meshID)) return false;

        for (int i = 0; i < 3; ++i){
            if(!tryParseFloat(s, inst.pos[i])) return false;
        }
        for (int i = 0; i < 9; ++i){
            if(!tryParseFloat(s, inst.R[i])) return false;
        }
        if(!tryParseFloat(s, inst.scale)) return false;
        
        return true;
    }
};


#endif //SOLID_PATRICLE_FRAME_READER_HPP_