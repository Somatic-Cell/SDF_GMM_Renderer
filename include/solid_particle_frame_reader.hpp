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
        out12[4] =  R(2, 0) * s; out12[5] =  R(2, 2) * s; out12[6]  = -R(2, 1) * s; out12[7]  = inst.pos[2] - 1.5/25.0f;
        out12[8] = -R(1, 0) * s; out12[9] = -R(1, 2) * s; out12[10] =  R(1, 1) * s; out12[11] = 1.0f - inst.pos[1];

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