#pragma once

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mmq {

struct Json {
    enum Type { Null, Bool, Number, String, Array, Object } type = Null;
    bool b = false;
    double num = 0;
    int64_t i64 = 0;
    bool is_int = false;
    std::string str;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;

    const Json * find(const std::string & k) const {
        for (const auto & kv : obj) {
            if (kv.first == k) {
                return &kv.second;
            }
        }
        return nullptr;
    }
    const Json & at(const std::string & k) const {
        const Json * p = find(k);
        if (!p) {
            throw std::runtime_error("missing json key " + k);
        }
        return *p;
    }
    int64_t as_i() const {
        if (type != Number) {
            throw std::runtime_error("json: expected number");
        }
        return is_int ? i64 : static_cast<int64_t>(num);
    }
    double as_f() const {
        if (type != Number) {
            throw std::runtime_error("json: expected number");
        }
        return is_int ? static_cast<double>(i64) : num;
    }
    const std::string & as_s() const {
        if (type != String) {
            throw std::runtime_error("json: expected string");
        }
        return str;
    }
    bool as_b() const {
        if (type != Bool) {
            throw std::runtime_error("json: expected bool");
        }
        return b;
    }
};

inline Json parse_json(const std::string & s) {
    size_t i = 0;
    auto skip = [&]() {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) {
            ++i;
        }
    };
    auto parse = [&](auto & self) -> Json {
        skip();
        if (i >= s.size()) {
            throw std::runtime_error("json: unexpected end");
        }
        Json j;
        const char c = s[i];
        if (c == 'n') {
            if (s.compare(i, 4, "null") != 0) {
                throw std::runtime_error("json: bad null");
            }
            i += 4;
            j.type = Json::Null;
            return j;
        }
        if (c == 't' || c == 'f') {
            if (s.compare(i, 4, "true") == 0) {
                i += 4;
                j.type = Json::Bool;
                j.b = true;
                return j;
            }
            if (s.compare(i, 5, "false") == 0) {
                i += 5;
                j.type = Json::Bool;
                j.b = false;
                return j;
            }
            throw std::runtime_error("json: bad bool");
        }
        if (c == '"') {
            ++i;
            std::string out;
            while (i < s.size()) {
                char ch = s[i++];
                if (ch == '"') {
                    break;
                }
                if (ch == '\\') {
                    if (i >= s.size()) {
                        throw std::runtime_error("json: bad escape");
                    }
                    const char e = s[i++];
                    switch (e) {
                        case '"':
                        case '\\':
                        case '/': out.push_back(e); break;
                        case 'b': out.push_back('\b'); break;
                        case 'f': out.push_back('\f'); break;
                        case 'n': out.push_back('\n'); break;
                        case 'r': out.push_back('\r'); break;
                        case 't': out.push_back('\t'); break;
                        case 'u': {
                            if (i + 4 > s.size()) {
                                throw std::runtime_error("json: bad \\u");
                            }
                            int cp = 0;
                            for (int k = 0; k < 4; ++k) {
                                const char h = s[i++];
                                cp <<= 4;
                                if (h >= '0' && h <= '9') cp += h - '0';
                                else if (h >= 'a' && h <= 'f') cp += h - 'a' + 10;
                                else if (h >= 'A' && h <= 'F') cp += h - 'A' + 10;
                                else throw std::runtime_error("json: bad \\u hex");
                            }
                            if (cp < 0x80) {
                                out.push_back(static_cast<char>(cp));
                            } else if (cp < 0x800) {
                                out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                            } else {
                                out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                                out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                            }
                            break;
                        }
                        default: throw std::runtime_error("json: bad escape");
                    }
                } else {
                    out.push_back(ch);
                }
            }
            j.type = Json::String;
            j.str = std::move(out);
            return j;
        }
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
            const size_t start = i;
            if (s[i] == '-') {
                ++i;
            }
            if (i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i]))) {
                throw std::runtime_error("json: bad number");
            }
            while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
                ++i;
            }
            bool pure = true;
            if (i < s.size() && s[i] == '.') {
                pure = false;
                ++i;
                while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
                    ++i;
                }
            }
            if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
                pure = false;
                ++i;
                if (i < s.size() && (s[i] == '+' || s[i] == '-')) {
                    ++i;
                }
                if (i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i]))) {
                    throw std::runtime_error("json: bad exponent");
                }
                while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
                    ++i;
                }
            }
            const std::string tok = s.substr(start, i - start);
            j.type = Json::Number;
            j.is_int = pure;
            if (pure) {
                j.i64 = std::stoll(tok);
                j.num = static_cast<double>(j.i64);
            } else {
                j.num = std::strtod(tok.c_str(), nullptr);
            }
            return j;
        }
        if (c == '[') {
            ++i;
            j.type = Json::Array;
            skip();
            if (i < s.size() && s[i] == ']') {
                ++i;
                return j;
            }
            while (true) {
                j.arr.push_back(self(self));
                skip();
                if (i >= s.size()) {
                    throw std::runtime_error("json: unclosed array");
                }
                if (s[i] == ',') {
                    ++i;
                    continue;
                }
                if (s[i] == ']') {
                    ++i;
                    break;
                }
                throw std::runtime_error("json: bad array");
            }
            return j;
        }
        if (c == '{') {
            ++i;
            j.type = Json::Object;
            skip();
            if (i < s.size() && s[i] == '}') {
                ++i;
                return j;
            }
            while (true) {
                skip();
                Json key = self(self);
                if (key.type != Json::String) {
                    throw std::runtime_error("json: object key");
                }
                skip();
                if (i >= s.size() || s[i] != ':') {
                    throw std::runtime_error("json: expected colon");
                }
                ++i;
                Json val = self(self);
                j.obj.emplace_back(std::move(key.str), std::move(val));
                skip();
                if (i >= s.size()) {
                    throw std::runtime_error("json: unclosed object");
                }
                if (s[i] == ',') {
                    ++i;
                    continue;
                }
                if (s[i] == '}') {
                    ++i;
                    break;
                }
                throw std::runtime_error("json: bad object");
            }
            return j;
        }
        throw std::runtime_error(std::string("json: unexpected character ") + c);
    };
    Json root = parse(parse);
    skip();
    if (i != s.size()) {
        throw std::runtime_error("json: trailing data");
    }
    return root;
}

}  // namespace mmq
