//
//  main.cpp
//  mlc_llm
//
//  Created by miyako on 2026/01/01.
//

#include "mlc_llm.h"

// A map to store active request contexts
struct RequestContext {
    size_t n;
    std::string accumulated_text;
};

std::mutex map_mutex;
std::map<std::string, RequestContext> active_requests;

static void set_request_context(unsigned int n, std::string& req_id) {
    
    std::lock_guard<std::mutex> lock(map_mutex);
    
    active_requests[req_id] = {n, ""};
}

static void unset_request_context(std::string& req_id) {
    
    std::lock_guard<std::mutex> lock(map_mutex);
    
    auto it = active_requests.find(req_id);
    if (it != active_requests.end()) {
        active_requests.erase(it);
    }
}

static void append_request_context_value(std::string& req_id, std::string& content) {
    
    std::lock_guard<std::mutex> lock(map_mutex);
    
    auto it = active_requests.find(req_id);
    if (it != active_requests.end()) {
        it->second.accumulated_text += content;
    }
}

// The Global Callback
static void StreamCallback(std::string json) {
    
    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errors;
    
    Json::CharReader *reader = builder.newCharReader();
    bool parse = reader->parse(json.c_str(),
                               json.c_str() + json.size(),
                               &root,
                               &errors);
    delete reader;
    
    if(parse)
    {
        if(root.isArray())
        {
            for(Json::Value::const_iterator it = root.begin() ; it != root.end() ; it++)
            {
                if(it->isObject())
                {
                    Json::Value id_node = it->get("id", Json::stringValue);
                    if(id_node.isString())
                    {
                        std::string req_id = id_node.asString();
                        
                        Json::Value choices_node = it->get("choices", Json::arrayValue);
                        if(choices_node.isArray())
                        {
                            for(Json::Value::const_iterator itt = choices_node.begin() ; itt != choices_node.end() ; itt++)
                            {
                                if(itt->isObject())
                                {
                                    std::string finish_reason;
                                    std::string content;
                                    unsigned int n;
                                    
                                    Json::Value finish_reason_node = itt->get("finish_reason", Json::arrayValue);
                                    if(finish_reason_node.isString())
                                    {
                                        finish_reason = finish_reason_node.asString().c_str();
                                    }
                                    
                                    Json::Value index_node = itt->get("index", Json::intValue);
                                    if(index_node.isNumeric())
                                    {
                                        n = index_node.asInt();
                                    }
                                    Json::Value delta_node = itt->get("delta", Json::objectValue);
                                    if(delta_node.isObject())
                                    {
                                        Json::Value content_node = delta_node["content"];
                                        if(content_node.isString())
                                        {
                                            content = content_node.asString();
                                            append_request_context_value(req_id, content);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

namespace fs = std::filesystem;
using namespace tokenizers; // mlc-ai namespace

static // Helper: Read entire file into a string (Blob)
std::string LoadBytesFromFile(const std::string& path) {
    std::ifstream fs(path, std::ios::in | std::ios::binary);
    if (!fs) throw std::runtime_error("Could not open file: " + path);
    
    std::string data((std::istreambuf_iterator<char>(fs)), std::istreambuf_iterator<char>());
    return data;
}

static // Unified Loader
std::unique_ptr<Tokenizer> LoadTokenizer(const std::string& model_path) {
    fs::path path(model_path);
    
    // 1. Check if the path points to a directory or a specific file
    fs::path json_path = path;
    fs::path model_file_path = path;

    if (fs::is_directory(path)) {
        // If user gave a folder, look for standard names
        json_path = path / "tokenizer.json";
        model_file_path = path / "tokenizer.model";
    }

    // 2. Try to load Hugging Face JSON first (preferred for modern models)
    if (fs::exists(json_path) && json_path.extension() == ".json") {
        std::cout << "Loading HF Tokenizer from: " << json_path << std::endl;
        std::string blob = LoadBytesFromFile(json_path.string());
        return Tokenizer::FromBlobJSON(blob);
    }
    
    // 3. Fallback to SentencePiece
    if (fs::exists(model_file_path) && model_file_path.extension() == ".model") {
        std::cout << "Loading SentencePiece from: " << model_file_path << std::endl;
        std::string blob = LoadBytesFromFile(model_file_path.string());
        return Tokenizer::FromBlobSentencePiece(blob);
    }

    return 0;
}

#ifdef WIN32
static std::wstring utf8_to_wstring(const std::string& str) {
    if (str.empty()) return std::wstring();

    // Get required buffer size in characters (including null terminator)
    int size_needed = MultiByteToWideChar(
        CP_UTF8,       // Source is UTF-8
        0,             // Default flags
        str.c_str(),   // Source string
        -1,            // Null-terminated
        nullptr,       // No output buffer yet
        0              // Requesting size
    );

    if (size_needed <= 0) return std::wstring();

    // Allocate buffer
    std::wstring wstr(size_needed, 0);

    // Perform conversion
    MultiByteToWideChar(
        CP_UTF8,
        0,
        str.c_str(),
        -1,
        &wstr[0],
        size_needed
    );

    // Remove the extra null terminator added by MultiByteToWideChar
    if (!wstr.empty() && wstr.back() == '\0') {
        wstr.pop_back();
    }

    return wstr;
}

static std::string wchar_to_utf8(const wchar_t* wstr) {
    if (!wstr) return std::string();
    
    // Get required buffer size in bytes
    int size_needed = WideCharToMultiByte(
                                          CP_UTF8,            // convert to UTF-8
                                          0,                  // default flags
                                          wstr,               // source wide string
                                          -1,                 // null-terminated
                                          nullptr, 0,         // no output buffer yet
                                          nullptr, nullptr
                                          );
    
    if (size_needed <= 0) return std::string();
    
    // Allocate buffer
    std::string utf8str(size_needed, 0);
    
    // Perform conversion
    WideCharToMultiByte(
                        CP_UTF8,
                        0,
                        wstr,
                        -1,
                        &utf8str[0],
                        size_needed,
                        nullptr,
                        nullptr
                        );
    
    // Remove the extra null terminator added by WideCharToMultiByte
    if (!utf8str.empty() && utf8str.back() == '\0') {
        utf8str.pop_back();
    }
    
    return utf8str;
}
#endif

// Improved signature: uses Eigen::Ref to avoid copies if passing blocks/maps
Eigen::VectorXf mean_pool(
    const Eigen::Ref<const Eigen::MatrixXf>& hidden,
    const Eigen::Ref<const Eigen::VectorXi>& mask
) {
    // 1. Safety Check
    if (hidden.rows() != mask.size()) {
        throw std::invalid_argument("Hidden state sequence length does not match mask length.");
    }

    // 2. Convert mask to float for matrix multiplication
    // Casting is usually very fast compared to the accumulation logic
    Eigen::VectorXf mask_f = mask.cast<float>();

    // 3. Calculate Count (Sum of mask)
    float count = mask_f.sum();
    
    // Edge case: empty mask
    if (count <= 0.0f) {
        return Eigen::VectorXf::Zero(hidden.cols());
    }

    // 4. Matrix Multiplication approach (The main optimization)
    // Formula: (1/N) * (mask^T * Hidden)
    //
    // mask_f             is [seq_len, 1]
    // hidden             is [seq_len, hidden_dim]
    // mask_f.transpose() is [1, seq_len]
    // result             is [1, hidden_dim]
    
    // Note: We create a temporary row vector, then transpose it back
    // to match the return type (VectorXf is a column vector).
    Eigen::VectorXf pooled = (mask_f.transpose() * hidden).transpose();

    return pooled / count;
}

Eigen::MatrixXf mean_pool_batch(
    const std::vector<Eigen::MatrixXf>& hidden_batch,
    const std::vector<Eigen::VectorXi>& mask_batch
) {
    // 1. Safety Checks
    if (hidden_batch.empty()) {
        return Eigen::MatrixXf(0, 0);
    }
    if (hidden_batch.size() != mask_batch.size()) {
        throw std::invalid_argument("Batch size mismatch between hidden states and masks.");
    }

    long batch_size = hidden_batch.size();
    long hidden_dim = hidden_batch[0].cols();

    // Allocate the result matrix once
    Eigen::MatrixXf out(batch_size, hidden_dim);

    // 2. Parallel Processing (OpenMP)
    // This distributes the rows across available CPU cores.
    #pragma omp parallel for
    for (int i = 0; i < (int)batch_size; ++i) {
        
        // --- Step A: Optimized Mean Pooling (Inlined) ---
        // We write directly into out.row(i) to avoid creating temporary VectorXf objects.
        
        const auto& hidden = hidden_batch[i];
        const auto& mask = mask_batch[i];

        // Convert mask to float for calculation
        Eigen::VectorXf mask_f = mask.cast<float>();
        float count = mask_f.sum();

        if (count > 0.0f) {
            // Matrix Mult: [1, seq] * [seq, dim] -> [1, dim]
            // We assign this directly to the output row.
            out.row(i) = mask_f.transpose() * hidden;
            out.row(i) /= count;
            
            // --- Step B: Optimized L2 Normalize (In-Place) ---
            // Calculate norm of the row we just wrote
            float norm = out.row(i).norm();
            
            if (norm > 1e-12f) {
                out.row(i) /= norm;
            }
        } else {
            // Handle edge case: empty mask -> zero vector
            out.row(i).setZero();
        }
    }

    return out;
}

Eigen::VectorXf l2_normalize(const Eigen::Ref<const Eigen::VectorXf>& v) {
    float norm = v.norm();
    // Use a small epsilon to prevent division by near-zero values
    // and ensure numerical stability.
    if (norm > 1e-12f)
        return v.normalized(); // Uses Eigen's optimized internal implementation
    // If norm is effectively zero, return the original (zero) vector
    return v;
}

#pragma mark -

static void usage(void)
{
    fprintf(stderr, "Usage:  mlc-llm -m model -w weights -i input\n\n");
    fprintf(stderr, "onnx-genai\n\n");
    fprintf(stderr, " -%c path     : %s\n", 'm' , "model");
    fprintf(stderr, " -%c path     : %s\n", 'w' , "weights");
    fprintf(stderr, " -%c path     : %s\n", 'e' , "embedding model");
    //
    fprintf(stderr, " -%c path     : %s\n", 'i' , "input");
    fprintf(stderr, " %c           : %s\n", '-' , "use stdin for input");
    fprintf(stderr, " -%c path     : %s\n", 'o' , "output (default=stdout)");
    //
    exit(1);
}

extern OPTARG_T optarg;
extern int optind, opterr, optopt;

#ifdef WIN32
OPTARG_T optarg = 0;
int opterr = 1;
int optind = 1;
int optopt = 0;
int getopt(int argc, OPTARG_T *argv, OPTARG_T opts) {
    
    static int sp = 1;
    register int c;
    register OPTARG_T cp;
    
    if(sp == 1)
        if(optind >= argc ||
           argv[optind][0] != '-' || argv[optind][1] == '\0')
            return(EOF);
        else if(wcscmp(argv[optind], L"--") == NULL) {
            optind++;
            return(EOF);
        }
    optopt = c = argv[optind][sp];
    if(c == ':' || (cp=wcschr(opts, c)) == NULL) {
        ERR(L": illegal option -- ", c);
        if(argv[optind][++sp] == '\0') {
            optind++;
            sp = 1;
        }
        return('?');
    }
    if(*++cp == ':') {
        if(argv[optind][sp+1] != '\0')
            optarg = &argv[optind++][sp+1];
        else if(++optind >= argc) {
            ERR(L": option requires an argument -- ", c);
            sp = 1;
            return('?');
        } else
            optarg = argv[optind++];
        sp = 1;
    } else {
        if(argv[optind][++sp] == '\0') {
            sp = 1;
            optind++;
        }
        optarg = NULL;
    }
    return(c);
}
#define ARGS (OPTARG_T)L"m:w:e:i:o:sp:-h"
#define _atoi _wtoi
#define _atof _wtof
#else
#define ARGS "m:w:e:i:o:sp:-h"
#define _atoi atoi
#define _atof atof
#endif

#pragma mark -

static long long get_created_timestamp() {
    // std::time(nullptr) returns the current time as a time_t (seconds since epoch)
    return static_cast<long long>(std::time(nullptr));
}

namespace fs = std::filesystem;
static std::string get_model_name(std::string model_path) {
    // 1. Create a path object
    fs::path path(model_path);
    
    // 2. Handle trailing slashes (e.g., "models/phi-3/")
    // If the path ends in a separator, filename() might return empty.
    if (path.filename().empty()) {
        path = path.parent_path();
    }
    
    // 3. Return the folder/filename
    // .filename() returns "phi-3.onnx" (with extension)
    // .stem() returns "phi-3" (removes extension)
    return path.stem().string();
}

// Generate a fingerprint based on model identity and hardware
static std::string get_system_fingerprint(const std::string& model_path, const std::string& provider) {
    // 1. Combine identifying factors (Model + Engine)
    std::string identifier = model_path + "_" + provider;
    
    // 2. Hash the string to get a unique number
    std::hash<std::string> hasher;
    size_t hash = hasher(identifier);
    
    // 3. Format as hex (e.g., "fp_1a2b3c4d")
    std::stringstream ss;
    ss << "fp_" << std::hex << hash;
    
    return ss.str();
}

static std::string get_openai_style_id() {
    const char charset[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    const size_t max_index = (sizeof(charset) - 1);
    
    std::string id = "chatcmpl-";
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, max_index - 1);
    
    for (int i = 0; i < 29; ++i) {
        id += charset[dis(gen)];
    }
    return id;
}

#pragma mark -

static void parse_request_embeddings(const std::string &json,
                                     std::string &input) {
    
    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errors;
    
    Json::CharReader *reader = builder.newCharReader();
    bool parse = reader->parse(json.c_str(),
                               json.c_str() + json.size(),
                               &root,
                               &errors);
    delete reader;
    
    if(parse)
    {
        if(root.isObject())
        {
            Json::Value input_node = root["input"];
            if(input_node.isString())
            {
                input = input_node.asString();
            }
        }
    }
}

static void parse_request(
                          const std::string &json,
                          std::string &prompt,
                          unsigned int *max_tokens,
                          unsigned int *top_k,
                          double *top_p,
                          double *temperature,
                          unsigned int *n,
                          bool *is_stream) {
    
    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errors;
    
    Json::CharReader *reader = builder.newCharReader();
    bool parse = reader->parse(json.c_str(),
                               json.c_str() + json.size(),
                               &root,
                               &errors);
    delete reader;
    
    if(parse)
    {
        if(root.isObject())
        {
            Json::Value messages_node = root["messages"];
            if(messages_node.isArray())
            {
                prompt = "";
                for(Json::Value::const_iterator it = messages_node.begin() ; it != messages_node.end() ; it++)
                {
                    if(it->isObject())
                    {
                        Json::Value defaultValue = "";
                        Json::Value role = it->get("role", defaultValue);
                        Json::Value content = it->get("content", defaultValue);
                        if ((role.isString()) && (content.isString()))
                        {
                            std::string role_str = role.asString();
                            std::string content_str = content.asString();
                            if ((role_str.length() != 0) && (content_str.length() != 0))
                            {
                                prompt += "<|";
                                prompt += role_str;
                                prompt += "|>";
                                prompt += content_str;
                                prompt += "<|end|>";
                            }
                        }
                    }
                }
                if(prompt.length() != 0) prompt += "<|assistant|>";
            }
            
            Json::Value top_p_node = root["top_p"];
            if(top_p_node.isNumeric())
            {
                *top_p = top_p_node.asDouble();
            }
            Json::Value top_k_node = root["top_k"];
            if(top_k_node.isNumeric())
            {
                *top_k = top_k_node.asInt();
            }
            Json::Value max_tokens_node = root["max_tokens"];
            if(max_tokens_node.isNumeric())
            {
                *max_tokens = max_tokens_node.asInt();
            }
            /*
             only these are set by AI-Kit
             */
            Json::Value temperature_node = root["temperature"];
            if(temperature_node.isNumeric())
            {
                *temperature = temperature_node.asDouble();
            }
            Json::Value n_node = root["n"];
            if(n_node.isNumeric())
            {
                *n = n_node.asInt();
            }
            max_tokens_node = root["max_completion_tokens"];
            if(max_tokens_node.isNumeric())
            {
                *max_tokens = max_tokens_node.asInt();
            }
            Json::Value stream_node = root["stream"];
            if(stream_node.isBool())
            {
                *is_stream = stream_node.asBool();
            }
        }
    }
}

static void before_run_embeddings(
                                  const std::string& request_body,
                                  std::string &input
                                  ) {
    parse_request_embeddings(request_body, input);
}

static void before_run_inference(
                                 const std::string& request_body,
                                 std::string &prompt,
                                 unsigned int *max_tokens,
                                 unsigned int *top_k,
                                 double *top_p,
                                 double *temperature,
                                 unsigned int *n,
                                 bool *is_stream) {
    
    parse_request(request_body, prompt, max_tokens, top_k, top_p, temperature, n, is_stream);
}

/*
 The chat completion chunk object
 https://platform.openai.com/docs/api-reference/chat-streaming/streaming
 */
static std::string create_stream_chunk(int n,
                                       const std::string& id,
                                       const std::string& model,
                                       const std::string& content,
                                       const std::string& fingerprint,
                                       bool finish) {
    Json::Value root;
    root["id"] = id;
    root["object"] = "chat.completion.chunk";
    root["created"] = (Json::UInt64)std::time(nullptr);
    root["model"] = model;
    root["system_fingerprint"] = fingerprint;//Deprecated
    
    Json::Value choice;
    choice["index"] = n;
    
    Json::Value delta;
    if (content.empty() && !finish) {
        delta["role"] = "assistant";
    } else {
        delta["content"] = content;
    }
    delta["logprobs"] = Json::nullValue;
    choice["delta"] = delta;
    
    if (finish) {
        choice["finish_reason"] = "stop";
    } else {
        choice["finish_reason"] = Json::nullValue;
    }
    root["choices"].append(choice);
    
    Json::StreamWriterBuilder writer;
    writer["indentation"] = "";
    return "data: " + Json::writeString(writer, root) + "\n\n";
}

static // Helper to convert int32 -> int64
std::vector<int64_t> ConvertToInt64(const std::vector<int>& input_ids) {
    std::vector<int64_t> output(input_ids.size());
    std::transform(input_ids.begin(), input_ids.end(), output.begin(),
                   [](int i) { return static_cast<int64_t>(i); });
    return output;
}

#pragma mark -

int main(int argc, OPTARG_T argv[]) {
    
#ifdef WIN32
    std::wstring model_path_u16;
    std::wstring weights_path_u16;
    std::wstring embedding_model_path_u16;
#endif
    std::string model_path;           // -m
    std::string weights_path;         // -w
    std::string embedding_model_path; // -e
    OPTARG_T input_path  = NULL;      // -i
    OPTARG_T output_path = NULL;      // -o
    
    // Server mode flags
    bool server_mode = false;         // -s
    int port = 8080;                  // -p
    std::string host = "127.0.0.1";   // -h
    
    std::vector<unsigned char> cli_request_json(0);
    
    int ch;
    
    while ((ch = getopt(argc, argv, ARGS)) != -1) {
        switch (ch){
            case 'm':
#ifdef WIN32
                model_path_u16 = optarg;
                model_path = wchar_to_utf8(model_path_u16.c_str());
#else
                model_path = optarg;
#endif
                break;
            case 'w':
#ifdef WIN32
                weights_path_u16 = optarg;
                weights_path = wchar_to_utf8(weights_path_u16.c_str());
#else
                weights_path = optarg;
#endif
                break;
            case 'e':
#ifdef WIN32
                embedding_model_path_u16 = optarg;
                embedding_model_path = wchar_to_utf8(embedding_model_path_u16.c_str());
#else
                embedding_model_path = optarg;
#endif
                break;
            case 'i':
                input_path = optarg;
                break;
            case 'o':
                output_path = optarg;
                break;
            case 's':
                server_mode = true;
                break;
            case 'p':
                port = std::stoi(optarg);
                break;
            case 'h':
#ifdef WIN32
                host = wchar_to_utf8(optarg);
#else
                host = optarg;
#endif
                break;
            case '-':
            {
                // Only relevant for CLI mode
                std::vector<uint8_t> buf(BUFLEN);
                size_t s;
                while ((s = fread(buf.data(), 1, buf.size(), stdin)) > 0) {
                    cli_request_json.insert(cli_request_json.end(), buf.begin(), buf.begin() + s);
                }
            }
                break;
            default:
                usage();
                break;
        }
    }
        
    std::string fingerprint;
    long long model_created = 0;
    std::string modelName;
    std::string model_parent_path;
    tvm::ffi::Function create_engine;

    tvm::ffi::Function init_background_engine;
    tvm::ffi::Function reload;
    tvm::ffi::Function unload;
    tvm::ffi::Function reset;
    tvm::ffi::Function chat_completion;
    tvm::ffi::Function abort;
    tvm::ffi::Function get_last_error;
    tvm::ffi::Function run_background_loop;
    tvm::ffi::Function run_background_stream_back_loop;
    tvm::ffi::Function exit_background_loop;
    
    if (model_path.length() != 0) {
        if (fs::exists(model_path)) {
            fingerprint = get_system_fingerprint(model_path, "directml");
            modelName = get_model_name(model_path);
            std::cerr << "[Chat] Loading from " << model_path << std::endl;
            try {
                
                if(weights_path == "")
                {
                    weights_path = fs::path(model_path).parent_path().c_str();
                }

                auto __create_engine__ = tvm::ffi::Function::GetGlobal("mlc.json_ffi.CreateJSONFFIEngine");
                if(__create_engine__.has_value()) {
                    create_engine = __create_engine__.value();
                    auto __engine__ = create_engine().as<tvm::ffi::Module>();
                    if(__engine__.has_value()) {
                        auto engine = __engine__.value();
                        auto __chat_completion__ = engine->GetFunction("chat_completion");
                        if(__chat_completion__.has_value()) {
                            chat_completion = __chat_completion__.value();
                            std::cout << "chat_completion" << std::endl;
                        }else{
                            throw "chat_completion function missing!";
                        }
                        auto __init_background_engine__ = engine->GetFunction("init_background_engine");
                        if(__init_background_engine__.has_value()) {
                            init_background_engine = __init_background_engine__.value();
                            std::cout << "init_background_engine" << std::endl;
                        }else{
                            throw "init_background_engine function missing!";
                        }
                        auto __reload__ = engine->GetFunction("reload");
                        if(__reload__.has_value()) {
                            reload = __reload__.value();
                            std::cout << "reload" << std::endl;
                        }else{
                            throw "reload function missing!";
                        }
                        auto __unload__ = engine->GetFunction("unload");
                        if(__unload__.has_value()) {
                            unload = __unload__.value();
                            std::cout << "unload" << std::endl;
                        }else{
                            throw "unload function missing!";
                        }
                        auto __abort__ = engine->GetFunction("abort");
                        if(__abort__.has_value()) {
                            abort = __abort__.value();
                            std::cout << "abort" << std::endl;
                        }else{
                            throw "abort function missing!";
                        }
                        auto __reset__ = engine->GetFunction("reset");
                        if(__reset__.has_value()) {
                            reset = __reset__.value();
                            std::cout << "reset" << std::endl;
                        }else{
                            throw "reset function missing!";
                        }
                        auto __get_last_error__ = engine->GetFunction("get_last_error");
                        if(__get_last_error__.has_value()) {
                            get_last_error = __get_last_error__.value();
                            std::cout << "get_last_error" << std::endl;
                        }else{
                            throw "get_last_error function missing!";
                        }

                        model_created = get_created_timestamp();
                        
                        tvm::ffi::TypedFunction<void(std::string)>
                        stream_callback = tvm::ffi::TypedFunction<void(std::string)>(StreamCallback);
                        
                        auto dummy_callback = tvm::ffi::TypedFunction<void(std::string)>(
                            [](std::string s) {
                                // Intentionally empty. Do NOT print here.
                            }
                        );
                        
                        init_background_engine(8, 0, dummy_callback);

                        std::string init_json = "{"
                        "\"model\": \"" + weights_path + "\","
                        "\"model_lib\": \"" + model_path + "\","
                        "\"mode\": \"interactive\""
                        "}";
                                    
                        std::cout << init_json << std::endl;

//                      unload();
                        
                        reset();
                        reload(init_json);
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "Failed to load model: " << e.what() << std::endl;
                return 1;
            }
        }
    }

        int device_id = 0;
        
        int device_type = kDLCPU;
#if defined(__APPLE__)
        device_type = kDLMetal;
#else
        device_type = kDLVulkan;
#endif
        
        if(model_created) {
            
            tvm::ffi::TypedFunction<void(std::string)>
            stream_callback = tvm::ffi::TypedFunction<void(std::string)>(StreamCallback);
            
//            auto dummy_callback = tvm::ffi::TypedFunction<void(std::string)>(
//                [](std::string s) {
//                    // Intentionally empty. Do NOT print here.
//                }
//            );
            
//            init_background_engine(8, device_id, dummy_callback);
//
//            std::string init_json = "{"
//            "\"model\": \"" + weights_path + "\","
//            "\"model_lib\": \"" + model_path + "\","
//            "\"mode\": \"interactive\""
//            "}";
                        
//            std::cout << init_json << std::endl;

//            reload("");
        }
        

    std::string embedding_fingerprint;
    long long embedding_model_created = 0;
    std::string embedding_modelName;
    
    // ---------------------------------------------------------
    // SERVER MODE
    // ---------------------------------------------------------
    if (server_mode) {
        httplib::Server svr;
        
        // Route: /v1/chat/completions
        svr.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
            
            std::cout << "[Server] /v1/chat/completions request received." << std::endl;
            
            try {
                
                if(model_created == 0) {
                    throw std::invalid_argument("[Chat] Model not loaded.");
                }
                
                std::string prompt;
                unsigned int max_tokens = 2048;
                unsigned int top_k = 50;
                double top_p = 0.9;
                double temperature = 0.7;
                unsigned int n = 1;
                bool is_stream = false;
                
                before_run_inference(req.body,
                                     prompt,
                                     &max_tokens,
                                     &top_k,
                                     &top_p,
                                     &temperature,
                                     &n,
                                     &is_stream);
                
                if(is_stream) {
                    std::string req_id = get_openai_style_id();
                    
                    // Corrected Lambda structure
                    res.set_chunked_content_provider("text/event-stream",
                                                     [&, req_id, prompt, max_tokens, top_k, top_p, temperature, n ](size_t offset, httplib::DataSink &sink) {
                        // Send initial role packet (optional but good practice)
                        for (int i = 0; i < n; i++) {
                            std::string role_chunk = "";
                            sink.write(role_chunk.data(), role_chunk.size());
                        }
                        
                        // Define a callback to handle tokens as they are generated
                        auto token_callback = [&](const std::string& token, unsigned int n) {
                            std::string chunk = "";
                            sink.write(chunk.data(), chunk.size());
                            return true; // Return false to stop inference if needed
                        };
                        
                        // Run Inference (You must implement run_inference_stream)
                        // Note: This function must block here until finished, calling token_callback repeatedly
//                        run_inference_stream(
//                                             model.get(),
//                                             tokenizer.get(),
//                                             modelName,
//                                             fingerprint,
//                                             model_created,
//                                             max_tokens,
//                                             top_k,
//                                             top_p,
//                                             temperature,
//                                             n,
//                                             prompt,
//                                             token_callback
//                                             );
                        // 4. Send finish reason
                        std::string finish_chunk = create_stream_chunk(n, req_id, modelName, fingerprint, "", true);
                        sink.write(finish_chunk.data(), finish_chunk.size());
                        
                        // 5. Send [DONE] to close the stream for the client
                        std::string done = "data: [DONE]\n\n";
                        sink.write(done.data(), done.size());
                        
                        sink.done(); // Close the connection
                        return true;
                    }
                                                     );
                    
                }else{
                    // Run Inference
//                    std::string response_json = run_inference(
//                                                              model.get(),
//                                                              tokenizer.get(),
//                                                              modelName,
//                                                              fingerprint,
//                                                              model_created,
//                                                              max_tokens,
//                                                              top_k,
//                                                              top_p,
//                                                              temperature,
//                                                              n,
//                                                              prompt
//                                                              );
//                    res.set_content(response_json, "application/json");
                    res.status = 200;
                }
            } catch (const std::exception& e) {
                // Build Error JSON
                Json::Value rootNode(Json::objectValue);
                Json::Value errorNode(Json::objectValue);
                errorNode["message"] = e.what();
                errorNode["type"] = "invalid_request_error";
                errorNode["param"] = Json::nullValue;
                errorNode["code"] = Json::nullValue;
                rootNode["error"] = errorNode;
                
                Json::StreamWriterBuilder writer;
                writer["indentation"] = "";
                std::string error_str = Json::writeString(writer, rootNode);
                
                res.set_content(error_str, "application/json");
                res.status = 400; // Bad Request as per requirement
                std::cerr << "[Server] Error: " << e.what() << std::endl;
            }
        });
        
        // Route: /v1/models
        svr.Get("/v1/models", [&](const httplib::Request& req, httplib::Response& res) {
            std::cout << "[Server] /v1/models request received." << std::endl;
            /*
             The model object
             https://platform.openai.com/docs/api-reference/models/object
             */
            // Create the list wrapper
            Json::Value root(Json::objectValue);
            root["object"] = "list";
            root["data"] = Json::Value(Json::arrayValue);
            // Create the model object
            if(model_created != 0) {
                Json::Value modelCard(Json::objectValue);
                modelCard["id"] = modelName;
                modelCard["object"] = "model";
                modelCard["created"] = model_created;
                modelCard["owned_by"] = "system";
                root["data"].append(modelCard);
            }
            if(embedding_model_created != 0) {
                Json::Value modelCard(Json::objectValue);
                modelCard["id"] = embedding_modelName;
                modelCard["object"] = "model";
                modelCard["created"] = embedding_model_created;
                modelCard["owned_by"] = "system";
                root["data"].append(modelCard);
            }
            // Serialize
            Json::StreamWriterBuilder writer;
            writer["indentation"] = ""; // Minified JSON
            std::string json_str = Json::writeString(writer, root);
            // Respond
            res.set_content(json_str, "application/json");
            res.status = 200;
        });
        
        // Route: /v1/embeddings
        svr.Post("/v1/embeddings", [&](const httplib::Request& req, httplib::Response& res) {
            
            std::cout << "[Server] /v1/embeddings request received." << std::endl;
            
            try {
                
                if(embedding_model_created == 0) {
                    throw std::invalid_argument("[Embedding] Model not loaded.");
                }
               
                std::string input;
                before_run_embeddings(req.body, input);
                
                std::string response_json;

//                if((embeddings_tokenizer != NULL) &&(num_input_nodes > 2)) {
                    /*
                     standard encoder-only model
                     input: input_ids, attention_mask, token_type_ids
                     output: last_hidden_state or logits
                     */
//                    std::vector<int> ids = embeddings_tokenizer->Encode(input);
//                    response_json = run_embeddings(
//                                                   embeddings_session.get(),
//                                                   ids, input_names_c_array,
//                                                   num_input_nodes,
//                                                   output_names_c_array,
//                                                   num_output_nodes);
//                }else{
//                    response_json = run_embeddings_e2e(
//                                                       embeddings_session.get(),
//                                                       input, input_names_c_array,
//                                                       num_input_nodes,
//                                                       output_names_c_array,
//                                                       num_output_nodes);
//                }
                res.set_content(response_json, "application/json");
                res.status = 200;
            } catch (const std::exception& e) {
                // Build Error JSON
                Json::Value rootNode(Json::objectValue);
                Json::Value errorNode(Json::objectValue);
                errorNode["message"] = e.what();
                errorNode["type"] = "invalid_request_error";
                errorNode["param"] = Json::nullValue;
                errorNode["code"] = Json::nullValue;
                rootNode["error"] = errorNode;
                
                Json::StreamWriterBuilder writer;
                writer["indentation"] = "";
                std::string error_str = Json::writeString(writer, rootNode);
                
                res.set_content(error_str, "application/json");
                res.status = 400; // Bad Request as per requirement
                std::cerr << "[Server] Error: " << e.what() << std::endl;
            }
            
        });
        
        std::cout << "[Server] Listening on " << host << ":" << port << std::endl;
        
        // Listen (Blocking call)
        if (!svr.listen(host.c_str(), port)) {
            std::cerr << "Error: Could not start server on " << host << ":" << port << std::endl;
            return 1;
        }
    }
    // ---------------------------------------------------------
    // CLI MODE
    // ---------------------------------------------------------
    else {
        // Handle input file reading if not piped via stdin ('-')
        if ((!cli_request_json.size()) && (input_path != NULL)) {
            FILE *f = _fopen(input_path, _rb);
            if(f) {
                fseek(f, 0, SEEK_END);
                size_t len = (size_t)ftell(f);
                fseek(f, 0, SEEK_SET);
                cli_request_json.resize(len);
                fread(cli_request_json.data(), 1, cli_request_json.size(), f);
                fclose(f);
            }
        }
        
        if (cli_request_json.size() == 0) {
            usage();
            return 1;
        }
                
        std::string request_str((const char *)cli_request_json.data(), cli_request_json.size());
        std::string response;
        
        try {
            
            std::string req_id = get_openai_style_id();
            unsigned int n = 1;
            
            set_request_context(n, req_id);
            chat_completion(request_str, req_id);
            response = active_requests[req_id].accumulated_text;
            unset_request_context(req_id);
            
        } catch (const std::exception& e) {
            // CLI Error Format
            Json::Value rootNode(Json::objectValue);
            Json::Value errorNode(Json::objectValue);
            rootNode["error"] = errorNode;
            errorNode["message"] = e.what();
            errorNode["type"] = "invalid_request_error";
            
            Json::StreamWriterBuilder writer;
            writer["indentation"] = "";
            response = Json::writeString(writer, rootNode);
        }
        
        // Output logic
        if(!output_path) {
            std::cout << response << std::endl;
        } else {
            FILE *f = _fopen(output_path, _wb);
            if(f) {
                fwrite(response.c_str(), 1, response.length(), f);
                fclose(f);
            }
        }
    }
    
    return 0;
}
