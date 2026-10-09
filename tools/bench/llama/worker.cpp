// Separate-process comparison worker; no numerical code enters Toklet.
#include "llama.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
constexpr size_t CAPACITY = 2048;
constexpr size_t CHUNK = 128;
constexpr uint64_t LIMIT = 6ULL * 1024 * 1024 * 1024;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
void require(bool valid, const char *message) {
    if (!valid) throw std::runtime_error(message);
}
struct Backend {
    Backend() { llama_backend_init(); }
    ~Backend() { llama_backend_free(); }
};
struct Batch {
    llama_batch value = llama_batch_init(CHUNK, 0, 1);
    ~Batch() { llama_batch_free(value); }
    Batch(const Batch &) = delete;
    Batch &operator=(const Batch &) = delete;
    Batch() = default;
};

struct Engine {
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model{nullptr, llama_model_free};
    std::unique_ptr<llama_context, decltype(&llama_free)> context{nullptr, llama_free};
    Batch batch;
    std::vector<llama_token> history;
    size_t evaluated = 0;
    int32_t vocab = 0;

    explicit Engine(const std::string &path) {
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = -1;
        mp.check_tensors = true;
        model.reset(llama_model_load_from_file(path.c_str(), mp));
        require(bool(model), "model load failed");
        vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        require(vocab == 151936 && llama_model_n_layer(model.get()) == 24 &&
                llama_model_n_embd(model.get()) == 896 && llama_model_n_head(model.get()) == 14 && llama_model_n_head_kv(model.get()) == 2,
                "expected pinned Qwen2.5-0.5B model shapes");
        auto cp = llama_context_default_params();
        cp.n_ctx = CAPACITY;
        cp.n_batch = CHUNK;
        cp.n_ubatch = CHUNK;
        cp.n_seq_max = 1;
        cp.n_threads = 1;
        cp.n_threads_batch = 1;
        cp.type_k = GGML_TYPE_F32;
        cp.type_v = GGML_TYPE_F32;
        cp.offload_kqv = true;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        context.reset(llama_init_from_model(model.get(), cp));
        require(bool(context), "context creation failed");
        require(llama_n_ctx(context.get()) == CAPACITY, "unexpected context capacity");
        history.reserve(CAPACITY);
        char file_type[16];
        require(llama_model_meta_val_str(model.get(), "general.file_type", file_type, sizeof(file_type)) > 0 && std::string(file_type) == "0", "expected all-FP32 GGUF");
    }
    void reset() {
        llama_memory_clear(llama_get_memory(context.get()), false);
        history.clear();
        evaluated = 0;
    }
    void set(const std::vector<llama_token> &tokens) {
        if (tokens.size() < history.size() || !std::equal(history.begin(), history.end(), tokens.begin())) reset();
        history.assign(tokens.begin(), tokens.end());
    }
    void append(const std::vector<llama_token> &tokens) {
        history.insert(history.end(), tokens.begin(), tokens.end());
    }
    void evaluate(size_t position_offset = 0) {
        while (evaluated < history.size()) {
            const size_t count = std::min(CHUNK, history.size() - evaluated);
            auto &b = batch.value;
            b.n_tokens = static_cast<int32_t>(count);
            for (size_t i = 0; i < count; ++i) {
                b.token[i] = history[evaluated + i];
                b.pos[i] = static_cast<llama_pos>(position_offset + evaluated + i);
                b.n_seq_id[i] = 1;
                b.seq_id[i][0] = 0;
                b.logits[i] = i + 1 == count;
            }
            if (llama_decode(context.get(), b) != 0) {
                // Vendor failures can leave processed microbatches. Require
                // replay rather than treating a partial chunk as committed.
                llama_memory_clear(llama_get_memory(context.get()), false);
                evaluated = 0;
                throw std::runtime_error("llama decode failed; prefix requires replay");
            }
            llama_synchronize(context.get());
            require(llama_get_logits_ith(context.get(), -1) != nullptr, "missing completed logits");
            evaluated += count;
        }
    }
    std::vector<llama_token> ids(const json &value) const {
        require(value.is_array(), "token input must be an array");
        require(value.size() <= CAPACITY, "too many IDs");
        std::vector<llama_token> result;
        result.reserve(value.size());
        for (const auto &v : value) {
            require(v.is_number_integer(), "ID must be an integer");
            if (!v.is_number_unsigned()) require(v.get<int64_t>() >= 0, "negative ID");
            const auto id = v.get<uint64_t>();
            require(id < static_cast<uint64_t>(vocab), "ID outside vocabulary");
            result.push_back(static_cast<llama_token>(id));
        }
        return result;
    }
    json request(const json &r) {
        require(r.is_object(), "request must be an object");
        for (const auto &[key, value] : r.items()) {
            (void)value;
            require(key == "action" || key == "tokens" || key == "continuation" || key == "path" || key == "position_offset", "unknown request field");
        }
        const auto action = r.at("action").get<std::string>();
        require(action == "reset" || action == "prompt" || action == "append" || action == "continue" || action == "prepare" || action == "stage" || action == "probe", "unknown action");
        const auto tokens = ids(r.value("tokens", json::array()));
        const auto fixed = ids(r.value("continuation", json::array()));
        require(action != "continue" || tokens.empty(), "continue accepts no tokens");
        const size_t base = action == "append" || action == "continue" || action == "stage" ? history.size() : 0;
        require(base + tokens.size() + fixed.size() <= CAPACITY, "prompt plus requested output exceeds session capacity");
        if (action == "prepare" || action == "stage" || action == "probe") {
            require(fixed.empty(), "preparation accepts no continuation");
            require(!r.contains("path") || action == "probe", "output path is only for probes");
            const auto offset_value = r.value("position_offset", json(0));
            require(offset_value.is_number_integer(), "position offset must be an integer");
            if (!offset_value.is_number_unsigned()) require(offset_value.get<int64_t>() >= 0, "negative position offset");
            const uint64_t offset = offset_value.get<uint64_t>();
            require(offset <= CAPACITY && tokens.size() <= CAPACITY - offset, "invalid position offset");
            if (action == "probe") {
                require(!tokens.empty() && r.contains("path"), "probe needs tokens and output path");
                const auto path = r.at("path").get<std::string>();
                require(!std::filesystem::exists(path), "probe output exists");
                // Diagnostics consume/reset cache state, including on I/O
                // failure, so an offset probe cannot contaminate zero-offset work.
                struct ResetOnExit { Engine &engine; ~ResetOnExit() {engine.reset();} } cleanup{*this};
                reset();
                append(tokens);
                evaluate(offset);
                static_assert(std::endian::native == std::endian::little && sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
                const auto *logits = llama_get_logits_ith(context.get(), -1);
                require(std::all_of(logits, logits + vocab, [](float x) {return std::isfinite(x);}), "nonfinite probe logits");
                std::ofstream output(path, std::ios::binary);
                output.write(reinterpret_cast<const char *>(logits), static_cast<std::streamsize>(vocab * sizeof(float)));
                output.flush();
                require(bool(output), "probe write failed");
            } else if (action == "prepare") {
                require(offset == 0, "prepare uses zero position offset");
                set(tokens);
                evaluate();
            } else {
                require(offset == 0, "stage uses zero position offset");
                append(tokens);
            }
            return { {"event", "prepared"}, {"evaluated_tokens", evaluated}, {"pending_tokens", history.size() - evaluated} };
        }
        require(!r.contains("path") && !r.contains("position_offset"), "generation accepts no probe fields");
        require(!fixed.empty() && base + tokens.size() > 0, "nonempty prompt and output required");
        std::vector<double> events;
        events.reserve(fixed.size());
        const auto started = Clock::now();
        if (action == "reset") { reset(); append(tokens); }
        else if (action == "prompt") set(tokens);
        else if (action == "append") append(tokens);
        const auto reused = evaluated;
        const auto pending = history.size() - evaluated;
        const auto prefill_start = Clock::now();
        evaluate();
        const auto prefill = elapsed(prefill_start);
        events.push_back(elapsed(started));
        history.push_back(fixed.front());
        for (size_t i = 1; i < fixed.size(); ++i) {
            evaluate();
            events.push_back(elapsed(started));
            history.push_back(fixed[i]);
        }
        return { {"event", "sample"}, {"sample", {
            {"prefill_seconds", prefill}, {"ttft_seconds", events.front()},
            {"decode_seconds", events.back() - events.front()}, {"total_seconds", elapsed(started)},
            {"token_event_seconds", events}, {"emitted_tokens", fixed.size()}, {"reused_tokens", reused},
            {"prefill_tokens", pending}, {"evaluated_tokens", evaluated}, {"pending_tokens", history.size() - evaluated}
        }} };
    }
};

int main(int argc, char **argv) {
    try {
        require(argc == 2, "usage: toklet-llama-bench PINNED_F32_GGUF");
        const auto file_bytes = std::filesystem::file_size(argv[1]);
        require(file_bytes <= (LIMIT - 1024ULL*1024*1024 - 50331648) / 2, "conservative load budget exceeded");
        Backend backend;
        bool metal = false;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            const auto device = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU && std::string(ggml_backend_dev_name(device)).find("MTL") != std::string::npos) metal = true;
        }
        require(metal, "Metal backend unavailable");
        const auto start = Clock::now();
        Engine engine(argv[1]);
        std::cout << json({{"event", "ready"}, {"load_seconds", elapsed(start)}, {"conversion_seconds", nullptr},
            {"memory_admitted_bytes", 2*file_bytes + 1024ULL*1024*1024 + 50331648}, {"logical_weights_bytes", llama_model_size(engine.model.get())},
            {"allocated_kv_bytes", 50331648}, {"scratch_bytes", nullptr}, {"chunk_size", CHUNK}, {"context_capacity", CAPACITY}}).dump() << std::endl;
        char line[65537];
        while (std::cin.getline(line, sizeof(line))) {
            try { std::cout << engine.request(json::parse(line)).dump() << std::endl; }
            catch (const std::exception &error) { std::cout << json({{"event", "error"}, {"message", error.what()}}).dump() << std::endl; }
        }
        require(std::cin.eof(), "request line exceeds 64 KiB or input failed");
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << std::endl;
        return 1;
    }
    return 0;
}
