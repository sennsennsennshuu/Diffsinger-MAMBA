/*
 * ONNX Runtime Custom Operator for SSM Selective Scan
 *
 * This implements the selective scan as an ONNX Runtime custom operator
 * for seamless integration with DiffSinger ONNX models.
 */

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <cstring>
#include "selective_scan.h"
#include "state_cache.h"

namespace ssm {
namespace ort {

// Custom operator kernel for selective scan
class SelectiveScanKernel : public Ort::CustomOpBase<SelectiveScanKernel, void> {
public:
    SelectiveScanKernel() = default;

    void Compute(OrtKernelContext* context) {
        Ort::KernelContext ctx(context);
        
        // Get input tensors
        auto input = ctx.GetInput(0);      // [batch, seq_len, d_inner]
        auto dt = ctx.GetInput(1);         // [batch, seq_len, n_heads]
        auto A = ctx.GetInput(2);          // [n_heads, d_state]
        auto B = ctx.GetInput(3);          // [batch, seq_len, d_state]
        auto C = ctx.GetInput(4);          // [batch, seq_len, d_state]
        
        // D (skip connection) is optional
        const Ort::Value* D_ptr = nullptr;
        if (ctx.GetInputCount() > 5) {
            D_ptr = &ctx.GetInput(5);
        }

        // Get tensor info
        auto input_info = input.GetTensorTypeAndShapeInfo();
        auto input_shape = input_info.GetShape();
        
        int batch_size = static_cast<int>(input_shape[0]);
        int seq_len = static_cast<int>(input_shape[1]);
        int d_inner = static_cast<int>(input_shape[2]);
        
        auto A_info = A.GetTensorTypeAndShapeInfo();
        auto A_shape = A_info.GetShape();
        int n_heads = static_cast<int>(A_shape[0]);
        int d_state = static_cast<int>(A_shape[1]);
        int head_dim = d_inner / n_heads;

        // Get data pointers
        const float* input_data = input.GetTensorData<float>();
        const float* dt_data = dt.GetTensorData<float>();
        const float* A_data = A.GetTensorData<float>();
        const float* B_data = B.GetTensorData<float>();
        const float* C_data = C.GetTensorData<float>();
        const float* D_data = D_ptr ? D_ptr->GetTensorData<float>() : nullptr;

        // Create output tensor
        std::vector<int64_t> output_shape = {batch_size, seq_len, d_inner};
        auto output = ctx.GetOutput(0, output_shape);
        float* output_data = output.GetTensorMutableData<float>();

        // Configure scan
        ScanConfig config;
        config.use_simd = true;
        config.use_openmp = true;
        config.chunk_size = 64;

        // Execute selective scan
        selective_scan(
            input_data, dt_data, A_data, B_data, C_data, D_data,
            output_data, batch_size, seq_len, d_inner, n_heads, d_state, head_dim,
            config
        );
    }

    static Ort::OpSchema GetSchema() {
        Ort::OpSchema schema("SSMSelectiveScan", "custom.ssm");
        
        // Input 0: input [batch, seq_len, d_inner]
        schema.AddInput("input", "Input tensor", "T", 3);
        
        // Input 1: dt [batch, seq_len, n_heads]
        schema.AddInput("dt", "Time delta", "T", 3);
        
        // Input 2: A [n_heads, d_state]
        schema.AddInput("A", "State transition matrix", "T", 2);
        
        // Input 3: B [batch, seq_len, d_state]
        schema.AddInput("B", "Input matrix", "T", 3);
        
        // Input 4: C [batch, seq_len, d_state]
        schema.AddInput("C", "Output matrix", "T", 3);
        
        // Input 5: D [n_heads] (optional)
        schema.AddInput("D", "Skip connection", "T", 1, true);
        
        // Output 0: output [batch, seq_len, d_inner]
        schema.AddOutput("output", "Output tensor", "T", 3);
        
        // Type constraint
        schema.AddTypeConstraint("T", {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT});
        
        return schema;
    }
};

// Custom operator kernel for optimized selective scan with state caching
class SelectiveScanCachedKernel : public Ort::CustomOpBase<SelectiveScanCachedKernel, void> {
public:
    SelectiveScanCachedKernel() = default;

    void Compute(OrtKernelContext* context) {
        Ort::KernelContext ctx(context);
        
        // Similar to SelectiveScanKernel but with state caching support
        auto input = ctx.GetInput(0);
        auto dt = ctx.GetInput(1);
        auto A = ctx.GetInput(2);
        auto B = ctx.GetInput(3);
        auto C = ctx.GetInput(4);
        
        const Ort::Value* D_ptr = nullptr;
        if (ctx.GetInputCount() > 5) {
            D_ptr = &ctx.GetInput(5);
        }
        
        // Cache key input (optional)
        std::string cache_key;
        if (ctx.GetInputCount() > 6) {
            auto key_tensor = ctx.GetInput(6);
            auto key_info = key_tensor.GetTensorTypeAndShapeInfo();
            auto key_shape = key_info.GetShape();
            if (key_shape.size() > 0 && key_shape[0] > 0) {
                const char* key_data = key_tensor.GetTensorData<char>();
                cache_key = std::string(key_data, key_shape[0]);
            }
        }

        auto input_info = input.GetTensorTypeAndShapeInfo();
        auto input_shape = input_info.GetShape();
        
        int batch_size = static_cast<int>(input_shape[0]);
        int seq_len = static_cast<int>(input_shape[1]);
        int d_inner = static_cast<int>(input_shape[2]);
        
        auto A_info = A.GetTensorTypeAndShapeInfo();
        auto A_shape = A_info.GetShape();
        int n_heads = static_cast<int>(A_shape[0]);
        int d_state = static_cast<int>(A_shape[1]);
        int head_dim = d_inner / n_heads;

        const float* input_data = input.GetTensorData<float>();
        const float* dt_data = dt.GetTensorData<float>();
        const float* A_data = A.GetTensorData<float>();
        const float* B_data = B.GetTensorData<float>();
        const float* C_data = C.GetTensorData<float>();
        const float* D_data = D_ptr ? D_ptr->GetTensorData<float>() : nullptr;

        std::vector<int64_t> output_shape = {batch_size, seq_len, d_inner};
        auto output = ctx.GetOutput(0, output_shape);
        float* output_data = output.GetTensorMutableData<float>();

        ScanConfig config;
        config.use_simd = true;
        config.use_openmp = true;
        config.chunk_size = 64;

        selective_scan(
            input_data, dt_data, A_data, B_data, C_data, D_data,
            output_data, batch_size, seq_len, d_inner, n_heads, d_state, head_dim,
            config
        );
    }

    static Ort::OpSchema GetSchema() {
        Ort::OpSchema schema("SSMSelectiveScanCached", "custom.ssm");
        
        schema.AddInput("input", "Input tensor", "T", 3);
        schema.AddInput("dt", "Time delta", "T", 3);
        schema.AddInput("A", "State transition matrix", "T", 2);
        schema.AddInput("B", "Input matrix", "T", 3);
        schema.AddInput("C", "Output matrix", "T", 3);
        schema.AddInput("D", "Skip connection", "T", 1, true);
        schema.AddInput("cache_key", "Cache key for state reuse", "tensor(string)", 1, true);
        
        schema.AddOutput("output", "Output tensor", "T", 3);
        schema.AddTypeConstraint("T", {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT});
        
        return schema;
    }
};

// Registration function
Ort::CustomOpDomain* CreateSSMCustomOpDomain() {
    auto* domain = new Ort::CustomOpDomain("custom.ssm");
    
    static Ort::CustomOpBase<SelectiveScanKernel, void> selective_scan_op;
    static Ort::CustomOpBase<SelectiveScanCachedKernel, void> selective_scan_cached_op;
    
    domain->Add(&selective_scan_op);
    domain->Add(&selective_scan_cached_op);
    
    return domain;
}

} // namespace ort
} // namespace ssm

// C API for registration
extern "C" {

void* SSM_CreateCustomOpDomain() {
    return ssm::ort::CreateSSMCustomOpDomain();
}

void SSM_DestroyCustomOpDomain(void* domain) {
    delete static_cast<Ort::CustomOpDomain*>(domain);
}

} // extern "C"
