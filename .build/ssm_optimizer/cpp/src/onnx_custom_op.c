/*
 * ONNX Runtime Custom Operator for SSM Selective Scan (C API version)
 *
 * Uses OrtApi functions for domain creation. No compile-time .lib linking needed.
 * Compatible with ONNX Runtime >= 1.10.
 */

#include <onnxruntime_c_api.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

void ssm_selective_scan_f32(
    const float* input, const float* dt, const float* A,
    const float* B, const float* C, const float* D,
    float* output,
    int batch_size, int seq_len, int d_inner,
    int n_heads, int d_state, int head_dim,
    int use_simd);

#ifdef __cplusplus
}
#endif

/* ─── Compute callback ─── */
static void SSM_Compute(OrtKernelContext* context, void* kernel_state) {
    (void)kernel_state;
    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    
    const OrtValue* input_val  = api->KernelContext_GetInput(context, 0);
    const OrtValue* dt_val     = api->KernelContext_GetInput(context, 1);
    const OrtValue* A_val      = api->KernelContext_GetInput(context, 2);
    const OrtValue* B_val      = api->KernelContext_GetInput(context, 3);
    const OrtValue* C_val      = api->KernelContext_GetInput(context, 4);
    const OrtValue* D_val      = NULL;
    if (api->KernelContext_GetInputCount(context) > 5)
        D_val = api->KernelContext_GetInput(context, 5);

    /* Get shapes from input[0] */{
        OrtTensorTypeAndShapeInfo* info;
        api->GetTensorTypeAndShape(input_val, &info);
        int64_t shape[8]; size_t dims;
        api->GetDimensions(info, shape, 8, &dims);
        api->ReleaseTensorTypeAndShapeInfo(info);
        int64_t batch_size = shape[0], seq_len = shape[1], d_inner = shape[2];
        
        /* Get n_heads, d_state from A */{
            OrtTensorTypeAndShapeInfo* A_info;
            api->GetTensorTypeAndShape(A_val, &A_info);
            api->GetDimensions(A_info, shape, 8, &dims);
            api->ReleaseTensorTypeAndShapeInfo(A_info);
            int64_t n_heads = shape[0], d_state = shape[1];
            int64_t head_dim = d_inner / n_heads;
            
            /* Data ptrs (const_cast for C API compatibility) */
            const float* inp, *d, *a, *b, *c, *dd;
            api->GetTensorMutableData((OrtValue*)input_val, (void**)&inp);
            api->GetTensorMutableData((OrtValue*)dt_val,    (void**)&d);
            api->GetTensorMutableData((OrtValue*)A_val,     (void**)&a);
            api->GetTensorMutableData((OrtValue*)B_val,     (void**)&b);
            api->GetTensorMutableData((OrtValue*)C_val,     (void**)&c);
            if (D_val) api->GetTensorMutableData((OrtValue*)D_val, (void**)&dd);
            else dd = NULL;
            
            /* Create output */{
                int64_t out_shape[] = {batch_size, seq_len, d_inner};
                OrtValue* out_val = api->KernelContext_GetOutput(context, 0, out_shape, 3);
                float* out;
                api->GetTensorMutableData(out_val, (void**)&out);
                
                ssm_selective_scan_f32(
                    inp, d, a, b, c, dd, out,
                    (int)batch_size, (int)seq_len, (int)d_inner,
                    (int)n_heads, (int)d_state, (int)head_dim, 1);
            }
        }
    }
}

/* ─── Entry point ORT expects ─── */
EXPORT OrtCustomOpDomain* OrtGetCustomOpDomain(void) {
    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    
    /* Define the custom op */
    OrtCustomOpDomain* domain = NULL;
    api->CreateCustomOpDomain("custom.ssm", &domain);
    
    OrtCustomOp* op;
    api->CustomOpDomain_Add(domain, &op);
    
    op->version = ORT_CUSTOM_OPS_ABI_VERSION;
    op->CreateKernel = NULL;      /* stateless op */
    op->Compute = SSM_Compute;
    op->GetName = NULL;
    op->GetExecutionProviderType = NULL;
    op->GetInputMemoryType = NULL;
    
    return domain;
}

EXPORT void SSMCustomOp_NoOp(void) {}
