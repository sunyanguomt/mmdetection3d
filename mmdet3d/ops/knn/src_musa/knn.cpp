// Modified from https://github.com/CVMI-Lab/PAConv/tree/main/scene_seg/lib/pointops/src/knnquery_heap

#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>
#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_privateuseone(), #x, " must be a MUSAtensor ")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)


void knn_kernel_launcher(
    int b,
    int n,
    int m,
    int nsample,
    const float *xyz,
    const float *new_xyz,
    int *idx,
    float *dist2,
    musaStream_t stream
    );

void knn_wrapper(int b, int n, int m, int nsample, at::Tensor xyz_tensor, at::Tensor new_xyz_tensor, at::Tensor idx_tensor, at::Tensor dist2_tensor)
{
    CHECK_INPUT(new_xyz_tensor);
    CHECK_INPUT(xyz_tensor);

    const float *new_xyz = new_xyz_tensor.data_ptr<float>();
    const float *xyz = xyz_tensor.data_ptr<float>();
    int *idx = idx_tensor.data_ptr<int>();
    float *dist2 = dist2_tensor.data_ptr<float>();

    musaStream_t stream = at::musa::getCurrentMUSAStream();

    knn_kernel_launcher(b, n, m, nsample, xyz, new_xyz, idx, dist2, stream);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knn_wrapper", &knn_wrapper, "knn_wrapper");
}
