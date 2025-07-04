// // ggml-cpu-taskflow.cpp
// #include <taskflow/taskflow.hpp>
// #include "ggml.h"

// extern "C" {
//     #include "ggml-cpu-taskflow.h"
// }

// void ggml_compute_forward_mul_mat_taskflow(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
//     // taskflow 初始化
//     tf::Executor executor;
//     tf::Taskflow flow;

//     // 举例：你可以提交一些 chunk 任务
//     for (int i = 0; i < 4; ++i) {
//         flow.emplace([=]() {
//             printf("Running chunk %d\n", i);
//             // 实际 matmul 逻辑...
//         });
//     }

//     executor.run(flow).wait();
// }
