// #include <taskflow/taskflow.hpp>
// #include "ggml-cpu.h"


// struct TaskflowWrapper {
//     tf::Executor executor;
//     tf::Taskflow flow;
// };

// extern "C" {

// void taskflow_graph_create(cgraph* g) {
//     g->taskflow_graph = new TaskflowWrapper();

//     // 示例：添加一些任务
//     TaskflowWrapper* wrapper = (TaskflowWrapper*)g->taskflow_graph;
//     wrapper->flow.emplace([]() { printf("Hello from Taskflow!\n"); });
// }

// void taskflow_graph_add_task(cgraph* g, void (*task_func)()) {
//     TaskflowWrapper* wrapper = (TaskflowWrapper*)g->taskflow_graph;
//     wrapper->flow.emplace(task_func);
// }

// void taskflow_graph_copy_cgraph(cgraph* g, void (*task_func)()) {

// }

// void taskflow_graph_run(cgraph* g) {
//     TaskflowWrapper* wrapper = (TaskflowWrapper*)g->taskflow_graph;
//     wrapper->executor.run(wrapper->flow).wait();
// }

// void taskflow_graph_destroy(cgraph* g) {
//     TaskflowWrapper* wrapper = (TaskflowWrapper*)g->taskflow_graph;
//     delete wrapper;
//     g->taskflow_graph = NULL;
// }

// }

// taskflow-impl.cpp

#include "ggml-impl.h" // 用于 struct taskflow_taskgraph 定义
#include <taskflow/taskflow.hpp>
#include <memory>
#include <unordered_map>
#include <cassert>

// 自定义包装结构，用于隐藏 C++ 类型
struct TaskflowWrapper {
    
    tf::Taskflow flow;
    tf::Executor executor;

    // 可选：你可以扩展图结构，比如维护节点ID到 task 的映射
};

// 创建 taskflow graph
extern "C" void taskflow_graph_init(struct taskflow_taskgraph* tg) {
    if (tg->taskflow_graph != nullptr) {
        // 如果已经初始化，直接返回
        return;
    }
    if (tg->is_init == 1) {
        // 如果已经初始化，直接返回
        return;
    }
    
    tg->taskflow_graph = new TaskflowWrapper();
    tg->is_init = 1;
}

// 添加 task：你也可以设计更通用的版本，比如传入 C 回调
extern "C" void taskflow_graph_add_task(struct taskflow_taskgraph* tg, const char* name) {
    auto* wrapper = static_cast<TaskflowWrapper*>(tg->taskflow_graph);
    wrapper->flow.emplace([=]() {
        printf("Running task: %s\n", name);
    }).name(name);
}

// 执行图
extern "C" void taskflow_graph_run(struct taskflow_taskgraph* tg) {
    auto* wrapper = static_cast<TaskflowWrapper*>(tg->taskflow_graph);
    wrapper->executor.run(wrapper->flow).wait();
}

// 清理资源
extern "C" void taskflow_graph_free(struct taskflow_taskgraph* tg) {
    delete static_cast<TaskflowWrapper*>(tg->taskflow_graph);
    tg->taskflow_graph = nullptr;
}

extern "C" void taskflow_graph_hello(struct taskflow_taskgraph* tg) {
    taskflow_graph_init(tg);
    auto* wrapper = static_cast<TaskflowWrapper*>(tg->taskflow_graph);
    wrapper->flow.emplace([]() {
        printf("Hello from Taskflow!\n");
    });
    printf("Hello Taskflow graph.\n");
}