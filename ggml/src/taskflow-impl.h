#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "ggml-impl.h"  // 用于 struct taskflow_taskgraph 定义

void taskflow_graph_hello(struct taskflow_taskgraph* tg);
void taskflow_graph_init(struct taskflow_taskgraph* tg);
void taskflow_graph_add_task(struct taskflow_taskgraph* tg, const char* name);
void taskflow_graph_run(struct taskflow_taskgraph* tg);
void taskflow_graph_free(struct taskflow_taskgraph* tg);

#ifdef __cplusplus
}
#endif
