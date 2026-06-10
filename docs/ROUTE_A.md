# Route-A：4 个实验

| 命令 | 输出目录 | 开启模块 | 训练方式 |
|------|---------|---------|---------|
| `./run_route_a.sh map` | `spa_a_map` | 仅 map | 全流程 140ep |
| `./run_route_a.sh fg` | `spa_a_fg` | 仅 fg | student_only ~40ep |
| `./run_route_a.sh attn` | `spa_a_attn` | 仅 attn | student_only ~40ep |
| `./run_route_a.sh all` | `spa_a_all` | map+fg+attn | 全流程 140ep |

```bash
# gpuq 一次提交 4 个
./submit_route_a_gpuq.sh all parallel

# 或单独
./submit_route_a_gpuq.sh map
```

Teacher（fg/attn 用）：`output/avenue/r0_mse_skip/checkpoint-best.pth`
