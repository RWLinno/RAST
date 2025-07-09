# RAST GPU设置和计时功能使用指南

## 🎯 问题解决方案

### ✅ 问题1：GPU使用问题
**原因：** 混合精度AMP与GPU设备设置冲突
**解决方案：** 
- 禁用AMP（`use_amp: False`）
- 直接指定GPU设备（`device_id: 2`）
- 强制模型和数据移动到指定设备

### ✅ 问题2：计时分析需求
**解决方案：** 
- 添加详细的模块计时功能（`timing_mode: True`）
- 分析每个模块的参数量和执行时间
- 提供性能优化建议并自动退出

## 🚀 新增参数

在`MODEL_PARAM`中添加以下参数：

```python
MODEL_PARAM = {
    # ... 其他参数 ...
    
    # GPU和计时参数
    "device_id": 2,          # 指定使用GPU 2
    "timing_mode": True,     # 启用计时分析模式  
    "use_amp": False,        # 禁用混合精度（解决GPU问题）
}
```

## 📋 使用方法

### 方法1：正常训练（修复GPU问题）
```bash
# 使用修复后的配置进行正常训练
python experiments/train.py -c RAST/train_PEMS04.py -g "2"
```

### 方法2：计时分析模式
```bash
# 使用计时分析配置
python experiments/train.py -c RAST/train_PEMS04_timing.py -g "2"
```

### 方法3：快速测试
```bash
# 运行快速测试脚本
cd BasicTS
python RAST/test_gpu_timing.py
```

## 🔧 关键修改

1. **GPU设备强制设置** - 模型初始化时自动移动到指定GPU
2. **AMP问题修复** - 禁用混合精度避免GPU冲突
3. **详细计时分析** - 逐模块性能分析和优化建议
4. **数据设备同步** - 确保所有张量在正确设备上

---
**状态：** ✅ GPU设置已修复，⏱️ 计时分析已实现
