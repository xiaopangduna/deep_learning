def check_model_equivalence(my_model, official_model, input_shape=(1, 3, 224, 224), atol=1e-6):
    import torch
    
    # Step 1: 参数形状检查
    my_state = my_model.state_dict()
    off_state = official_model.state_dict()
    if [p.shape for p in my_state.values()] != [p.shape for p in off_state.values()]:
        return False, "参数形状不一致"
    
    # Step 2: 尝试加载官方权重
    try:
        my_model.load_state_dict(off_state)
    except RuntimeError as e:
        return False, f"加载权重失败: {e}"
    
    # Step 3: 前向输出比对
    x = torch.randn(input_shape)
    y_my = my_model(x)
    y_off = official_model(x)
    
    if torch.allclose(y_my, y_off, atol=atol):
        return True, "完全一致 ✅"
    else:
        return False, "forward 输出不一致 ❌"
