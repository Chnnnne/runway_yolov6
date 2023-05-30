import torch


def generate_anchors(feats, fpn_strides, grid_cell_size=5.0, grid_cell_offset=0.5,  device='cpu', is_eval=False, mode='af'):
    '''
    Generate anchors from features.
    
    return param:
    - anchor的xyxy信息：     anchors = (8400, 4)   xyxy, 原图尺度上 ，anchor的 cell_size是40个像素
    - anchor的中心点坐标：    anchor_points = (8400, 2)   
    - 记录个数的list:        n_anchors_list = [6400, 1600, 400]  
    - 记录下采样倍数的list:   stride_tensor = (8400, 1)
    '''
    # feats = [(N, 64, 80, 80), (N, 128, 40, 40), (N, 256, 20, 20)] 
    # fpn_strides = [8, 16, 32]
    anchors = []
    anchor_points = []
    stride_tensor = []
    num_anchors_list = []
    assert feats is not None
    if is_eval:
        for i, stride in enumerate(fpn_strides):
            _, _, h, w = feats[i].shape
            shift_x = torch.arange(end=w, device=device) + grid_cell_offset
            shift_y = torch.arange(end=h, device=device) + grid_cell_offset
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            anchor_point = torch.stack(
                    [shift_x, shift_y], axis=-1).to(torch.float)
            if mode == 'af': # anchor-free
                anchor_points.append(anchor_point.reshape([-1, 2]))
                stride_tensor.append(
                torch.full(
                    (h * w, 1), stride, dtype=torch.float, device=device))
            elif mode == 'ab': # anchor-based
                anchor_points.append(anchor_point.reshape([-1, 2]).repeat(3,1))
                stride_tensor.append(
                    torch.full(
                        (h * w, 1), stride, dtype=torch.float, device=device).repeat(3,1))
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        return anchor_points, stride_tensor
    else:
        for i, stride in enumerate(fpn_strides): # i = 0, stride = 8 
            _, _, h, w = feats[i].shape # h = 80, w = 80
            cell_half_size = grid_cell_size * stride * 0.5 # cell_half_size = 5 * 8 * 0.5 = 20 
            shift_x = (torch.arange(end=w, device=device) + grid_cell_offset) * stride # （[0, 1, 2, ..., 79] + 0.5） * 8  = (4, 12, 20, ... 636)
            shift_y = (torch.arange(end=h, device=device) + grid_cell_offset) * stride #  (4, 12, 20, ... 636)
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x) # shape: (80, 80)
            anchor = torch.stack(
                [
                    shift_x - cell_half_size, shift_y - cell_half_size,
                    shift_x + cell_half_size, shift_y + cell_half_size
                ],
                axis=-1).clone().to(feats[0].dtype) # shape(80, 80),(80, 80), (80, 80), (80, 80) ----stack----> (80, 80, 4)
            anchor_point = torch.stack(
                [shift_x, shift_y], axis=-1).clone().to(feats[0].dtype) # shape(80, 80),(80, 80) ---stack---> (80,80,2)= anchor_point

            if mode == 'af': # anchor-free
                anchors.append(anchor.reshape([-1, 4]))
                anchor_points.append(anchor_point.reshape([-1, 2]))
            elif mode == 'ab': # anchor-based
                anchors.append(anchor.reshape([-1, 4]).repeat(3,1))
                anchor_points.append(anchor_point.reshape([-1, 2]).repeat(3,1))
            num_anchors_list.append(len(anchors[-1]))
            stride_tensor.append(
                torch.full(
                    [num_anchors_list[-1], 1], stride, dtype=feats[0].dtype)) 
        anchors = torch.cat(anchors)
        anchor_points = torch.cat(anchor_points).to(device)
        stride_tensor = torch.cat(stride_tensor).to(device)
        return anchors, anchor_points, num_anchors_list, stride_tensor

