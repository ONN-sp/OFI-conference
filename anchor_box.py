import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import namedtuple

with tf.device('/gpu:1'):
    img = plt.imread('cat_dog.jpg')
    h, w = img.shape[0:2]
    print(h, w)

    def MultiBoxPrior(feature_map, sizes = [0.75, 0.5, 0.25], ratios = [1, 2, 0.5]):
        # 以像素为中心返回锚框
        # sizes:锚框为原图大小的比例;
        # ratios:宽高比;
        # 我们通常只对包含 s1 或 r1 的大小与宽高比的组合感兴趣，即(s1,r1),(s1,r2),…,(s1,rm),(s2,r1),(s3,r1),…,(sn,r1);
        # 也就是说，以相同像素为中心的锚框的数量为 n+m−1.
        pairs = [] # 保存(s, sqrt(r))
        for r in ratios:
            pairs.append([sizes[0], np.sqrt(r)])
        for s in sizes[1:]:
            pairs.append([s, np.sqrt(ratios[0])])
        pairs = np.array(pairs) # n+m-1个
        ss1 = pairs[:, 0] * pairs[:, 1] # size * sqrt(ration)
        ss2 = pairs[:, 0] / pairs[:, 1] # size / sqrt(retion)
        base_anchors = tf.stack([-ss1, -ss2, ss1, ss2], axis=1) / 2
        h, w = feature_map.shape[-2:]
        shifts_x = tf.divide(tf.range(0, w), w)
        shifts_y = tf.divide(tf.range(0, h), h)
        shift_x, shift_y = tf.meshgrid(shifts_x, shifts_y)
        shift_x = tf.reshape(shift_x, (-1,))
        shift_y = tf.reshape(shift_y, (-1,))
        shifts = tf.stack((shift_x, shift_y, shift_x, shift_y), axis=1)
        anchors = tf.add(tf.reshape(shifts, (-1,1,4)), tf.reshape(base_anchors, (1,-1,4)))
        return tf.cast(tf.reshape(anchors, (1,-1,4)), tf.float32)
    x = tf.zeros((1,3,h,w))
    y = MultiBoxPrior(x)
    print(y.shape) # shape = (1, 436900, 4),1为batch_size,436900为这张图片的锚框个数,4为锚框的左上角坐标和右下角坐标(归一化后的)
    def bbox_to_rect(bbox, color):
    # 将边界框(左上x, 左上y, 右下x, 右下y)格式转换成matplotlib格式：
    # ((左上x, 左上y), 宽, 高)
        return plt.Rectangle(
            xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
            fill=False, edgecolor=color, linewidth=2)
    # 本函数已保存在d2lzh_pytorch包中方便以后使用
    def show_bboxes(axes, bboxes, labels=None, colors=None):
        def _make_list(obj, default_values=None):
            if obj is None:
                obj = default_values
            elif not isinstance(obj, (list, tuple)):
                obj = [obj]
            return obj

        labels = _make_list(labels)
        colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
        for i, bbox in enumerate(bboxes):
            color = colors[i % len(colors)]
            rect = bbox_to_rect(bbox.numpy(), color)
            axes.add_patch(rect)
            if labels and len(labels) > i:
                text_color = 'k' if color == 'w' else 'w'
                axes.text(rect.xy[0], rect.xy[1], labels[i],
                    va='center', ha='center', fontsize=6,
                    color=text_color, bbox=dict(facecolor=color, lw=0))
    
    boxes = tf.reshape(y, (h,w,5,4))
    fig = plt.imshow(img)
    bbox_scale = tf.constant([[w,h,w,h]], dtype=tf.float32)
    show_bboxes(fig.axes, tf.multiply(boxes[200,250,:,:], bbox_scale), 
        ['s=0.75, r=1', 's=0.75, r=2', 's=0.55, r=0.5', 
        's=0.5, r=1', 's=0.25, r=1'])
    plt.show()
    def compute_intersection(set_1, set_2):
    # 计算anchor之间的交集
    # tensorflow auto-broadcasts singleton dimensions
        lower_bounds = tf.maximum(tf.expand_dims(set_1[:,:2], axis=1), tf.expand_dims(set_2[:,:2], axis=0)) # (n1, n2, 2)
        upper_bounds = tf.minimum(tf.expand_dims(set_1[:,2:], axis=1), tf.expand_dims(set_2[:,2:], axis=0)) # (n1, n2, 2)
        # 设置最小值
        intersection_dims = tf.clip_by_value(upper_bounds - lower_bounds, clip_value_min=0, clip_value_max=3) # (n1, n2, 2)
        return tf.multiply(intersection_dims[:, :, 0], intersection_dims[:, :, 1]) # (n1, n2)
    def compute_jaccard(set_1, set_2):
        """
        计算anchor之间的Jaccard系数(IoU)
        Args:
            set_1: a tensor of dimensions (n1, 4), anchor表示成(xmin, ymin, xmax, ymax)
            set_2: a tensor of dimensions (n2, 4), anchor表示成(xmin, ymin, xmax, ymax)
        Returns:
            Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, shape: (n1, n2)
        """
        # Find intersections
        intersection = compute_intersection(set_1, set_2)
        # Find areas of each box in both sets
        areas_set_1 = tf.multiply(tf.subtract(set_1[:, 2], set_1[:, 0]), tf.subtract(set_1[:, 3], set_1[:, 1]))  # (n1)
        areas_set_2 = tf.multiply(tf.subtract(set_2[:, 2], set_2[:, 0]), tf.subtract(set_2[:, 3], set_2[:, 1]))  # (n2)
        # Find the union
        union = tf.add(tf.expand_dims(areas_set_1, axis=1), tf.expand_dims(areas_set_2, axis=0))  # (n1, n2)
        union = tf.subtract(union, intersection)  # (n1, n2)
        return tf.divide(intersection, union) #(n1, n2)
    def assign_anchor(bb, anchor, jaccard_threshold=0.5):
        """
        # 按照「9.4.1. 生成多个锚框」图9.3所讲为每个anchor分配真实的bb, anchor表示成归一化(xmin, ymin, xmax, ymax).
        https://zh.d2l.ai/chapter_computer-vision/anchor.html
        Args:
            bb: 真实边界框(bounding box), shape:(nb, 4)
            anchor: 待分配的anchor, shape:(na, 4)
            jaccard_threshold: 预先设定的阈值
        Returns:
            assigned_idx: shape: (na, ), 每个anchor分配的真实bb对应的索引, 若未分配任何bb则为-1
        """
        na = anchor.shape[0]
        nb = bb.shape[0]
        jaccard = compute_jaccard(anchor, bb).numpy()   # shape: (na, nb)
        assigned_idx = np.ones(na) * -1 # 初始全为-1
        # 先为每个bb分配一个anchor（不要求满足jaccard_threshold）
        jaccard_cp = jaccard.copy()
        for j in range(nb):
            i = np.argmax(jaccard_cp[:, j])
            assigned_idx[i] = j
            jaccard_cp[i, :] = float("-inf")    # 赋值为负无穷, 相当于去掉这一行

        # 处理还未被分配的anchor， 要求满足jaccard_threshold
        for i in range(na):
            if assigned_idx[i] == -1:
                j = np.argmax(jaccard[i, :])
                if jaccard[i, j] >= jaccard_threshold:
                    assigned_idx[i] = j
        return tf.cast(assigned_idx, tf.int32)
    def xy_to_cxcy(xy):
        """
        将(x_min, y_min, x_max, y_max)形式的anchor转换成(center_x, center_y, w, h)形式的.
        https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py
        Args:
            xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
        Returns: 
            bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
        """
        return tf.concat(((xy[:, 2:] + xy[:, :2]) / 2,  #c_x, c_y
                xy[:, 2:] - xy[:, :2]), axis=1)
    def MultiBoxTarget(anchor, label):
        """
        # 按照「9.4.1. 生成多个锚框」所讲的实现, anchor表示成归一化(xmin, ymin, xmax, ymax).
        https://zh.d2l.ai/chapter_computer-vision/anchor.html
        Args:
            anchor: torch tensor, 输入的锚框, 一般是通过MultiBoxPrior生成, shape:（1，锚框总数，4）
            label: 真实标签, shape为(bn, 每张图片最多的真实锚框数, 5)
                第二维中，如果给定图片没有这么多锚框, 可以先用-1填充空白, 最后一维中的元素为[类别标签, 四个坐标值]
        Returns:
            列表, [bbox_offset, bbox_mask, cls_labels]
            bbox_offset: 每个锚框的标注偏移量，形状为(bn，锚框总数*4)
            bbox_mask: 形状同bbox_offset, 每个锚框的掩码, 一一对应上面的偏移量, 负类锚框(背景)对应的掩码均为0, 正类锚框的掩码均为1
            cls_labels: 每个锚框的标注类别, 其中0表示为背景, 形状为(bn，锚框总数)
        """
        assert len(anchor.shape) == 3 and len(label.shape) == 3
        bn = label.shape[0]

        def MultiBoxTarget_one(anchor, label, eps=1e-6):
            """
            MultiBoxTarget函数的辅助函数, 处理batch中的一个
            Args:
                anchor: shape of (锚框总数, 4)
                label: shape of (真实锚框数, 5), 5代表[类别标签, 四个坐标值]
                eps: 一个极小值, 防止log0
            Returns:
                offset: (锚框总数*4, )
                bbox_mask: (锚框总数*4, ), 0代表背景, 1代表非背景
                cls_labels: (锚框总数, 4), 0代表背景
            """
            an = anchor.shape[0]
            assigned_idx = assign_anchor(label[:, 1:], anchor) ## (锚框总数, )
            # 决定anchor留下或者舍去
            bbox_mask = tf.repeat(tf.expand_dims(tf.cast((assigned_idx >= 0), dtype=tf.double), axis=-1), repeats=4, axis=1)

            cls_labels = np.zeros(an, dtype=int) # 0表示背景
            assigned_bb = np.zeros((an, 4), dtype=float) # 所有anchor对应的bb坐标
            for i in range(an):
                bb_idx = assigned_idx[i]
                if bb_idx >= 0: # 即非背景
                    cls_labels[i] = label.numpy()[bb_idx, 0] + 1 # 要注意加1
                    assigned_bb[i, :] = label.numpy()[bb_idx, 1:]

            center_anchor = tf.cast(xy_to_cxcy(anchor), dtype=tf.double)  # (center_x, center_y, w, h)
            center_assigned_bb = tf.cast(xy_to_cxcy(assigned_bb), dtype=tf.double) # (center_x, center_y, w, h)

            offset_xy = 10.0 * (center_assigned_bb[:,:2] - center_anchor[:,:2]) / center_anchor[:,2:]
            offset_wh = 5.0 * tf.math.log(eps + center_assigned_bb[:, 2:] / center_anchor[:, 2:])
            offset = tf.multiply(tf.concat((offset_xy, offset_wh), axis=1), bbox_mask)    # (锚框总数, 4)

            return tf.reshape(offset, (-1,)), tf.reshape(bbox_mask, (-1,)), cls_labels

        batch_offset = []
        batch_mask = []
        batch_cls_labels = []
        for b in range(bn):
            offset, bbox_mask, cls_labels = MultiBoxTarget_one(anchor[0, :, :], label[b,:,:])

            batch_offset.append(offset)
            batch_mask.append(bbox_mask)
            batch_cls_labels.append(cls_labels)

        batch_offset = tf.convert_to_tensor(batch_offset)
        batch_mask = tf.convert_to_tensor(batch_mask)
        batch_cls_labels = tf.convert_to_tensor(batch_cls_labels)

        return [batch_offset, batch_mask, batch_cls_labels]
    Pred_BB_Info = namedtuple("Pred_BB_Info", 
        ["index", "class_id", "confidence", "xyxy"])
    def non_max_suppression(bb_info_list, nms_threshold=0.5):
        """
        非极大抑制处理预测的边界框
        Args:
            bb_info_list: Pred_BB_Info的列表, 包含预测类别、置信度等信息
            nms_threshold: 阈值
        Returns:
            output: Pred_BB_Info的列表, 只保留过滤后的边界框信息
        """
        output = []
        # 现根据置信度从高到底排序
        sorted_bb_info_list = sorted(bb_info_list,
                        key = lambda x: x.confidence, 
                        reverse=True)
        while len(sorted_bb_info_list) != 0:
            best = sorted_bb_info_list.pop(0)
            output.append(best)

            if len(sorted_bb_info_list) == 0:
                break
            bb_xyxy = []
            for bb in sorted_bb_info_list:
                bb_xyxy.append(bb.xyxy)

            iou = compute_jaccard(tf.convert_to_tensor(best.xyxy),
                        tf.squeeze(tf.convert_to_tensor(bb_xyxy), axis=1))[0] # shape: (len(sorted_bb_info_list), )
            n = len(sorted_bb_info_list)
            sorted_bb_info_list = [
                        sorted_bb_info_list[i] for i in 
                        range(n) if iou[i] <= nms_threshold]
        return output
    def MultiBoxDetection(cls_prob, loc_pred, anchor, nms_threshold=0.5):
        """
        # 按照「9.4.1. 生成多个锚框」所讲的实现, anchor表示成归一化(xmin, ymin, xmax, ymax).
        https://zh.d2l.ai/chapter_computer-vision/anchor.html
        Args:
            cls_prob: 经过softmax后得到的各个锚框的预测概率, shape:(bn, 预测总类别数+1, 锚框个数)
            loc_pred: 预测的各个锚框的偏移量, shape:(bn, 锚框个数*4)
            anchor: MultiBoxPrior输出的默认锚框, shape: (1, 锚框个数, 4)
            nms_threshold: 非极大抑制中的阈值
        Returns:
            所有锚框的信息, shape: (bn, 锚框个数, 6)
            每个锚框信息由[class_id, confidence, xmin, ymin, xmax, ymax]表示
            class_id=-1 表示背景或在非极大值抑制中被移除了
        """
        assert len(cls_prob.shape) == 3 and len(loc_pred.shape) == 2 and len(anchor.shape) == 3
        bn = cls_prob.shape[0]
        def MultiBoxDetection_one(c_p, l_p, anc, nms_threshold=0.5):
            """
            MultiBoxDetection的辅助函数, 处理batch中的一个
            Args:
                c_p: (预测总类别数+1, 锚框个数)
                l_p: (锚框个数*4, )
                anc: (锚框个数, 4)
                nms_threshold: 非极大抑制中的阈值
            Return:
                output: (锚框个数, 6)
            """
            pred_bb_num = c_p.shape[1]
            # 加上偏移量
            anc = tf.add(anc, tf.reshape(l_p, (pred_bb_num, 4))).numpy()

            # 最大的概率
            confidence = tf.reduce_max(c_p, axis=0)
            # 最大概率对应的id
            class_id = tf.argmax(c_p, axis=0)
            confidence = confidence.numpy()
            class_id = class_id.numpy()

            pred_bb_info = [Pred_BB_Info(index=i,
                        class_id=class_id[i]-1,
                        confidence=confidence[i],
                        xyxy=[anc[i]]) # xyxy是个列表
                    for i in range(pred_bb_num)]
            # 正类的index
            obj_bb_idx = [bb.index for bb 
                    in non_max_suppression(pred_bb_info,
                                nms_threshold)]
            output = []
            for bb in pred_bb_info:
                output.append(np.append([
                    (bb.class_id if bb.index in obj_bb_idx 
                            else -1.0),
                    bb.confidence],
                    bb.xyxy))

            return tf.convert_to_tensor(output) # shape: (锚框个数， 6)

        batch_output = []
        for b in range(bn):
            batch_output.append(MultiBoxDetection_one(cls_prob[b],
                            loc_pred[b], anchor[0],
                            nms_threshold))

        return tf.convert_to_tensor(batch_output)
    anchors = tf.convert_to_tensor([[0.1, 0.08, 0.52, 0.92],
                [0.08, 0.2, 0.56, 0.95],
                [0/15, 0.3, 0.62, 0.91],
                [0.55, 0.2, 0.9, 0.88]])
    offset_preds = tf.convert_to_tensor([0.0] * (4 * len(anchors)))
    cls_probs = tf.convert_to_tensor([[0., 0., 0., 0.], # 背景的预测概率
                    [0.9, 0.8, 0.7, 0.1],    # 狗的预测概率
                    [0.1, 0.2, 0.3, 0.9]])   # 猫的预测概率
    output = MultiBoxDetection(
    tf.expand_dims(cls_probs, 0),
    tf.expand_dims(offset_preds, 0),
    tf.expand_dims(anchors, 0),
    nms_threshold=0.5)
    print(output)
    # 移除掉类别为-1的预测边界框,并可视化非极大值抑制保留的结果
    fig = plt.imshow(img)
    for i in output[0].numpy():
        if i[0] == -1:
            continue
        label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
        show_bboxes(fig.axes, tf.multiply(i[2:], bbox_scale), label)
    plt.show()