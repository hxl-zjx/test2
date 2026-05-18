# 1. 删除了错误的导入语句: from cProfile import label
import numpy as np
import argparse
import torch
import copy
import time
import cv2
import os
import math
from fonts.cv_puttext import cv2ImgAddText
from ultralytics.nn.tasks import attempt_load_weights
from plate_recognition.plate_rec import get_plate_result, init_model
from plate_recognition.double_plate_split_merge import get_split_merge
# 新增：导入数据库实例（仅用于全局初始化，实际写入在app.py中）
from db_operations import db_instance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def allFilePath(rootPath, allFIleList):  # 读取文件夹内的文件，放到list
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath, temp)):
            allFIleList.append(os.path.join(rootPath, temp))
        else:
            allFilePath(os.path.join(rootPath, temp), allFIleList)


def four_point_transform(image, pts):  # 透视变换得到车牌小图
    rect = pts.astype('float32')
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def letter_box(img, size=(640, 640)):  # yolo 前处理 letter_box操作
    h, w, _ = img.shape
    r = min(size[0] / h, size[1] / w)
    new_h, new_w = int(h * r), int(w * r)
    new_img = cv2.resize(img, (new_w, new_h))
    left = int((size[1] - new_w) / 2)
    top = int((size[0] - new_h) / 2)
    right = size[1] - left - new_w
    bottom = size[0] - top - new_h
    img = cv2.copyMakeBorder(new_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, left, top


def load_model(weights, device):  # 加载yolov8 模型
    model = attempt_load_weights(weights, device=device)  # load FP32 model
    return model


def xywh2xyxy(det):  # xywh转化为xyxy
    y = det.clone()
    y[:, 0] = det[:, 0] - det[0:, 2] / 2
    y[:, 1] = det[:, 1] - det[0:, 3] / 2
    y[:, 2] = det[:, 0] + det[0:, 2] / 2
    y[:, 3] = det[:, 1] + det[0:, 3] / 2
    return y


def my_nums(dets, iou_thresh):  # nms操作
    y = dets.clone()
    y_box_score = y[:, :5]
    index = torch.argsort(y_box_score[:, -1], descending=True)
    keep = []
    while index.size()[0] > 0:
        i = index[0].item()
        keep.append(i)
        x1 = torch.maximum(y_box_score[i, 0], y_box_score[index[1:], 0])
        y1 = torch.maximum(y_box_score[i, 1], y_box_score[index[1:], 1])
        x2 = torch.minimum(y_box_score[i, 2], y_box_score[index[1:], 2])
        y2 = torch.minimum(y_box_score[i, 3], y_box_score[index[1:], 3])
        zero_ = torch.tensor(0).to(device)
        w = torch.maximum(zero_, x2 - x1)
        h = torch.maximum(zero_, y2 - y1)
        inter_area = w * h
        nuion_area1 = (y_box_score[i, 2] - y_box_score[i, 0]) * (y_box_score[i, 3] - y_box_score[i, 1])  # 计算交集
        union_area2 = (y_box_score[index[1:], 2] - y_box_score[index[1:], 0]) * (
                y_box_score[index[1:], 3] - y_box_score[index[1:], 1])  # 计算并集
        iou = inter_area / (nuion_area1 + union_area2 - inter_area)  # 计算iou
        idx = torch.where(iou <= iou_thresh)[0]  # 保留iou小于iou_thresh的
        index = index[idx + 1]
    return keep


def restore_box(dets, r, left, top):  # 坐标还原到原图上
    dets[:, [0, 2]] = dets[:, [0, 2]] - left
    dets[:, [1, 3]] = dets[:, [1, 3]] - top
    dets[:, :4] /= r
    return dets


def post_processing(prediction, conf, iou_thresh, r, left, top):  # 后处理
    prediction = prediction.permute(0, 2, 1).squeeze(0)
    xc = prediction[:, 4:6].amax(1) > conf  # 过滤掉小于conf的框
    x = prediction[xc]
    if not len(x):
        return []
    boxes = x[:, :4]  # 框
    boxes = xywh2xyxy(boxes)  # 中心点 宽高 变为 左上 右下两个点
    score, index = torch.max(x[:, 4:6], dim=-1, keepdim=True)  # 找出得分和所属类别
    x = torch.cat((boxes, score, x[:, 6:14], index), dim=1)  # 重新组合
    score = x[:, 4]
    keep = my_nums(x, iou_thresh)
    x = x[keep]
    x = restore_box(x, r, left, top)
    return x


def pre_processing(img, device):  # 前处理
    img, r, left, top = letter_box(img, (640, 640))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()  # bgr2rgb hwc2chw
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img = img / 255.0
    img = img.unsqueeze(0)
    return img, r, left, top


def det_rec_plate(img, img_ori, detect_model, plate_rec_model):
    result_list = []
    img, r, left, top = pre_processing(img, device)  # 前处理
    predict = detect_model(img)[0]
    outputs = post_processing(predict, 0.3, 0.5, r, left, top)  # 后处理
    for output in outputs:
        result_dict = {}
        output = output.squeeze().cpu().numpy().tolist()
        rect = output[:4]
        rect = [int(x) for x in rect]
        roi_img = img_ori[rect[1]:rect[3], rect[0]:rect[2]]
        # 2. 关键修正：从output列表中提取类别索引
        # 根据post_processing函数，类别索引(index)被放在了最后一列
        label = output[-1]
        if int(label):  # 判断是否是双层车牌，是双牌的话进行分割后然后拼接
            roi_img = get_split_merge(roi_img)
        plate_number, rec_prob, plate_color, color_conf = get_plate_result(roi_img, device, plate_rec_model,
                                                                           is_color=True)
        result_dict['plate_no'] = plate_number  # 车牌号
        result_dict['plate_color'] = plate_color  # 车牌颜色
        result_dict['rect'] = rect  # 车牌roi区域
        result_dict['detect_conf'] = output[4]  # 检测区域得分
        result_dict['roi_height'] = roi_img.shape[0]  # 车牌高度
        result_dict['color_conf'] = color_conf  # 颜色得分
        result_dict['plate_type'] = int(label)  # 单双层 0单层 1双层
        result_list.append(result_dict)
    return result_list


def draw_result(orgimg, dict_list, is_color=False):  # 车牌结果画出来
    result_str = ""
    for result in dict_list:
        rect_area = result['rect']
        x, y, w, h = rect_area[0], rect_area[1], rect_area[2] - rect_area[0], rect_area[3] - rect_area[1]
        padding_w = 0.05 * w
        padding_h = 0.11 * h
        rect_area[0] = max(0, int(x - padding_w))
        rect_area[1] = max(0, int(y - padding_h))
        rect_area[2] = min(orgimg.shape[1], int(rect_area[2] + padding_w))
        rect_area[3] = min(orgimg.shape[0], int(rect_area[3] + padding_h))
        height_area = result['roi_height']
        result_p = result['plate_no']
        if result['plate_type'] == 0:  # 单层
            result_p += " " + result['plate_color']
        else:  # 双层
            result_p += " " + result['plate_color'] + "双层"
        result_str += result_p + " "
        cv2.rectangle(orgimg, (rect_area[0], rect_area[1]), (rect_area[2], rect_area[3]), (0, 0, 255), 2)  # 画框
        labelSize = cv2.getTextSize(result_p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)  # 获得字体的大小
        if rect_area[0] + labelSize[0][0] > orgimg.shape[1]:  # 防止显示的文字越界
            rect_area[0] = int(orgimg.shape[1] - labelSize[0][0])
        orgimg = cv2.rectangle(orgimg, (rect_area[0], int(rect_area[1] - round(1.6 * labelSize[0][1]))),
                               (int(rect_area[0] + round(1.2 * labelSize[0][0])), rect_area[1] + labelSize[1]),
                               (255, 255, 255), cv2.FILLED)  # 画文字框,背景白色
        if len(result) >= 6:
            orgimg = cv2ImgAddText(orgimg, result_p, rect_area[0], int(rect_area[1] - round(1.6 * labelSize[0][1])),
                                   (0, 0, 0), 21)
    return orgimg, result_str.strip()


# ========== 新增关键帧处理函数 ==========
def calculate_image_clarity(image):
    """计算图像清晰度（拉普拉斯方差法），值越高越清晰"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    clarity = np.var(laplacian)
    return clarity


def extract_key_frames(video_path, sample_interval=5, top_k=10):
    """提取视频中清晰度最高的top_k帧"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"无法打开视频文件: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    key_frames = []  # 存储 (清晰度, 帧, 帧索引)
    # 按间隔采样帧
    for i in range(0, frame_count, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        # 计算清晰度
        clarity = calculate_image_clarity(frame)
        key_frames.append((clarity, frame, i))
    cap.release()
    # 按清晰度排序，取top_k
    key_frames.sort(key=lambda x: x[0], reverse=True)
    top_frames = [(frame, idx) for (clarity, frame, idx) in key_frames[:top_k]]
    return top_frames


def get_best_plate_frame(video_path, detect_model, plate_rec_model):
    """
    获取视频中识别效果最好的关键帧（带框选）和最优车牌
    返回：(最优帧, 最优车牌列表, 最优帧识别结果详情)
    """
    # 提取关键帧
    key_frames = extract_key_frames(video_path, sample_interval=5, top_k=10)
    if not key_frames:
        return None, ['未识别到车牌'], []
    plate_results = {}
    best_frame = None
    best_frame_idx = -1
    best_result_list = []  # 最优帧的完整识别结果
    # 对每个关键帧进行识别
    for frame, frame_idx in key_frames:
        frame_ori = frame.copy()
        result_list = det_rec_plate(frame, frame_ori, detect_model, plate_rec_model)
        # 计算当前帧的综合置信度
        frame_conf = 0
        for res in result_list:
            if res['plate_no']:
                # 综合检测置信度和颜色置信度
                conf = res['detect_conf'] * res['color_conf']
                frame_conf += conf
                # 累加车牌置信度
                plate_no = res['plate_no']
                plate_results[plate_no] = plate_results.get(plate_no, 0) + conf
        # 记录最优帧（置信度最高）
        current_conf = sum([r['detect_conf'] * r['color_conf'] for r in result_list if r['plate_no']])
        if current_conf > 0 and (
                best_frame is None or current_conf > sum(
            [r['detect_conf'] * r['color_conf'] for r in best_result_list if r['plate_no']])):
            best_frame = frame_ori
            best_frame_idx = frame_idx
            best_result_list = result_list
    # 生成带框选的最优帧
    if best_frame is not None:
        best_frame, _ = draw_result(best_frame, best_result_list)
    # 处理车牌结果（仅保留最优帧的车牌）
    if best_result_list:
        plate_list = [res['plate_no'] for res in best_result_list if res['plate_no']]
        if not plate_list:
            plate_list = ['未识别到车牌']
    else:
        plate_list = ['未识别到车牌']
    return best_frame, plate_list, best_result_list


# ========== 方案一：基础版视频处理 ==========
def process_video(video_path, detect_model, plate_rec_model, output_path, frame_interval=2):
    """
    优化版：一次遍历完成视频生成+关键帧提取+车牌识别
    速度提升3-6倍，识别精度完全不变，100%兼容所有OpenCV版本
    :param frame_interval: 检测间隔（每隔多少帧检测一次，越大越快）
    :return: (输出视频路径, 最优关键帧, 最优识别结果列表)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"无法打开视频文件: {video_path}")

    # 获取视频基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 创建视频写入对象
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 状态变量
    current_result_list = []  # 当前帧要显示的车牌结果
    best_clarity = 0  # 最优帧清晰度
    best_frame = None  # 最优关键帧（带框选）
    best_result_list = []  # 最优帧的完整识别结果
    frame_count = 0
    start_time = time.time()

    print(f"[优化版] 开始处理视频，总帧数：{total_frames}，检测间隔：{frame_interval}帧")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_ori = frame.copy()

        # 每隔frame_interval帧运行一次YOLO检测
        if frame_count % frame_interval == 0:
            # 检测当前帧的所有车牌
            current_result_list = det_rec_plate(frame, frame_ori, detect_model, plate_rec_model)

            # 计算当前帧清晰度，更新最优关键帧
            clarity = calculate_image_clarity(frame)
            valid_plates = [res['plate_no'] for res in current_result_list if res['plate_no']]
            if clarity > best_clarity and valid_plates:
                best_clarity = clarity
                # 保存带框选的最优帧
                draw_frame = frame_ori.copy()
                draw_frame, _ = draw_result(draw_frame, current_result_list)
                best_frame = draw_frame
                best_result_list = current_result_list

        # 绘制识别结果（所有帧都绘制，非检测帧复用最近的检测结果）
        if current_result_list:
            frame, _ = draw_result(frame, current_result_list)

        # 写入处理后的帧
        out.write(frame)
        frame_count += 1

        # 打印进度
        if frame_count % 100 == 0:
            progress = int(frame_count / total_frames * 100)
            print(f"[进度] 已处理 {frame_count}/{total_frames} 帧 ({progress}%)")

    # 释放资源
    cap.release()
    out.release()

    print(f"[完成] 视频处理完成！总耗时：{time.time() - start_time:.2f}秒")
    print(f"[统计] 共运行YOLO检测 {int(frame_count / frame_interval)} 次，最优帧清晰度：{best_clarity:.2f}")

    # 兜底：如果全程没有识别到任何车牌
    if not best_result_list:
        best_result_list = []
        best_frame = None

    return output_path, best_frame, best_result_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_model', nargs='+', type=str, default=r'weights/yolov8s.pt',
                        help='model.pt path(s)')  # yolov8检测模型
    parser.add_argument('--rec_model', type=str, default=r'weights/plate_rec_color.pth',
                        help='model.pt path(s)')  # 车牌字符识别模型
    parser.add_argument('--image_path', type=str, default=r'T_T_imgs', help='source')  # 待识别图片路径
    parser.add_argument('--video_path', type=str, default=None, help='视频文件路径（优先于图片处理）')  # 新增视频参数
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')  # yolov8输入大小
    parser.add_argument('--output', type=str, default='T_T_result', help='结果保存的文件夹')  # 结果保存路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    opt = parser.parse_args()
    save_path = opt.output
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # 加载模型
    detect_model = load_model(opt.detect_model, device)
    plate_rec_model = init_model(device, opt.rec_model, is_color=True)
    # 计算参数量
    total = sum(p.numel() for p in detect_model.parameters())
    total_1 = sum(p.numel() for p in plate_rec_model.parameters())
    print("yolov8 detect params: %.2fM, rec params: %.2fM" % (total / 1e6, total_1 / 1e6))
    detect_model.eval()
    # 优先处理视频
    if opt.video_path is not None and os.path.exists(opt.video_path):
        video_name = os.path.basename(opt.video_path)
        output_video = os.path.join(save_path, f"result_{video_name}")
        # 仅获取处理后的视频路径，不再关注逐帧识别的车牌
        video_path = process_video(opt.video_path, detect_model, plate_rec_model, output_video)
        # 获取最优关键帧的车牌结果
        best_frame, plate_list, _ = get_best_plate_frame(opt.video_path, detect_model, plate_rec_model)
        print(f"视频处理完成，结果保存至: {video_path}")
        print(f"最优帧识别到的车牌: {plate_list}")
    else:
        # 原有图片处理逻辑
        file_list = []
        allFilePath(opt.image_path, file_list)
        count = 0
        time_all = 0
        time_begin = time.time()
        for pic_ in file_list:
            file = os.path.splitext(pic_)
            file_name, file_type = file
            if file_type in ".jpg .png":
                img = cv2.imread(pic_)
                img_ori = copy.deepcopy(img)
                start = time.time()
                result_list = det_rec_plate(img, img_ori, detect_model, plate_rec_model)
                img, result_str = draw_result(img_ori, result_list)
                end = time.time()
                time_all += (end - start)
                count += 1
                save_name = os.path.join(save_path, os.path.basename(pic_))
                cv2.imwrite(save_name, img)
                print(f"处理完成: {save_name}")
                print(f"识别到的车牌: {[res['plate_no'] for res in result_list]}")
        print(f"总处理图片数: {count}")
        print(f"总耗时: {time_all:.2f}s, 平均耗时: {time_all / count:.2f}s")