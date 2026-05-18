# -*- coding: utf-8 -*-
import os
import sys
import cv2
import time
import torch
import copy
import numpy as np
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
# 新增：导入缓存和数据库
from flask_caching import Cache
from db_operations import db_instance
from db_config import CACHE_CONFIG

# 导入detect_rec_plate中的核心函数（仅保留方案一相关）
from detect_rec_plate import (
    load_model, init_model, det_rec_plate, draw_result,
    process_video, device as drp_device, get_best_plate_frame
)

# 初始化Flask应用
app = Flask(__name__, static_folder='.')
# 初始化缓存
cache = Cache(app, config=CACHE_CONFIG)

# 配置项
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
FRAME_FOLDER = 'frames'  # 新增：存储视频关键帧
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov'}
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100MB

# 创建必要文件夹
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)  # 创建关键帧存储文件夹

# 设备配置
device = drp_device

# 模型路径配置
DETECT_MODEL_PATH = os.path.join('weights', 'yolov8s.pt')
REC_MODEL_PATH = os.path.join('weights', 'plate_rec_color.pth')

# 全局加载模型（全部改为英文，彻底避免乱码）
print("Loading license plate detection and recognition models...")
try:
    detect_model = load_model(DETECT_MODEL_PATH, device)
    plate_rec_model = init_model(device, REC_MODEL_PATH, is_color=True)
    detect_model.eval()
    print("Models loaded successfully!")
except Exception as e:
    print(f"Model loading failed: {str(e)}")
    raise

# 辅助函数：检查文件扩展名是否合法
def allowed_file(filename, file_type):
    if file_type == 'image':
        allowed = ALLOWED_IMAGE_EXTENSIONS
    elif file_type == 'video':
        allowed = ALLOWED_VIDEO_EXTENSIONS
    else:
        return False
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed

# 主页路由
@app.route('/')
def index():
    """Return frontend page"""
    try:
        return send_file('index.html')
    except Exception as e:
        return f"index.html not found: {str(e)}", 404

# 上传识别接口（支持图片/视频）
@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle image/video upload and return recognition result"""
    try:
        # 检查文件是否存在
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected', 'success': False}), 400
        file = request.files['file']
        # 自动检测文件类型（修复之前的文件类型报错）
        filename = file.filename
        ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        if ext in ALLOWED_IMAGE_EXTENSIONS:
            file_type = 'image'
        elif ext in ALLOWED_VIDEO_EXTENSIONS:
            file_type = 'video'
        else:
            return jsonify({
                'error': f'Unsupported file format. Supported: {",".join(ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS)}',
                'success': False
            }), 400

        if filename == '':
            return jsonify({'error': 'Empty filename', 'success': False}), 400

        # 校验文件大小
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        if file_type == 'image' and file_size > MAX_IMAGE_SIZE:
            return jsonify({'error': 'Image size exceeds 5MB limit', 'success': False}), 400
        elif file_type == 'video' and file_size > MAX_VIDEO_SIZE:
            return jsonify({'error': 'Video size exceeds 100MB limit', 'success': False}), 400

        # 保存上传文件
        filename = secure_filename(filename)
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)

        start_time = time.time()
        result_filename = f"result_{int(time.time())}_{filename}"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        plate_list = []
        frame_url = ""  # 视频关键帧URL
        avg_confidence = 0.0  # 平均置信度

        # 处理图片
        if file_type == 'image':
            img = cv2.imread(upload_path)
            if img is None:
                return jsonify({'error': 'Cannot read image file', 'success': False}), 400
            img_ori = copy.deepcopy(img)
            result_list = det_rec_plate(img, img_ori, detect_model, plate_rec_model)
            result_img, result_str = draw_result(img_ori, result_list)
            cv2.imwrite(result_path, result_img)
            # 提取车牌和置信度
            plate_list = [res['plate_no'] for res in result_list if res['plate_no']]
            confidences = [res['detect_conf'] for res in result_list if res['plate_no']]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            if not plate_list:
                plate_list = ['未识别到车牌']

        # 处理视频（方案一：基础版逐帧检测）
        # 处理视频（方案一：极致优化版，速度提升3-6倍）
        elif file_type == 'video':
            try:
                # 优化：一次调用完成所有处理，不再重复处理视频
                # frame_interval=2：每隔2帧检测一次，速度提升2倍，效果几乎无损失
                # 可以调整为3（速度提升3倍）或4（速度提升4倍）
                video_save_path, best_frame, result_list = process_video(
                    upload_path, detect_model, plate_rec_model, result_path, frame_interval=2
                )

                # 保存关键帧（和原来的逻辑完全一致）
                if best_frame is not None:
                    frame_filename = f"frame_{int(time.time())}_{os.path.splitext(filename)[0]}.jpg"
                    frame_path = os.path.join(FRAME_FOLDER, frame_filename)
                    cv2.imwrite(frame_path, best_frame)
                    frame_url = f'/frames/{frame_filename}'
                else:
                    frame_url = ""

                # 提取车牌和置信度（和原来的逻辑完全一致）
                plate_list = [res['plate_no'] for res in result_list if res['plate_no']]
                confidences = [res['detect_conf'] for res in result_list if res['plate_no']]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

                # 兜底：如果未识别到车牌
                if not plate_list:
                    plate_list = ['未识别到车牌']

            except Exception as e:
                return jsonify({'error': f'Video processing failed: {str(e)}', 'success': False}), 500

        # 计算处理时间
        process_time = round(time.time() - start_time, 2)

        # 新增：写入数据库
        if plate_list != ['未识别到车牌']:
            # 将多个车牌用逗号分隔存储
            plate_number_str = ','.join(plate_list)
            # 插入数据库
            db_instance.insert_record(
                file_name=filename,
                file_path=upload_path,
                plate_number=plate_number_str,
                confidence=avg_confidence,
                file_type=file_type,
                result_image_path=result_path if file_type == 'image' else frame_url
            )
            # 清除历史记录缓存，保证数据最新
            cache.delete('latest_records')
            print(f"[成功] 识别记录已存入数据库：{plate_number_str}")

        # 返回结果（新增frame_url字段）
        return jsonify({
            'file_url': f'/results/{result_filename}',
            'frame_url': frame_url,  # 视频关键帧URL（图片识别为空）
            'plate': plate_list,
            'success': True,
            'process_time': process_time,
            'file_type': file_type
        })
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}', 'success': False}), 500

# 新增：获取最新识别记录接口（带缓存）
@app.route('/api/records', methods=['GET'])
def get_records():
    """获取最新20条识别记录"""
    records = cache.get('latest_records')
    if not records:
        records = db_instance.get_all_records(limit=20)
        cache.set('latest_records', records, timeout=3600)
        print("[成功] 从数据库获取记录并缓存")
    else:
        print("[成功] 从缓存获取记录")
    return jsonify(records)

# 新增：根据车牌查询记录接口
@app.route('/api/records/<plate_number>', methods=['GET'])
def get_record_by_plate(plate_number):
    """根据车牌号码查询历史记录"""
    records = db_instance.get_record_by_plate(plate_number)
    return jsonify(records)

# 提供结果文件访问（图片/视频）
@app.route('/results/<filename>')
def send_result(filename):
    """Return recognition result file (image/video)"""
    try:
        file_path = os.path.join(RESULT_FOLDER, filename)
        # 根据扩展名判断文件类型
        ext = filename.rsplit('.', 1)[1].lower()
        if ext in ALLOWED_VIDEO_EXTENSIONS:
            return send_file(file_path, mimetype=f'video/{ext}')
        else:
            return send_file(file_path)
    except Exception as e:
        return jsonify({'error': f'File not found: {str(e)}', 'success': False}), 404

# 新增：提供关键帧访问
@app.route('/frames/<filename>')
def send_frame(filename):
    """Return video key frame"""
    try:
        file_path = os.path.join(FRAME_FOLDER, filename)
        return send_file(file_path)
    except Exception as e:
        return jsonify({'error': f'Frame not found: {str(e)}', 'success': False}), 404

# ===================== 以下是修复后的删除接口 =====================
# 新增：删除单条识别记录（已修复，真正调用数据库）
@app.route('/api/records/<int:record_id>', methods=['DELETE'])
def delete_record(record_id):
    """删除指定ID的识别记录"""
    try:
        # 调用数据库删除方法
        success = db_instance.delete_record(record_id)
        if success:
            # 删除缓存，保证下次查询最新数据
            cache.delete('latest_records')
            print(f"[成功] 已删除ID为{record_id}的识别记录")
            return jsonify({'success': True, 'msg': '记录删除成功'})
        else:
            return jsonify({'success': False, 'error': '记录不存在或已被删除'}), 404
    except Exception as e:
        print(f"[错误] 删除记录异常：{e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# 新增：清空所有识别记录（已修复，真正调用数据库）
@app.route('/api/records', methods=['DELETE'])
def clear_all_records():
    """清空所有识别记录（重置自增ID）"""
    try:
        # 调用数据库清空方法
        success = db_instance.clear_all_records()
        if success:
            # 清空缓存，保证数据一致
            cache.delete('latest_records')
            print("[成功] 所有识别记录已清空，自增ID已重置")
            return jsonify({'success': True, 'msg': '所有记录已清空'})
        else:
            return jsonify({'success': False, 'error': '清空失败'}), 500
    except Exception as e:
        print(f"[错误] 清空记录异常：{e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# 主函数
if __name__ == '__main__':
    # 配置文件大小限制
    app.config['MAX_CONTENT_LENGTH'] = MAX_VIDEO_SIZE  # 最大支持100MB
    # 检查关键文件（英文提示）
    if not os.path.exists('index.html'):
        print("Error: index.html not found in root directory!")
        exit(1)
    if not os.path.exists(DETECT_MODEL_PATH):
        print(f"Error: Detection model not found: {DETECT_MODEL_PATH}")
        exit(1)
    if not os.path.exists(REC_MODEL_PATH):
        print(f"Error: Recognition model not found: {REC_MODEL_PATH}")
        exit(1)
    # 英文启动提示
    print("License plate recognition system starting...")
    print("Support: Image (PNG/JPG/JPEG) & Video (MP4/AVI/MOV)")
    print("Access URL: http://localhost:5000")
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )