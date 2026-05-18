# db_operations.py - 数据库CRUD操作类（已添加自动创建数据库功能，修复GBK编码错误）
import pymysql
from typing import Optional, List, Dict
from db_config import MYSQL_CONFIG


class PlateDB:
    def __init__(self):
        self.conn = None
        self.cursor = None
        self._connect_and_create_db()  # 修改：先连接并创建数据库

    def _connect_and_create_db(self):
        """先连接MySQL服务器（不指定数据库），自动创建不存在的数据库"""
        try:
            # 第一步：不指定数据库名连接MySQL服务器
            temp_config = MYSQL_CONFIG.copy()
            db_name = temp_config.pop('database')  # 临时移除数据库名

            self.conn = pymysql.connect(**temp_config)
            self.cursor = self.conn.cursor(pymysql.cursors.DictCursor)

            # 第二步：检查数据库是否存在，不存在则创建
            self.cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS {db_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            print(f"[成功] 数据库 {db_name} 检查/创建成功")

            # 第三步：切换到目标数据库
            self.conn.select_db(db_name)

        except pymysql.Error as e:
            print(f"[错误] 数据库连接/创建失败：{e}")
            raise e

    def _reconnect_if_needed(self):
        try:
            self.conn.ping()
        except pymysql.Error:
            print("[提示] 数据库连接断开，正在重连...")
            self._connect_and_create_db()  # 修改：重连时也执行创建逻辑

    def init_table(self):
        self._reconnect_if_needed()
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS plate_records (
            id INT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键ID',
            file_name VARCHAR(255) NOT NULL COMMENT '上传的文件名',
            file_path VARCHAR(512) NOT NULL COMMENT '文件存储路径',
            plate_number VARCHAR(20) NOT NULL COMMENT '识别出的车牌号码',
            confidence FLOAT NOT NULL COMMENT '识别置信度(0-1)',
            file_type VARCHAR(10) NOT NULL COMMENT '文件类型(image/video)',
            result_image_path VARCHAR(512) NOT NULL COMMENT '标注后的结果图片路径',
            create_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '识别时间',
            PRIMARY KEY (id),
            INDEX idx_plate_number (plate_number),
            INDEX idx_create_time (create_time)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='车牌识别记录表';
        """
        try:
            self.cursor.execute(create_table_sql)
            self.conn.commit()
            print("[成功] 数据库表 plate_records 创建成功！")
        except pymysql.Error as e:
            print(f"[错误] 创建表失败：{e}")
            self.conn.rollback()
            raise e

    def insert_record(self,
                      file_name: str,
                      file_path: str,
                      plate_number: str,
                      confidence: float,
                      file_type: str,
                      result_image_path: str) -> int:
        self._reconnect_if_needed()
        insert_sql = """
        INSERT INTO plate_records (
            file_name, file_path, plate_number, confidence, file_type, result_image_path
        ) VALUES (%s, %s, %s, %s, %s, %s);
        """
        try:
            self.cursor.execute(insert_sql, (
                file_name, file_path, plate_number, confidence, file_type, result_image_path
            ))
            self.conn.commit()
            return self.cursor.lastrowid
        except pymysql.Error as e:
            print(f"[错误] 插入记录失败：{e}")
            self.conn.rollback()
            return 0

    def get_all_records(self, limit: int = 20) -> List[Dict]:
        self._reconnect_if_needed()
        query_sql = """
        SELECT * FROM plate_records 
        ORDER BY create_time DESC 
        LIMIT %s;
        """
        try:
            self.cursor.execute(query_sql, (limit,))
            return self.cursor.fetchall()
        except pymysql.Error as e:
            print(f"[错误] 查询记录失败：{e}")
            return []

    def get_record_by_plate(self, plate_number: str) -> List[Dict]:
        self._reconnect_if_needed()
        query_sql = """
        SELECT * FROM plate_records 
        WHERE plate_number = %s 
        ORDER BY create_time DESC;
        """
        try:
            self.cursor.execute(query_sql, (plate_number,))
            return self.cursor.fetchall()
        except pymysql.Error as e:
            print(f"[错误] 查询车牌记录失败：{e}")
            return []

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    # 添加到db_operations.py的PlateDB类中
    def delete_record(self, record_id: int) -> bool:
        """删除指定ID的记录"""
        self._reconnect_if_needed()
        delete_sql = "DELETE FROM plate_records WHERE id = %s;"
        try:
            self.cursor.execute(delete_sql, (record_id,))
            self.conn.commit()
            return self.cursor.rowcount > 0
        except pymysql.Error as e:
            print(f"[错误] 删除记录失败：{e}")
            self.conn.rollback()
            return False

    def clear_all_records(self) -> bool:
        """清空所有识别记录"""
        self._reconnect_if_needed()
        clear_sql = "TRUNCATE TABLE plate_records;"
        try:
            self.cursor.execute(clear_sql)
            self.conn.commit()
            return True
        except pymysql.Error as e:
            print(f"[错误] 清空记录失败：{e}")
            self.conn.rollback()
            return False
# 全局数据库实例（避免重复创建连接）
db_instance = PlateDB()