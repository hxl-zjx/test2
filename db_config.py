# db_config.py - MySQL数据库配置文件
# 👇 只需要修改下面这两个参数为你自己的MySQL信息
MYSQL_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",          # 你的MySQL用户名
    "password": "l200412193",    # 你的MySQL密码（必须改）
    "database": "plate_recognition_db",
    "charset": "utf8mb4"
}

# 缓存配置（无需修改）
CACHE_CONFIG = {
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 3600
}