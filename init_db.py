# init_db.py - 一键初始化数据库表
from db_operations import db_instance

if __name__ == "__main__":
    print("正在初始化数据库...")
    db_instance.init_table()
    db_instance.close()
    print("✅ 数据库初始化完成！")