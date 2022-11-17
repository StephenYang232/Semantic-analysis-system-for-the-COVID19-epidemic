# 为了避免circular import，将db的初始化放到此文件中
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail
# 以下语句需要在调用视图函数之前执行
db = SQLAlchemy()
mail = Mail()