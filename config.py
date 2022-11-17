# flask的配置项目名字都是大写字母
JSON_AS_ASCII = False
# 数据库的配置变量
HOSTNAME = '127.0.0.1'
PORT     = '3306'
DATABASE = 'db_ayd'
USERNAME = 'root'
PASSWORD = 'qwezxcswh123'
DB_URI = 'mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'.format(USERNAME,PASSWORD,HOSTNAME,PORT,DATABASE)
SQLALCHEMY_DATABASE_URI = DB_URI
SQLALCHEMY_TRACK_MODIFICATIONS = True
# session的密钥
SECRET_KEY = "qwertyuio123456"

# 邮箱配置
# 项目中使用个人的QQ邮箱
MAIL_SERVER = "smtp.qq.com"
MAIL_PORT = 465
MAIL_USE_TLS = False
MAIL_USE_SSL = True
MAIL_DEBUG = True
MAIL_USERNAME = "2856348252@qq.com"
MAIL_PASSWORD = "wgmvqjdigelzdehf"
MAIL_DEFAULT_SENDER = "2856348252@qq.com"
