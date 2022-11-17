from flask import Flask,session,g
from flask_migrate import Migrate
import config
from exts import db,mail
from blueprints import qa_bp
from blueprints import user_bp
from models import UserModel



app = Flask(__name__)
app.config.from_object(config)
# 将app绑定到db上
db.init_app(app)
# 将app绑定到mail上
mail.init_app(app)

# 数据迁移
migrate = Migrate(app,db)

# 绑定两个Blueprints
app.register_blueprint(qa_bp)
app.register_blueprint(user_bp)

# 钩子函数
@app.before_request
def before_request():
    # 从session中获取user_id
    user_id = session.get("user_id")
    if user_id:
        try:
            user = UserModel.query.get(user_id)
            # 将user绑定到全局变量上(给g绑定一个叫user的变量，它的值是user这个变量)
            # setattr(g,"user",user)
            g.user = user
        except:
            g.user = None

# 请求来了 -> before_request -> 视图函数 -> 视图函数中返回模板 -> context_processor
# 上下文处理器(渲染的所有模板都会执行以下代码)
@app.context_processor
def context_processor():
    # 用户登录后才会显示
    if hasattr(g,"user"):
        return {"user":g.user}
    else:
        return {}



if __name__ == '__main__':
    app.run()
