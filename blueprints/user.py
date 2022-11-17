from flask import Blueprint,render_template,request,redirect,url_for,jsonify,session,flash
from exts import mail
from flask_mail import Message      # 导入邮箱信息
from models import EmailCaptchaModel,UserModel
import string
import random
from datetime import datetime
from exts import db
from .forms import RegisterForm,LoginForm
from werkzeug.security import generate_password_hash,check_password_hash

bp = Blueprint("user",__name__,url_prefix="/user")

@bp.route("/login",methods=['GET','POST'])
def login():
    if request.method == 'GET':
        return render_template("login.html")
    else:
        form = LoginForm(request.form)
        if form.validate():
            email = form.email.data
            password = form.password.data
            user = UserModel.query.filter_by(email=email).first()
            if user and check_password_hash(user.password,password):
                # 若登录成功，将user_id存入session
                session['user_id'] = user.id
                return redirect("/")
            else: # 若未登陆成功，则重定向回登录页面
                flash("邮箱和密码不匹配！")
                return redirect(url_for("user.login"))
        else:
            flash("邮箱或密码格式错误！")
            return redirect(url_for("user.login"))

# 退出登录
@bp.route("/logout")
def logout():
    # 清除session数据
    session.clear()
    return redirect(url_for('user.login'))



@bp.route("/register",methods=['GET','POST'])
def register():
    if request.method == "GET":
        return render_template("register.html")
    else:   # 以下代码需要在POST请求下执行
        # 获取前端form表单上传的数据
        form = RegisterForm(request.form)
        if form.captcha.data != None:
            input_captcha = form.captcha.data
            email = form.email.data
            captcha = EmailCaptchaModel.query.filter_by(email=email).first().captcha
            if form.validate() and input_captcha==captcha:  # 同时也使用了validate_captcha和validate_email函数验证
                # 获取相应数据
                username = form.username.data
                password = form.password.data
                # 使用MD5(hash)对于用户密码进行加密后存入数据库
                hash_password = generate_password_hash(password)
                # 创建用户
                user = UserModel(email=email, username=username, password=hash_password)
                db.session.add(user)
                db.session.commit()
                # 注册成功则跳转至登录页面
                return redirect(url_for("user.login"))
            else:  # 若验证失败(注册失败)
                flash("注册信息输入有误！")
                return redirect(url_for("user.register"))
        else:
            flash("验证码输入有误！")
            return redirect(url_for("user.register"))

@bp.route("/captcha",methods=['POST'])
def get_captcha():
    # POST方式获取验证码"1248164362@qq.com"
    email = request.form.get('email')
    # 生成验证码
    letters = string.ascii_letters + string.digits
    captcha = "".join(random.sample(letters,4))
    if email:
        message = Message(
            subject="邮箱测试",
            recipients=[email],
            body=f"【病例轨迹研究所】您的注册验证码是：{captcha}，请不要告诉任何人哦！"
        )
        # 发送对应body内容至对应邮箱
        mail.send(message)
        # 判断邮箱是否已存在，若存在则更新验证码和接收验证码时间
        captcha_model = EmailCaptchaModel.query.filter_by(email=email).first()
        if captcha_model:
            captcha_model.captcha = captcha
            captcha_model.create_time = datetime.now()
            db.session.commit()
        else: # 若不存在
            captcha_model = EmailCaptchaModel(email=email,captcha=captcha)
            db.session.add(captcha_model)
            db.session.commit()
        print("captcha:",captcha)
        # code:200,表示请求成功
        return jsonify({"code": 200})
    else:
        # code:400,表示客户端错误
        return jsonify({"code": 400,"message": "请先输入邮箱！"})

