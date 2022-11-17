import wtforms
from wtforms.validators import length,email,EqualTo
from models import EmailCaptchaModel,UserModel

class RegisterForm(wtforms.Form):
    username = wtforms.StringField(validators=[length(min=3,max=20)])
    email = wtforms.StringField(validators=[email()])
    captcha = wtforms.StringField(validators=[length(min=4, max=4)])
    password = wtforms.StringField(validators=[length(min=6,max=20)])
    password_confirm = wtforms.StringField(validators=[EqualTo("password")])
    # 验证验证码
    def validate_captcha(self,field):
        captcha = field.data
        email = self.email.data
        # 获取输入email对应的验证码
        captcha_model = EmailCaptchaModel.query.filter_by(email=email).first()
        if not captcha_model or captcha_model.captcha.lower() != captcha.lower():
            raise wtforms.ValidationError("邮箱验证码错误！")

    def validate_email(self, field):
        email = field.data
        # 获取数据库中表单接收email对应的用户信息
        user_model = UserModel.query.filter_by(email=email).first()
        # 若该邮箱已被注册过
        if user_model:
            raise wtforms.ValidationError("该邮箱已被注册过！")

class LoginForm(wtforms.Form):
    email = wtforms.StringField(validators=[email()])
    password = wtforms.StringField(validators=[length(min=6, max=20)])


class UploadForm(wtforms.Form):
    title = wtforms.StringField(validators=[length(min=0, max=200)])
    content = wtforms.StringField(validators=[length(min=3, max=1500)])



