from flask import g,redirect,url_for
from functools import wraps
# 需要登录后才可使用
def login_required(func):
    # @wraps装饰器未写可能会导致func(属性)丢失
    @wraps(func)
    def wrapper(*args,**kwargs):
        if hasattr(g,"user"):
            return func(*args,**kwargs)
        else:
            return redirect(url_for("user.login"))
    return wrapper