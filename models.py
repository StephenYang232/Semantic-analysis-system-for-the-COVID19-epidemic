from exts import db
from datetime import datetime


class EmailCaptchaModel(db.Model):
    __tablename__ = "email_captcha"
    id = db.Column(db.Integer,primary_key=True,autoincrement=True)
    email = db.Column(db.String(100),nullable=False,unique=True)
    captcha = db.Column(db.String(10),nullable=False)
    # default值只会在第一次执行时使用，后续更改需要使用函数进行更新
    create_time = db.Column(db.DateTime,default=datetime.now)

class UserModel(db.Model):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(200), nullable=False, unique=True)
    email = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(200), nullable=False)
    join_time = db.Column(db.DateTime,default=datetime.now)


class UploadDataModel(db.Model):
    __tablename__ = "track_data"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    title = db.Column(db.String(200), nullable=True)
    content = db.Column(db.Text, nullable=False)
    patient = db.Column(db.String(200), nullable=False)
    gender = db.Column(db.String(100), nullable=True)
    age = db.Column(db.String(100), nullable=True)
    place = db.Column(db.String(600), nullable=True)
    city = db.Column(db.String(200), nullable=True)
    province = db.Column(db.String(200), nullable=True)
    town = db.Column(db.String(200), nullable=True)
    county = db.Column(db.String(200), nullable=True)
    district = db.Column(db.String(200), nullable=True)
    relation = db.Column(db.String(500), nullable=True)
    rel_pat = db.Column(db.String(500), nullable=True)
    onset_time = db.Column(db.String(200), nullable=True)
    symptoms = db.Column(db.String(500), nullable=True)
    track = db.Column(db.Text, nullable=False)
    create_time = db.Column(db.DateTime, default=datetime.now)
    author_id = db.Column(db.Integer,db.ForeignKey("user.id"))

    author = db.relationship("UserModel",backref="track_datas")