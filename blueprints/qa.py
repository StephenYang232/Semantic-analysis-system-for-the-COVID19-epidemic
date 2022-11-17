from flask import Blueprint,render_template,g,request,redirect,url_for,flash
from decorators import login_required
from .forms import UploadForm
from models import UploadDataModel,UserModel
from exts import db
from sqlalchemy import or_

bp = Blueprint("qa",__name__,url_prefix="/")

@bp.route("/")
def index():
    data = UploadDataModel.query.order_by(UploadDataModel.create_time.desc()).all()
    return render_template('index.html',data=data)

@bp.route("/DataTrans/uploadData",methods=["GET","POST"])
@login_required # 相当于 login_required(UploadData)
def UploadData():
    # 判断是否已登录，若未登录则跳转至登录页面
    if request.method == "GET":
        return render_template("DataTrans.html")
    else:
        form = UploadForm(request.form)
        if form.validate():
            title = form.title.data
            content = form.content.data
            from bilstm_crf import call_blicrf
            kwords, syms = call_blicrf(content)
            kw = process_kwords(kwords,syms)
            uploadData = UploadDataModel(title=title,content=content,author=g.user,patient=kw['patient'],age=kw['age'],
                                         gender=kw['gender'],place=kw['place'],relation=kw['relation'],rel_pat=kw['rel_pat'],
                                         onset_time=kw['onset_time'],symptoms=kw['symptoms'],track=kw['track'],town=kw['town'],
                                         province=kw['province'],county=kw['county'],district=kw['district'],city=kw['cities'])
            db.session.add(uploadData)
            db.session.commit()
            return redirect("/")
        else:
            flash("标题或内容格式错误或为空！")
            return redirect(url_for("qa.UploadData"))

def process_kwords(kwords, syms):
    patient = None
    gender = None
    age = None
    place = []
    cities = []
    province = []
    town = []
    county = []
    district = []
    relation = None
    rel_pat = None
    onset_time = None
    symptoms = []
    track = []
    for i in range(len(kwords)):
        if syms[i]=='PAT'and patient==None:
            patient=kwords[i]
        elif syms[i]=='GEN':
            gender=kwords[i]
        elif syms[i]=='AGE':
            age=kwords[i]
        elif syms[i]=='PLACE':
            place.append(kwords[i])
            track.append(kwords[i])
        elif syms[i] == 'CITY':
            cities.append(kwords[i])
            track.append(kwords[i])
        elif syms[i] == 'REL':
            relation = kwords[i]
        elif syms[i] == 'PAT'and patient!=None and rel_pat==None:
            rel_pat = kwords[i]
        elif syms[i] == 'DATE':
            onset_time = kwords[i]
        elif syms[i] == 'SYM':
            symptoms.append(kwords[i])
        elif syms[i] == 'PROVINCE':
            province.append(kwords[i])
            track.append(kwords[i])
        elif syms[i] == 'COUNTY':
            county.append(kwords[i])
            track.append(kwords[i])
        elif syms[i] == 'TOWN':
            town.append(kwords[i])
            track.append(kwords[i])
        elif syms[i] == 'DISTRCT':
            district.append(kwords[i])
            track.append(kwords[i])
    place = ' '.join(place)
    cities = ' '.join(cities)
    symptoms = ' '.join(symptoms)
    province = ' '.join(province)
    county = ' '.join(county)
    town = ' '.join(town)
    district = ' '.join(district)
    track = ' '.join(track)
    kw = {'patient':patient,'age':age,'gender':gender,'place':place,'cities':cities,'province':province,'town':town,
          'county':county,'district':district,'relation':relation,'rel_pat':rel_pat,'onset_time':onset_time,
          'symptoms':symptoms,'track':track}
    for key,value in kw.items():
        if value==' 'or value==None or len(value)==0:
            kw[key]='未知'
    return kw




@bp.route("/DataTrans/Display",methods=["GET"])
@login_required # 相当于 login_required(UploadData)
def Display():
    inf = []
    age_group = []
    people = []
    link = []
    places = []
    place_data = []
    place_stat = {}
    dates = []
    count = UploadDataModel.query.count()
    for i in range(1,count+1):
        line = UploadDataModel.query.filter_by(id=i).first()
        datum = {'id':line.id,'name':line.patient,'gender':line.gender,'age':line.age,'track':line.track,'onset_time':line.onset_time,'symptoms':line.symptoms}
        dates.append(line.create_time.date())
        # 构造关系图数据
        if line.rel_pat != '未知':
            p = line.patient+'('+str(line.id)+')'
            rel_person = {line.rel_pat:60}
            person = {p:40}
            rel = [line.rel_pat,p,line.relation]
            if person not in people:
                people.append(person)
            if rel_person not in people:
                people.append(rel_person)
            if rel not in link:
                link.append(rel)

        age_group.append(line.age)
        place_stat = place_analysis(line.track,place_stat)
        inf.append(datum)
    # 构造折线图数据
    axis,date_count = daily_pat_count(dates)
    # 柱状图数据
    nl = top_n_scores(6, place_stat)
    for n in range(len(nl)):
        places.append(nl[n][0])
        place_data.append(nl[n][1])
    # 饼状图数据
    age_data = age_division(age_group)
    age_div = [age_data['老年人(60岁以上)'],age_data['中年人(45岁-59岁)'],age_data['青年人(18岁-44岁)'],age_data['少年儿童(18岁以下)']]

    return render_template('display.html',axis=axis,data=date_count,people=people,link=link,track_inf=inf,places=places,place_data=place_data,age_data=age_div)


def place_analysis(place_data,place_stat):
    place_list = place_data.split(' ')
    for i in range(len(place_list)):
        if place_list[i] not in place_stat.keys():
            place_stat[place_list[i]] = 1
        elif place_stat[place_list[i]]:
            place_stat[place_list[i]] += 1
    return place_stat

def top_n_scores(n,score_dict):
    lot = [(k,v)for k,v in score_dict.items()]
    nl = []
    while len(lot)>0:
        nl.append(max(lot,key=lambda x:x[1]))
        lot.remove(nl[-1])
    return nl[0:n]

def age_division(age_group):
    age_div = {'老年人(60岁以上)':0,'中年人(45岁-59岁)':0,'青年人(18岁-44岁)':0,'少年儿童(18岁以下)':0}
    for i in range(len(age_group)):
        if age_group[i] != '未知':
            age=int(age_group[i][:-1])
        else:
            continue
        if age >= 60:
            age_div['老年人(60岁以上)'] += 1
        elif age >= 45 and age < 60:
            age_div['中年人(45岁-59岁)'] += 1
        elif age >= 18 and age < 45:
            age_div['青年人(18岁-44岁)'] += 1
        else:
            age_div['少年儿童(18岁以下)'] += 1
    return age_div

def daily_pat_count(dates):
    date_note = {}
    date_key = []
    date_value = []
    for i in range(len(dates)):
        date = str(dates[i])
        if date not in date_note.keys():
            date_note[date] = 1
        else:
            date_note[date] += 1
    for key,value in date_note.items():
        date_key.append(key)
        date_value.append(value)
    return date_key[-5:],date_value[-5:]

@bp.route('/search')
def search():
    # 传递方式: /search/?q=xxx
    q = request.args.get("query")
    # filter_by: 直接使用字段名称; filter: 使用模型.字段名称
    questions = UploadDataModel.query.filter(or_(UploadDataModel.title.contains(q),
                                                 UploadDataModel.content.contains(q),)).order_by(db.text('-create_time'))

    return render_template("index.html",data=questions)