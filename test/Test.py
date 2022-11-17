# from bilstm_crf import call_blicrf,Mymodel
#
# k,s=call_blicrf('确诊病例5：女，21岁，北京人，现住朝阳区太平庄南里幸福一村，曾到访天堂超市酒吧（工体西路6号），曾去过天津，6月14日诊断为确诊病例，是确诊病例4的女儿。')
# print(k)
# print(s)
str = '长安广场 长安广场 长安二宽牛肉拉面馆 碧景园菜鸟驿站 路石油销售总公司 雄县服务区 唐县服务区 蔡庄荆蒲兰村'
place_stat = {}
def place_analysis(place_data):
    place_list = place_data.split(' ')
    for i in range(len(place_list)):
        if place_list[i] not in place_stat.keys():
            place_stat[place_list[i]] = 1
        elif place_stat[place_list[i]]:
            place_stat[place_list[i]] += 1

def top_n_scores(n,score_dict):
    lot = [(k,v)for k,v in score_dict.items()]
    nl = []
    while len(lot)>0:
        nl.append(max(lot,key=lambda x:x[1]))
        lot.remove(nl[-1])
    return nl[0:n]

place_analysis(str)
nl = top_n_scores(6,place_stat)
print(place_stat)
print(nl)