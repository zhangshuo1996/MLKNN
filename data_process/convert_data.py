"""
将数据库中 label_patent_industry 表中的数据 转化为 类似这种数据
-------------------------------

content,|,互联互通,产品功耗,滑轮提手,声音,APP操控性,呼吸灯,外观,底座,制热范围,遥控器电池,味道,制热效果,衣物烘干,体积大小
也没有烧伤,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0
主要是之前在北方呆过几几年,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0
温暖过冬,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0
开箱试了一下看着很不错,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0

-------------------------------

"""
from util import db
from config import *
from tqdm import tqdm
import numpy as np
import os


class ConvertData:

    label_list = []
    patent_id_list = []
    patent_id_vec_dict = {}

    def __init__(self):
        db.create_engine(**MYSQL_CONFIG)
        self.construct_label_list()

    def construct_label_list(self):
        """
        获取标签列表，并将标签列表写入文件，形如：

        ------------------------------------------------------------------

        |
        互联互通
        产品功耗
        ...

        ------------------------------------------------------------------
        :return:
        """
        sql = """
            select industry_title
            from label_patent_industry
            where used = 1
            GROUP BY industry_title
        """
        data = db.select(sql)
        self.label_list = [dic["industry_title"] for dic in data]
        self.write_label_list_to_file()

    def write_label_list_to_file(self):
        """
        将标签列表写入文件
        :return:
        """
        base_path = os.getcwd()
        file_path = os.path.join(base_path, '..', 'my_dataset', 'vocabulary_label.txt')
        write_str = "|\n"
        for label in self.label_list:
            write_str += label + '\n'

        if os.path.exists(file_path):
            return
        else:
            with open(file_path, 'w', encoding='utf8') as f:
                f.write(write_str)

    def get_patent_label_info(self):
        """
        获取 label_patent_industry 表中原始的专利-行业对应关系，并转化数据格式为：
        {
            patent_id: {label1, label2,...},
            ...
        }
        :return:
        """
        sql = """
            select lp.patent_id, lp.industry_title
            from label_patent_industry lp
            where used = 1            
        """
        data = db.select(sql)
        patent_id__info = {}
        for dic in data:
            if dic["patent_id"] in patent_id__info.keys():
                patent_id__info[dic["patent_id"]].add(dic["industry_title"])
            else:
                patent_id__info[dic["patent_id"]] = {dic["industry_title"]}
        patent_id_set = set()
        for dic in data:
            if dic["patent_id"] not in patent_id_set:
                patent_id_set.add(dic["patent_id"])
        self.patent_id_list = list(patent_id_set)
        self.get_patent_vec_by_id_list()
        return patent_id__info

    def construct_patent_industry_label(self):
        """
        将这种数据格式：
        {
            patent_text: {label1, label2,...}
        }
        转化为：
        {
            patent_id: [0, 1, 0, 0, ...] 顺序为标签列表中的顺序
            ...
        }
        :return:
        """
        patent_id__info = self.get_patent_label_info()
        patent_id__label = {}
        for patent_id, label_set in patent_id__info.items():

            this_patent_label_list = []
            for label in self.label_list:
                if label in label_set:
                    this_patent_label_list.append(1)
                else:
                    this_patent_label_list.append(0)
            patent_id__label[patent_id] = this_patent_label_list

        return patent_id__label

    def get_patent_vec_by_id_list(self):
        """
        根据专利id获取专利的向量
        :param patent_id_list:
        :return:
        """
        sql = """
            select patent_id, vec
            from patent_vec
            where patent_id in (
        """
        for patent_id in self.patent_id_list:
            sql += str(patent_id) + ","
        sql = sql[0:-1]
        sql += ")"
        data = db.select(sql)
        for dic in data:
            if dic["patent_id"] not in self.patent_id_vec_dict.keys():
                self.patent_id_vec_dict[dic["patent_id"]] = eval(dic["vec"])

    def write_to_file(self):
        """
        将最终的训练及测试数据写入文件
        {
            patent_id: [0, 1, 0, 0, ...] 顺序为标签列表中的顺序
            ...
        }
        :return:
        """
        patent_id__label = self.construct_patent_industry_label()
        index = 0
        base_path = os.getcwd()

        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for patent_id, label_f_list in tqdm(patent_id__label.items()):
            index += 1
            if patent_id not in self.patent_id_vec_dict.keys():
                continue
            patent_vec = self.patent_id_vec_dict[patent_id]
            flag = index % 10
            if flag < 2:  # 测试数据
                test_x.append(patent_vec)
                test_y.append(label_f_list)
            else:  # 训练数据
                train_x.append(patent_vec)
                train_y.append(label_f_list)

        train_x_np = np.array(train_x)
        train_y_np = np.array(train_y)
        test_x_np = np.array(test_x)
        test_y_np = np.array(test_y)

        np.save(os.path.join(base_path, "..", "my_dataset", "train_x.npy"), train_x_np)
        np.save(os.path.join(base_path, "..", "my_dataset", "train_y.npy"), train_y_np)
        np.save(os.path.join(base_path, "..", "my_dataset", "test_x.npy"), test_x_np)
        np.save(os.path.join(base_path, "..", "my_dataset", "test_y.npy"), test_y_np)


if __name__ == '__main__':
    c = ConvertData()
    c.write_to_file()

