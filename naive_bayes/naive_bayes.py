import pandas as pd
import numpy as np
import pickle
import os

pkl_dir = os.path.dirname(os.path.abspath(__file__))
pkl_file = 'emails_words_count.pkl'

'''
该函数用于将邮件中所有单词变为小写，并将出现过的单词转换为列表
输出如下所示：左边为邮件文本，右边为出现过的单词的列表
 text  ...                                              words
0  Subject: naturally irresistible your corporate...  ...  [love, are, 100, even, isguite, %, to, and, yo...
1  Subject: the stock trading gunslinger  fanny i...  ...  [group, penultimate, tanzania, bedtime, edt, s...
2  Subject: unbelievable new homes made easy  im ...  ...  [complete, loan, rate, this, subject:, 3, adva...
'''
def process_email(text):
    text = text.lower()
    return list(set(text.split()))


'''
【建立模型】记录每个单词分别在垃圾邮件和非垃圾邮件中的出现次数
输入：
  - emails：电子邮件数据集
输出：
  - model：二维数组，大小为 (每个邮件的不重复单词个数, 2)
    - [word]['spam']：记录该单词出现在垃圾邮件中的个数
    - [word]['ham']：记录该单词出现在非垃圾邮件中的个数
'''
def words_count(emails):
    pkl_path = pkl_dir + '/' + pkl_file
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            model = pickle.load(f)
        return model

    model = {} # 该字典记录了每个单词分别在垃圾邮件和非垃圾邮件中的出现次数

    for index, email in emails.iterrows(): # 遍历所有的邮件
        for word in email['words']: # 检测该邮件中的每一个单词
            if word not in model:  # 如果字典中没有这个单词
                model[word] = {'spam': 1, 'ham': 1}  # 初始化为 1，防止后面计算概率时除数为零
            else:                  # 如果字典中已有这个单词
                if email['spam'] == 1:  # 如果该邮件是垃圾邮件
                    model[word]['spam'] += 1
                else:               # 如果该邮件是非垃圾邮件
                    model[word]['ham'] += 1

    with open(pkl_path, 'wb') as f:
        pickle.dump(model, f)

    return model


'''
【推理预测】预测指定邮件是否为垃圾邮件
输入：
  - email：待预测的电子邮件文本
  - model：
  - num_spam：数据集中垃圾邮件的总数
  - num_ham：数据集中非垃圾邮件的总数
'''
def predict_naive_bayes(email, model, num_spam, num_ham):
    email = email.lower() # 待预测邮件文本全部小写化
    words = set(email.split()) # 生成单词列表

    # 计算邮件的总数
    # total = num_ham + num_spam

    # 计算概率
    spams, hams = [1.0], [1.0]
    for word in words:
        if word in model:
            spams.append(model[word]['spam'] / num_spam)
            hams.append(model[word]['ham'] / num_ham)
        # np.prod()：计算数组中所有元素的乘积
        prod_spams = np.prod(spams) * num_spam
        prod_hams = np.prod(hams) * num_ham

    return prod_spams / (prod_hams + prod_spams)

# ============================= 主程序 main ===================================
if __name__ == '__main__':
    emails = pd.read_csv('emails.csv')
    emails['words'] = emails['text'].apply(process_email)

    num_emails = len(emails)
    num_spam = sum(emails['spam'])
    num_ham = num_emails - num_spam

    print("邮件总数:", num_emails)
    print("垃圾邮件数:", num_spam)
    print("邮件是垃圾邮件的先验概率:", num_spam / num_emails) # 计算先验概率
    print()

    # 记录每个单词分别在垃圾邮件和非垃圾邮件中的出现次数
    model = words_count(emails)

    # 出现单词 lottery 的邮件是垃圾邮件的概率？
    sum = model['lottery']['spam'] + model['lottery']['ham']
    print("单词 lottery 出现在邮件的次数：", model['lottery'])
    print("出现单词 lottery 的邮件是垃圾邮件的概率：", model['lottery']['spam'] / sum)

    # 出现单词 sale 的邮件是垃圾邮件的概率？
    sum = model['sale']['spam'] + model['sale']['ham']
    print("单词 sale 出现在邮件的次数：", model['sale'])
    print("出现单词 sale 的邮件是垃圾邮件的概率：", model['sale']['spam'] / sum)

    email1 = "Hi mom how are you"
    email2 = "buy cheap lottery easy money now"
    email3 = "meet me at the lobby of the hotel at nine am"
    email4 = "asdfgh"  # 不包含字典中任何一个单词，其为垃圾邮件的概率等于先验概率
    email5 = "As the sun rose, I packed my bags with essentials like a camera, a water bottle, and a few snacks. Excited and ready for the adventure, I hopped into my car and set the GPS to Laojun Mountain. The drive was a pleasant one, with lush green fields and rolling hills passing by the window."

    result = predict_naive_bayes(email5, model, num_spam, num_ham)
    print(result)
