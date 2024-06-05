import pandas as pd

df = pd.DataFrame([('E146', 100.92, '[-inf ~ -999998.0]'),
                   ('E138', 107.92, '[-999998.0 ~ 2]'),
                   ('E095', 116.92, '[1.5 ~ 3.5]')],
                    columns=['name', 'score', 'value'])

print(df)
print()

# 逐行操作，其实就是调用DataFrame的iterrows()函数
for row_index, row in df.iterrows():
    print('行号：', row_index)
    print('第 {} 行的值: '.format(row_index))
    print(row)
    print('第 {} 行 value 列的值: '.format(row_index), row['value'])
    print()

# 逐列操作，其实就是调用DataFrame的iteritems()函数
# 注：panda2.0中已无此方法，若要使用则需降级版本
for col_index, col_value in df.iteritems():
    print('列名：', col_index)
    print('{} 列的值：'.format(col_index))
    print(col_value)
