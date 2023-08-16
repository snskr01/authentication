#list oftuples is database
#any changes made to tuple replaces whole tuple
table={'1000':('hrudhay',7676458621,50000),
'1001':('sanskar',7406923999,40000),
'1002':('vaiebhav',7864875039,40000),
'1003':('safwan',9449109613,70000)}

import numpy as np




def getbalance(uid):
    a = table.get(uid)
    b = a[2]
    return b

def getnumber(uid):
    a = table.get(uid)
    b = a[1]
    return b

def withdraw(uid,amount):
    a = table.get(uid)
    b = a[2]
    print('before balance:')
    print(b)
    b = b - amount
    c = (a[0], a[1], b)
    table.update({uid:c})
    return b


    