import numpy as np



'''


TR=TR= np.zeros(10)
for i in range(10):
    TR[i] =TR[i]+TR1[i] + TR2[i] +TR3[i] + TR4[i] + TR5[i]



TR = TR/5
print(TR)
'''

#正态分布 （均值，标准差） (20,11) ~ (20,20)
'''
TR1=[1669.463161945343,989.6879103183746,690.9348495006561,523.0128326416016,696.5282714366913,1430.835167169571,1174.513776063919]
TR2=[340.844740152359,1048.5959525108337,347.723929643631,1229.2341203689575,740.4422242641449,625.2633376121521,970.6315288543701]
TR3=[708.6293323040009,909.7196562290192,670.6358041763306,1070.3237495422363,664.7604954242706,995.1825473308563,1089.7496767044067]
TR4=[953.8365371227264,691.089795589447,664.9627070426941,1195.4636199474335,1366.308732509613,1110.5121412277222,210.46446299552917]
TR5=[830.8394749164581,855.7719044685364,862.4834823608398,833.3385579586029,1270.6581745147705,737.4802379608154,464.3001661300659]

TR=TR= np.zeros(7)
for i in range(7):
    TR[i] =TR[i]+TR1[i] + TR2[i] +TR3[i] + TR4[i]

TR = TR/4
print(TR)
'''
#result=[ 918.19344288  909.77332866  593.56432259 1004.50858063  867.00993091 1040.44829834  861.33986115]

# 指数分布 β=1/λ   β；11~21
'''
TR1=[134.52871441841125,180.8011884689331,166.75038743019104,83.06882739067078,493.14239978790283,661.6007649898529,233.7367022037506]
TR2=[104.28409790992737,83.74151492118835,226.59320783615112,172.9823796749115,679.0435330867767,216.5367546081543,606.4557433128357]
TR3=[303.5581841468811,137.22881031036377,259.0201299190521,564.6417121887207,74.20242881774902,285.3818163871765,162.26707649230957]
TR4=[78.80924272537231,435.6611795425415,108.97421741485596,501.89453291893005,148.37164449691772,418.9557042121887,58.517003774642944]
TR5=[51.26856708526611,184.92096376419067,285.987717628479,150.04101872444153,437.4146783351898,191.38395714759827,438.9172086715698]

TR=TR= np.zeros(7)
for i in range(7):
    TR[i] =TR[i]+TR1[i] + TR2[i] +TR3[i] + TR4[i]+ TR5[i]

TR = TR/5
print(TR)
'''
#result=[134.48976126 204.4707314  209.46513205 294.52569418 366.4349369 354.77179947 299.97874689]