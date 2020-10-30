import random, time
import numpy as np
from multiprocessing.managers import BaseManager
from multiprocessing import freeze_support, Queue
from multiprocessing import Process
from Two_Replication_distribute.client import win_client_run

# 任务个数
task_number = 20

# 收发队列
task_quue = Queue(task_number)
result_queue = Queue(task_number)


def get_task():
    return task_quue


def get_result():
    return result_queue


He = [[9, 8, 1, 3, 4, 3, 3, 10, 2, 1],
      [10, 10, 17, 1, 7, 15, 15, 10, 10, 10],
      [28, 8, 8, 30, 28, 17, 18, 31, 10, 28],
      [19, 3, 18, 33, 38, 40, 20, 36, 14, 26],
      [46, 27, 24, 27, 14, 6, 18, 30, 17, 1],
      [29, 30, 26, 1, 23, 32, 7, 16, 4, 41],
      [20, 28, 65, 64, 14, 44, 23, 19, 54, 41],
      [48, 75, 26, 19, 79, 48, 45, 72, 42, 11],
      [17, 86, 45, 4, 37, 58, 26, 17, 43, 64],
      [24, 28, 37, 97, 14, 71, 62, 87, 78, 11]]

He1 = [[3, 6, 6, 6, 3, 9, 2, 4, 9, 9],
       [2, 11, 15, 2, 8, 18, 17, 7, 15, 5],
       [1, 26, 5, 5, 23, 8, 1, 9, 28, 6],
       [35, 4, 6, 22, 35, 13, 31, 12, 29, 29],
       [25, 23, 29, 38, 19, 19, 36, 1, 9, 16],
       [44, 31, 28, 45, 22, 2, 5, 29, 17, 2],
       [1, 24, 57, 9, 38, 24, 43, 19, 44, 22],
       [6, 29, 10, 59, 78, 71, 8, 33, 7, 5],
       [79, 6, 78, 25, 19, 84, 2, 62, 49, 30],
       [93, 80, 61, 96, 35, 27, 89, 44, 46, 7]]

He2 = [[7, 7, 8, 2, 7, 4, 3, 5, 6, 5],
       [8, 16, 15, 1, 5, 13, 1, 11, 9, 18],
       [17, 10, 14, 26, 7, 4, 17, 19, 14, 26],
       [5, 14, 29, 4, 5, 8, 19, 3, 4, 4],
       [37, 15, 37, 30, 37, 7, 5, 31, 25, 7],
       [3, 38, 13, 43, 32, 58, 56, 44, 3, 50],
       [42, 27, 59, 26, 36, 44, 2, 67, 41, 41],
       [39, 22, 18, 64, 46, 73, 13, 44, 72, 1],
       [79, 36, 52, 60, 30, 28, 14, 50, 23, 75],
       [98, 26, 9, 88, 95, 74, 27, 59, 28, 59]]

He3 = [[8, 2, 1, 3, 2, 6, 1, 4, 9, 5],
       [17, 7, 13, 3, 7, 4, 1, 3, 17, 8],
       [11, 9, 6, 15, 11, 17, 1, 7, 1, 2],
       [30, 22, 35, 6, 11, 15, 8, 7, 24, 17],
       [38, 38, 44, 21, 30, 34, 31, 19, 9, 12],
       [27, 27, 22, 37, 40, 21, 9, 56, 34, 46],
       [11, 55, 13, 46, 22, 50, 12, 33, 65, 2],
       [12, 32, 12, 43, 34, 42, 28, 21, 59, 17],
       [4, 44, 2, 81, 35, 70, 41, 6, 20, 62],
       [84, 15, 22, 34, 70, 40, 16, 78, 7, 45]]

He4 = [[1, 1, 1, 3, 1, 8, 3, 7, 3, 4],
       [9, 6, 2, 1, 19, 5, 19, 19, 17, 6],
       [16, 4, 19, 15, 3, 21, 10, 25, 3, 6],
       [28, 5, 10, 4, 19, 31, 36, 17, 32, 3],
       [4, 44, 36, 46, 14, 6, 8, 35, 17, 24],
       [51, 33, 14, 6, 45, 54, 25, 53, 40, 37],
       [47, 52, 43, 33, 7, 19, 22, 17, 6, 14],
       [54, 75, 21, 55, 71, 33, 77, 40, 69, 60],
       [13, 4, 37, 52, 42, 15, 27, 62, 22, 46],
       [32, 18, 3, 88, 78, 68, 20, 62, 89, 40]]

# 正态分布 （均值，标准差）（20，11）~（20，20）
Ze=[[27.62117768467989, 30.866184746145763, 30.559814679582487, 31.3666899405599, 27.916199329470327,
    54.38477386599491, 33.541368694193224, 21.40231097528221, 35.715477631571474, 38.81846259114848],
[37.77006802832193, 13.766761137703103, 28.97111345812609, 23.149772514122134, 23.52408878796675,
    27.50834760859369, 3.3270278947594676, 27.259085846966602, 28.832430401012225, 23.64977298769366],
[21.34900770546612, 40.97063554189312, 18.716519055160877, 5.966517021351356, 25.53188748383473,
    27.950340982085653, 12.502354856981125, 2.1732329731582105, 25.89682094379251, 33.590220339219506],
[10.549480637824725, 40.86434870274604, 28.407257076481844, 7.082879722158941, 20.86226734913784,
    40.03886896631394, 2.377499968337439, 1.0160680814883207, 27.533800870129916, 19.689525987447215],
[20.9600093836833, 18.599520049964944, 28.752614053248184, 2.184292814553615, 29.32506407196665,
    8.214246110222073, 32.593758218000985, 38.29943126393938, 6.879996380340408, 17.104600191183504],
[37.554986191003366, 36.41693675381203, 38.68263143234819, 33.09412366418505, 19.909616858753882,
    25.203484529638196, 34.89443239115107, 19.71896367241376, 8.711057355875921, 36.34902002431933],
[29.306903277200245, 38.42695632515432, 9.198740089802312, 45.18241084083763, 7.380135411078696,
    41.30696333479612, 45.80900746811772, 12.300670218276773, 25.323478595535143, 46.1734428566095],
[35.45855919863213, 15.762644039709867, 40.35319825518876, 15.514229780508586, 28.21385659597418,
    12.472861245289529, 16.96640259437253, 25.675571012282013, 25.72873603661673, 32.43166167239754],
[19.610587994429338, 20.069590700002394, 8.034539069009707, 30.395006609691343, 10.254894298068974,
    33.33499232324965, 25.560311640374472, 41.16377006956533, 1.5423471594594567, 45.03966544094053],
[36.300606524926, 19.56837581807415, 30.426290649328863, 43.93627828092403, 43.66531260504958,
    33.36291000366608, 16.5022764274012, 29.74233271177005, 46.04302243187067, 58.73982538280477]]


Ze1=[[2.7049938612499638, 40.65841797727415, 4.416400730767355, 9.206152609850632, 24.040552106255376,
    4.704630948161029, 6.8705017390904395, 26.16522934145292, 7.012745961564114, 3.0838086430842253],
[18.12438091724859, 7.603083686517447, 16.683137284062493, 13.69593286008902, 19.147304435176437,
    26.41002078644183, 25.90292018261658, 39.56957010859987, 11.111086528299419, 43.33053619735425],
[23.808974275935242, 0.5022019300472529, 13.827136878765605, 15.960258275771722, 18.657982134169657,
    7.815169352469878, 16.049295320576636, 21.854642558050003, 12.55693244834692, 17.56782414308003],
[23.043230938597823, 12.6156436891271, 25.875046124936283, 33.49116905167063, 53.90989343281669,
    46.79535949125044, 14.865803908614975, 13.330121562140807, 21.407250430962975, 30.14752158963251],
[12.558503106681485, 4.6432494785427565, 12.876151280017126, 37.75869245116711, 26.786351250980964,
    13.484215930100202, 10.115582059954543, 15.9890602082605, 33.91210707705943, 50.825434420822305],
[32.750084699965846, 17.32260920609137, 20.032926236127636, 5.715967053250203, 2.1385629535197737,
    27.623705606388725, 18.93601221006147, 21.657827774365263, 26.370708693765263, 21.21048805752335],
[42.19368616247371, 26.251846504146048, 10.671888066606313, 9.143742824791648, 25.144113265338394,
    47.05890956355948, 36.14858710482105, 8.8062122221195, 19.02236527806332, 19.290480951398713],
[14.490691725690649, 33.03543132483958, 27.35964945700878, 37.297340750369415, 26.05300700194943,
    15.231434359316527, 20.525215691879467, 1.0565321036814908, 5.427190227011753, 7.004268838460641],
[0.6436127608669509, 53.25508236301534, 2.4467860827454473, 1.2525256362650978, 45.77699811445706,
    24.679549176504516, 29.835370886893955, 34.29141025434075, 22.51392672389215, 13.162470766488674],
[19.989714316341686, 54.679988386443604, 13.047960592206149, 25.43578612977176, 15.482702151662792,
    12.839695753028936, 53.07579953233606, 37.41151924907143, 2.4614333387387255, 16.15684176519784]]


Ze2=[[12.99294316762791, 42.216023773610104, 19.9020857215406, 18.186308410001367, 23.274317722807243,
    14.916241980463253, 5.7740852338167485, 19.081399010061048, 16.735972287982577, 6.853801501271054],
[38.876240554862086, 6.145268652137707, 12.122480841975504, 32.31076415121286, 18.23445832695799,
    19.75297935641584, 23.36728788122413, 33.69892358407043, 30.26671545101108, 16.428946905229218],
[22.40913192406523, 12.582804945418639, 19.803558441852353, 3.0316195714227554, 33.688243566799066,
    13.565969906236269, 39.02423918430114, 27.166182218728203, 10.07107652498371, 11.30453753707981],
[39.15734357704844, 24.863949836175095, 41.62334961681215, 29.175709382490354, 40.137081456092986,
    6.793467703613006, 55.135265629482724, 21.1652690664963, 13.155995867666963, 15.195318277256813],
[37.85218018679551, 22.84468363647849, 40.37756825663982, 43.23758458105103, 9.155839690151682,
    10.192286228445944, 37.544505177949915, 27.212627524713653, 4.2662644715047975, 3.373982662710457],
[51.66099634493508, 12.613327997517871, 17.930775377832287, 38.55461083218212, 26.947719718086326,
    49.03479364423558, 48.86284097402643, 24.213785966120533, 20.249796976923726, 3.747297646210228],
[10.053132876974372, 32.04308438243251, 13.414124853961004, 22.159164105888465, 21.006401235011165,
    51.22697237617737, 26.32184098093552, 36.11328911386259, 16.88578728731626, 24.998226683875178],
[19.951382804907745, 24.854209016854064, 15.277151465553844, 15.530722825234482, 38.77236661482982,
    38.792497764862496, 28.745559697023193, 7.259729885235997, 35.43673104471485, 41.034140754602134],
[27.90875204068774, 17.994680192367248, 51.354152719468374, 35.71772240691027, 43.73365386468804,
    34.20986588738447, 6.705516551066232, 18.635502783978783, 26.46460737992327, 4.446156023861086],
[36.938552717304354, 22.232769522485576, 10.717875933744073, 40.468556572981456, 2.588141281830076,
    31.82214163514447, 38.72785691719969, 9.252182187019573, 16.693847795692843, 5.694100093725387]]


Ze3=[[26.892592617129573, 26.373306501894298, 55.285683238499075, 18.041790202840396, 9.207192158232171,
    25.55080153425816, 19.866290716197398, 9.689060825147916, 13.905788289372227, 23.176669862112067],
[7.915117682743164, 8.040939210162087, 9.31022035722253, 34.5148161349723, 15.444071241952534,
    11.331390876942647, 22.002536051172314, 22.373949073970678, 13.560087260116624, 12.956413669047441],
[2.244880641849214, 38.844912924322664, 11.059306134282709, 35.924396168284986, 31.066080688846462,
    19.971174128437795, 49.021907427784384, 12.71127184144322, 37.36042119942107, 33.434115872628574],
[39.5885073614628, 20.69964159396087, 25.033138631090033, 20.365694310498032, 17.070003994067655,
    16.417055855210368, 26.037371718556237, 31.598841903717258, 30.546172887731018, 27.13150291031983],
[25.803913229149956, 31.902887441186486, 31.996086984457435, 28.75539879760899, 24.980476402996906,
    37.5756788058483, 34.389072942021315, 9.315553513345035, 38.00239782876457, 27.422371237535796],
[20.857519313456713, 13.200817971211173, 27.13813788848092, 20.6867727747588, 27.387786708992063,
    19.119694711165874, 14.762826520655945, 41.283301687686674, 25.636072496631183, 30.940062768770332],
[20.06261572442253, 9.043031596443582, 7.585755846761028, 0.30046320987538166, 1.4882729401276293,
    19.94028340197286, 27.49971809787823, 42.4768733632775, 44.13519702948268, 18.065527095916043],
[39.10566784864127, 36.98957818878617, 19.248789848354004, 34.01621244990989, 28.931219682545226,
    30.058894637441817, 7.537506610535189, 24.14907036649368, 13.023387847565807, 52.011253437795986],
[9.890477717766002, 58.014837305380496, 3.54993868265057, 9.619663994151342, 18.6986541015865,
    16.981271710581314, 19.556142075358473, 18.10614377988946, 62.451254253038584, 12.648752995527445],
[13.300110630016647, 25.067893085376713, 38.45128908224768, 30.6412605914662, 13.43018178651767,
    30.6961171193177, 16.759355798440225, 36.66029714957044, 74.42278243175248, 19.26557016476211]]


Ze4=[[24.333688554301137, 20.887163512856922, 26.631612811854176, 28.408295091382595, 2.4301647297924873,
    24.702848433877634, 21.14327775273649, 35.78954493128835, 21.985292822415992, 26.10174232834911],
[22.791282364246705, 9.107807607584347, 20.498322734071362, 22.665190497292436, 20.479623025337386,
    36.315483223179754, 10.86265974434275, 27.757625867284162, 29.789406241965164, 3.0978832616630463],
[18.679351696408055, 46.510015382796496, 25.829869924263086, 4.3308938284959755, 44.759457774306156,
    6.481958492707712, 35.57677240992574, 18.36642011453436, 28.780436842747587, 35.31872194288267],
[24.329299099815707, 34.591696705890605, 7.170492075549435, 18.84564049678192, 19.952101456508213,
    39.03765109074348, 28.720406209686097, 6.8758017651642565, 8.971319278155628, 31.020070783260593],
[18.796063089061246, 11.66736533062891, 18.90450622370167, 37.0729510223738, 20.909457527912167,
    25.671638764582532, 32.970538991635884, 25.857100064019228, 25.350510887344072, 37.6228646683568],
[30.17775959561404, 22.091888189355416, 29.478700885856885, 36.44108999313039, 34.8970782593508,
    5.647703363664682, 15.699732626387554, 16.971607567053585, 23.254915554877375, 3.6169342769167834],
[21.493105008142123, 4.029113136245195, 22.175993193979323, 27.11154824320274, 1.9067763142863043,
    10.775540883662673, 33.79117784084836, 27.078426921767203, 60.19913010872567, 21.981352970772477],
[19.009968190201143, 1.045499626350324, 35.59492929220484, 22.118670760099228, 28.49107084702493,
    18.3664770227974, 5.57747689610367, 10.668724097275874, 58.21267335163629, 41.51595261071016],
[13.130935801342837, 32.75146193996056, 12.59287369480392, 39.84871525997217, 18.793033801085706,
    17.109659169999144, 4.673188874166099, 15.316497829386744, 27.656364436952927, 12.431444658101977],
[32.69096909118758, 55.9067530643066, 22.484194080174348, 26.497256980317797, 36.968162892615595,
    50.527315599181506, 39.23569324590672, 9.85273418734815, 6.155587142272976, 5.3228269381423505]]

# 指数分布 β=1/λ   β；11~21
Ee=[[0.8323678535711884, 10.876294463596299, 0.9536754257704547, 5.293106757029529, 1.2993887794227945,
            7.61037116287269, 11.971916551738166, 10.483716817313462, 15.636274434355517, 0.8124259281588808],
    [0.6483922491157562, 6.435742876452355, 18.48907390161046, 17.979080998020763, 30.503987267349128,
        1.3639504790436598, 24.550231357251768, 4.131475291103846, 25.68180013130973, 12.224224563090152],
    [4.953395144194205, 4.95999644973569, 5.36201561276998, 0.9060601229000804, 4.295826487679851,
        3.254388297718213, 8.839608223801417, 44.24937993685808, 3.639367147741159, 0.6790692097240598],
    [20.943217885956486, 14.626609905399182, 5.854804953252631, 9.628626299656133, 23.31679712736878,
        13.54995912542136, 11.199195839965846, 5.417736766451543, 0.12602298703356185, 31.28941170781712],
    [69.71278343284246, 21.700917793919093, 8.658603300276598, 40.63285614445452, 13.82638678170059,
        10.074620229868172, 3.2434805486131424, 5.826236136946424, 19.217055445977895, 4.3784158709417555],
    [9.988595419741104, 60.072245576414446, 6.927327514203118, 65.97940409152042, 3.083215516085974,
        66.25054651241219, 19.57514025100137, 61.51845198542291, 14.9564923704392, 60.806677755629444],
    [4.576764653946501, 10.372745930307916, 3.3774653532668, 1.3840651001456685, 17.417151331126217,
        12.499778873328315, 15.715111274470171, 1.3555380522633476, 29.932237626704612, 18.937301809210787],
    [3.2626224013210106, 14.733681019121738, 8.559099912873005, 42.27777867458492, 38.18656562230289,
        63.77905373449322, 4.5233817429966665, 10.152597623438908, 30.66934743505263, 14.356007092970989],
    [4.438066315474881, 8.47473869713229, 8.28925424735812, 6.880877020406312, 4.952021212438446,
        78.00844457799498, 89.0749480991454, 12.854929615344657, 40.757574191572864, 3.411517507806678],
    [55.78876815722758, 6.730117002718501, 4.957109029145715, 22.793674156809285, 4.528272891120207,
        3.552923036705608, 58.30552115284145, 11.173988430900872, 5.594214383678944, 12.889398428193537]]


Ee1=[[6.371067902184769, 3.4724696106489086, 4.06128298696204, 0.7810264339423035, 11.790763595189746,
        11.185857491097911, 0.2370427505477018, 39.146991400967714, 16.334823556843528, 42.04630376734452],
    [36.00779238407143, 8.83484626949745, 1.5776418402409704, 11.555361989635411, 0.162817545849212,
        6.641335364493037, 1.5884684905353974, 26.402373436417918, 4.608102657736157, 14.43169715869259],
    [0.8155817308305571, 43.5320167954007, 7.089024836794925, 3.964607199039511, 9.217371603278,
        29.283301898790423, 19.938653146712316, 2.149564444515126, 30.480634158540152, 22.46677456311922],
    [1.4287018172046164, 2.0330658116121336, 11.175712839792284, 1.9077432446839502, 0.9347654741029724,
        8.004921632345113, 10.792615428779449, 29.231527510689496, 35.57297333583809, 12.830264373803772],
    [12.336929172703364, 17.15199416139907, 12.477024354707869, 100.8302778837507, 13.841119184375625,
        15.024435804987561, 7.347424126972162, 12.797045313944762, 32.319743772131204, 5.82995926226583],
    [2.2898164660528164, 1.3416080617761337, 21.02946806097361, 4.222606755629445, 14.04758933933088,
        5.014272153259628, 12.020915998535449, 2.609536100796557, 12.058152395768708, 10.746688594866352],
    [29.236837164118068, 22.69599337073572, 18.848693662139446, 5.159662548917881, 3.076764534475609,
        5.265388011766912, 29.05239335910589, 21.49673213064793, 18.514256841430637, 37.74445554755623],
    [26.988549617972332, 9.285185738340543, 23.818642351961298, 15.1041768324226, 16.738653102814705,
        27.542901245656036, 3.7082588411533144, 6.491627502365186, 40.80093160192346, 18.79394518793334],
    [0.1277358608361876, 4.4596831453832495, 1.9843893954773666, 10.659707709505446, 23.958268438625993,
        71.74384759186283, 26.016425094305323, 1.2238307290299801, 8.900188806417722, 22.96997819622435],
    [22.354554397767433, 32.5545465068045, 46.72051653893376, 6.229686479875161, 20.47803908613806,
        11.172460750264218, 5.445139042289729, 6.294880547087054, 0.6570857163038757, 2.569209611000213]]


Ee2=[[8.321492909004457, 8.187323601269748, 6.690127143159335, 14.640446016255401, 8.407487851400449,
        4.542647904965742, 1.6486005045978238, 4.575032532773652, 14.596316183167602, 2.405100131841827],
    [7.399918077026136, 23.203776621878482, 4.35085108113431, 3.1466031571815263, 1.3958122657789902,
        0.33757384300634796, 15.518756771771223, 8.505656249498374, 20.815625468942912, 6.5232209733198],
    [4.196802700543525, 13.351455920102596, 2.2798549690961036, 1.7006311911433893, 17.43596891023644,
        2.3194829790674616, 14.483967762224584, 12.849667257499416, 4.237145217671423, 11.650633172917313],
    [12.347624774783032, 27.284812749625907, 11.187214997900618, 20.809229977610855, 6.124025594803915,
        19.64813264776666, 7.623526097443065, 5.187193347301361, 6.8235011841863304, 29.65859881742895],
    [7.4066305829446275, 3.8345906265172256, 3.3885929218584767, 0.5467289244966997, 21.509656742944806,
        13.752016280256923, 0.1275583549367314, 2.5458676393219655, 2.0993626579413567, 6.357538814757896],
    [0.9670310563438718, 23.534646678862707, 3.701472571189978, 11.141041988964094, 49.645855339290684,
        4.054060889203846, 35.31613760347067, 26.73588909861721, 10.489378783797354, 32.75725656111171],
    [1.8557626218342036, 49.22843333208302, 2.550761426860961, 13.099419012985893, 59.549867888481955,
        56.568577914894014, 1.9158605206476975, 0.6033153565428755, 5.25878337303232, 25.043366949914503],
    [7.326015550670136, 18.367520976923924, 8.083300846982418, 2.7835435355827056, 5.50318289968326,
        0.13300517250474098, 0.9859521859853417, 12.618732251079267, 40.27482860067276, 15.593293499185584],
    [15.526908394294747, 19.428850211614627, 10.99396220242294, 0.916857918616174, 28.024394822277333,
        28.63310492747184, 4.447323568783542, 22.41504259119255, 27.887675952203793, 5.359219843204245],
    [12.821203893936996, 92.74535322764358, 40.92617877556546, 23.121400171762595, 11.287509688127322,
        6.247822873173262, 0.7287255107154171, 8.582666775831456, 20.59813572501107, 74.94780410375674]]


Ee3=[[2.2296936121972126, 3.0183579926450794, 20.752847752795624, 11.765113932621635, 27.623771609984825,
        2.4624261110237033, 0.2563665413209297, 0.3994076314801017, 0.43392506153682137, 4.251568280951621],
    [36.990938840568525, 4.0281258646338145, 1.6068180083546404, 20.663381382685166, 23.706314378328347,
        30.66994806038217, 32.530994548982854, 32.34590508362189, 5.5834533299925635, 20.176637966285114],
    [2.292433265874363, 9.042813544914361, 2.874845450189339, 3.929428304023953, 6.6572132996172115,
        43.61780702018348, 3.804326727657044, 17.284989624290727, 0.2394969525518669, 16.039041702245875],
    [12.029591576981597, 21.664246613980136, 25.919859347547238, 2.6783030072298852, 7.353819990597357,
        20.30781209876297, 20.85963430942028, 4.5254378577083125, 13.936854963062698, 12.619210519951643],
    [4.021902787160573, 10.372037425295225, 28.530178086290594, 59.548307301090084, 9.925149226217535,
        4.395600896066555, 7.3914648960797456, 5.8542566235208096, 6.077624303498377, 0.13051445101461484],
    [17.474438624877827, 40.019603968996066, 9.893681552535833, 3.3550144459382203, 1.9574544604825594,
        16.946460738404408, 32.190377769591514, 12.365433548979873, 16.47085948532477, 11.07789943435465],
    [25.40044585590774, 1.2193425471467334, 1.5765159610691721, 16.233923942764594, 0.030861650417183823,
        5.358628883242595, 0.8862092476822245, 10.737065430435237, 32.66692318226636, 15.56503930345333],
    [15.693253977365027, 10.840182465099147, 47.89011977534333, 53.55928448007417, 8.061463331683795,
        40.50271205386103, 0.5885678174740412, 36.068686656753044, 23.76171306041683, 45.634948655316066],
    [18.98862798344458, 21.85012755457172, 46.27245379429985, 15.365403105269458, 41.634756651261874,
        7.041199308003144, 21.264638378037656, 20.123603995951758, 8.896359454052583, 27.342681400469633],
    [10.500288074178094, 5.406364531245384, 29.16641588383024, 1.0548505791538205, 20.198255962955457,
        8.032866288198104, 22.78037737809022, 21.23823685690588, 23.47752206774677, 17.93366124620583]]


Ee4=[[0.027639868910378913, 8.474367670957793, 0.25563733024427215, 21.39433297406845, 9.782192228120724,
        9.343958149991149, 0.9764503888869344, 1.639181752997753, 2.8187375752833934, 5.11133886131572],
    [9.483921704340634, 4.2637491658862965, 7.204350572427689, 1.9283727633111027, 25.618016747655535,
        6.964135210101052, 6.630433715632515, 11.997045647470628, 0.590070022802271, 13.134091892474874],
    [3.6664613821152674, 18.160654420358053, 15.848188925661546, 3.0904431945984694, 5.713183084153791,
        3.6366754434830857, 2.716638954495236, 3.9149353212751974, 7.247365493777798, 8.691654600117632],
    [51.986821946027376, 6.991115183179045, 29.097566020003786, 23.484004835776634, 7.062528717078615,
        0.6312323251267365, 1.089160346723071, 2.5359867495363524, 12.606806437963538, 2.5798110092479054],
    [14.810830222071676, 34.92376401198735, 3.692416309817516, 7.834259162340246, 5.214239322116649,
        18.192210784090975, 42.78927090604712, 12.74136894626935, 4.486206156472067, 5.598259896714901],
    [1.1610542869045652, 16.26015352715393, 3.7596708997158705, 33.19904566421367, 37.125747325280734,
        9.009448107188266, 0.9940379670437423, 3.0221721483869812, 17.698060743507277, 23.02177719262485],
    [11.306920909115485, 10.527615761386286, 17.94049372409582, 4.736964422586866, 25.110788747809146,
        7.656253666958259, 4.923157539838771, 20.939209843527124, 4.622714263136357, 5.602378375680015],
    [68.37074733761379, 1.023152523369899, 3.090227560539399, 8.662791943633342, 26.370322391936433,
        0.8224759172413451, 43.115264767949185, 15.136741523995362, 11.460734980276309, 5.479940359693641],
    [7.764579299298634, 51.013273430153816, 2.7047963847848924, 12.33873913629235, 7.02718995154987,
        42.69156112966499, 10.825278593845171, 10.589445877035226, 1.146916095081615, 3.1869840900357844],
    [64.15317611793365, 41.104183893969946, 32.277532710823735, 8.81518852697109, 10.77246046295566,
        6.02265155260646, 2.277750113076787, 24.899932581551575, 19.399373402797558, 3.6176957778738688]]
De = Ee3[6]


# 创建类似的queueManager
class QueueManager(BaseManager):
    pass


def win_server_run(matrixA, matrixB):
    # window下多进程可能有问题，添加这句话缓解
    freeze_support()
    # 注册在网络上，callable 关联了Queue 对象
    # 将Queue对象在网络中暴露
    # window下绑定调用接口不能直接使用lambda，所以只能先定义函数再绑定
    QueueManager.register('get_task_queue', callable=get_task)
    QueueManager.register('get_result_queue', callable=get_result)
    # 绑定端口和设置验证口令
    manager = QueueManager(address=('127.0.0.1', 8001), authkey='qiye'.encode())

    # 启动管理，监听信息通道
    manager.start()

    try:

        # 通过网络获取任务队列和结果队列
        task = manager.get_task_queue()
        result = manager.get_result_queue()
        
        # 放几个任务进去:
        '''
        for i in range(30):
            n = random.randint(0, 10000)
            print('Put task %d...' % n)
            task.put(n)
        '''
        if matrixA.shape[0] % 10 == 0:
            # 在垂直方向把矩阵A分成10块
            mlist = np.vsplit(matrixA, 10)
            # print(mlist)
            for j in range(2):
                for i in range(len(mlist)):
                    # n = np.arange(0, 32, 2)
                    # n = n.reshape(4,4)
                    # print('Put task %d...' % i)
                    # print(mlist[i])
                    task.put([mlist[i], matrixB, i])

            p1 = Process(target=win_client_run, args=(De[0],))
            p1.start()
            p2 = Process(target=win_client_run, args=(De[1],))
            p2.start()
            p3 = Process(target=win_client_run, args=(De[2],))
            p3.start()
            p4 = Process(target=win_client_run, args=(De[3],))
            p4.start()
            p5 = Process(target=win_client_run, args=(De[4],))
            p5.start()
            p6 = Process(target=win_client_run, args=(De[5],))
            p6.start()
            p7 = Process(target=win_client_run, args=(De[6],))
            p7.start()
            p8 = Process(target=win_client_run, args=(De[7],))
            p8.start()
            p9 = Process(target=win_client_run, args=(De[8],))
            p9.start()
            p10 = Process(target=win_client_run, args=(De[9],))
            p10.start()

            final_result = [ [] for i in range(len(mlist))]
            a= []
            while 1:
                try:
                    if [] in final_result:
                        print('----------------reuslt.get-------------------')
                        n = result.get(1)
                        #print("获得结果 ",n[1])#查询最后一个结果，用于确认，可删除
                        if len(n):
                            final_result[n[0][1]] = n[0][0]  # 将结果矩阵的行加入对应的列表位置
                    else:
                        #final = [i[0] for i in final_result]  # 整合结果
                        print('-----------------result.vstack-----------------')
                        a = np.vstack(final_result)
                        # print(a)
                        #print(np.shape(a))
                        return a
                        break
                except:
                    #final = [i[0] for i in final_result]  # 整合结果
                    a = np.vstack(final_result)
                    return a
                    # print(a)
                    break
            
            p1.terminate()
            p2.terminate()
            p3.terminate()
            p4.terminate()
            p5.terminate()
            p6.terminate()
            p7.terminate()
            p8.terminate()
            p9.terminate()
            p10.terminate()
            return a
            '''
            while [] in final_result:
                n = result.get()
                #print("获得结果 ",n[1])#查询最后一个结果，用于确认，可删除
                final_result[n[0][1]].append(n[0][0])  # 将结果矩阵的行加入对应的列表位置
            
            final = [i[0] for i in final_result]  # 整合结果
            a = np.vstack(final)
            # print(a)
            #print(np.shape(a))
            p1.terminate()
            p2.terminate()
            p3.terminate()
            p4.terminate()
            p5.terminate()
            p6.terminate()
            p7.terminate()
            p8.terminate()
            p9.terminate()
            p10.terminate()
            return a
            '''
        elif matrixA.shape[0] % 9 == 0:
            # 在垂直方向把矩阵A分成10块
            mlist = np.vsplit(matrixA, 9)
            # print(mlist)
            for j in range(2):
                for i in range(len(mlist)):
                    # n = np.arange(0, 32, 2)
                    # n = n.reshape(4,4)
                    # print('Put task %d...' % i)
                    # print(mlist[i])
                    task.put([mlist[i], matrixB, i])
            # 从result队列读取结果:
            # print('Try get results...')
            # 这里添加了client函数.把它放到单独的一个进程中去
            p1 = Process(target=win_client_run, args=(De[0],))
            p1.start()
            p2 = Process(target=win_client_run, args=(De[1],))
            p2.start()
            p3 = Process(target=win_client_run, args=(De[2],))
            p3.start()
            p4 = Process(target=win_client_run, args=(De[3],))
            p4.start()
            p5 = Process(target=win_client_run, args=(De[4],))
            p5.start()
            p6 = Process(target=win_client_run, args=(De[5],))
            p6.start()
            p7 = Process(target=win_client_run, args=(De[6],))
            p7.start()
            p8 = Process(target=win_client_run, args=(De[7],))
            p8.start()
            p9 = Process(target=win_client_run, args=(De[8],))
            p9.start()
            p10 = Process(target=win_client_run, args=(De[9],))
            p10.start()
            
            final_result = [[] for i in range(len(mlist))]
            while 1:
                try:
                    if [] in final_result:
                         #括号内可以加上timeout=1，超过1秒就不获取这次的值了 
                        print('----------------reuslt.get-------------------')
                        n = result.get(1)
                        #print("获得结果 ", n[1])  # 查询最后一个结果，用于确认，可删除
                        if len(n):
                            final_result[n[0][1]]= n[0][0]  # 将结果矩阵的行加入对应的列表位置
                    else:
                        #final = [i[0] for i in final_result]  # 整合结果
                        print('-----------------result.vstack-----------------')
                        a = np.vstack(final_result)
                        # print(a)
                        return a
                        break
                except:
                    #final = [i[0] for i in final_result]  # 整合结果
                    a = np.vstack(final_result)
                    # print(a)
                    return a
                    break
            p1.terminate()
            p2.terminate()
            p3.terminate()
            p4.terminate()
            p5.terminate()
            p6.terminate()
            p7.terminate()
            p8.terminate()
            p9.terminate()
            p10.terminate()

            return a
            '''
            while [] in final_result:
                n = result.get()
                #print("获得结果 ",n[1])#查询最后一个结果，用于确认，可删除
                final_result[n[0][1]].append(n[0][0])  # 将结果矩阵的行加入对应的列表位置
            
            final = [i[0] for i in final_result]  # 整合结果
            a = np.vstack(final)
            # print(a)
            #print(np.shape(a))
            p1.terminate()
            p2.terminate()
            p3.terminate()
            p4.terminate()
            p5.terminate()
            p6.terminate()
            p7.terminate()
            p8.terminate()
            p9.terminate()
            p10.terminate()
            return a  
            '''   
        '''        
        if matrixA.shape[0] % 10 == 0:
            #  #开始获取结果队列中的结果,先获取一次结果，剩下9个的结果放到循环里
            #  mresult = result.get()
            #  for i in range(1,10):
            #      #括号内可以加上timeout=10，超过10秒就不获取这次的值了
            #       r = result.get()
            #       mresult = np.vstack((mresult,r))
            #       #print(mresult)
            #       #print('Result: %s' % r)
            #       #print('Result:')
            #       #print(r)
            final_result = [[] for i in range(len(mlist))]
            while 1:
                try:
                    if [] in final_result:
                        n = result.get()
                        #print("获得结果 ",n[1])#查询最后一个结果，用于确认，可删除
                        final_result[n[0][1]].append(n[0][0])  # 将结果矩阵的行加入对应的列表位置
                    else:
                        final = [i[0] for i in final_result]  # 整合结果
                        a = np.vstack(final)
                        # print(a)
                        #print(np.shape(a))
                        p1.terminate()
                        p2.terminate()
                        p3.terminate()
                        p4.terminate()
                        p5.terminate()
                        p6.terminate()
                        p7.terminate()
                        p8.terminate()
                        p9.terminate()
                        p10.terminate()
                        return a
                except:
                    final = [i[0] for i in final_result]  # 整合结果
                    a = np.vstack(final)
                    # print(a)
                    print(np.shape(a))
                    p1.terminate()
                    p2.terminate()
                    p3.terminate()
                    p4.terminate()
                    p5.terminate()
                    p6.terminate()
                    p7.terminate()
                    p8.terminate()
                    p9.terminate()
                    p10.terminate()
                    return a
                    break

        elif matrixA.shape[0] % 9 == 0:
            # 开始获取结果队列中的结果,先获取一次结果，剩下9个的结果放到循环里
            #  mresult = result.get()
            #  for i in range(1,9):
            #      #括号内可以加上timeout=10，超过10秒就不获取这次的值了
            #       r = result.get()
            #       mresult = np.vstack((mresult,r))
            #       #print(mresult)
            #       #print('Result: %s' % r)
            #       #print('Result:')
            #       #print(r)
            final_result = [[] for i in range(len(mlist))]
            while 1:
                try:
                    if [] in final_result:
                        n = result.get()
                        #print("获得结果 ", n[1])  # 查询最后一个结果，用于确认，可删除
                        final_result[n[0][1]].append(n[0][0])  # 将结果矩阵的行加入对应的列表位置
                    else:
                        final = [i[0] for i in final_result]  # 整合结果
                        a = np.vstack(final)
                        # print(a)
                        #print(np.shape(a))
                        p1.terminate()
                        p2.terminate()
                        p3.terminate()
                        p4.terminate()
                        p5.terminate()
                        p6.terminate()
                        p7.terminate()
                        p8.terminate()
                        p9.terminate()
                        p10.terminate()
                        return a
                except:
                    final = [i[0] for i in final_result]  # 整合结果
                    a = np.vstack(final)
                    # print(a)
                    print(np.shape(a))
                    p1.terminate()
                    p2.terminate()
                    p3.terminate()
                    p4.terminate()
                    p5.terminate()
                    p6.terminate()
                    p7.terminate()
                    p8.terminate()
                    p9.terminate()
                    p10.terminate()
                    return a
                    break
    '''
    except:
        print('Manager error')
    
    finally:
        manager.shutdown()
    
    manager.shutdown()

'''
if __name__ == '__main__':
    win_server_run()
'''