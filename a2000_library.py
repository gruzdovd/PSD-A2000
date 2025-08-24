import ctypes
import math
import platform
import time

import numpy as np

if platform.system() == "Windows":
    libname = "E:\\PSD_calculation\\model_library_f77\\a2000_module.dll"
    _lib = ctypes.WinDLL(libname)
else:
    libname = "/home/gruzdov/a_2000/a2000_module.so"
    _lib = ctypes.CDLL(libname)

a2000_module = _lib

class OutputData(ctypes.Structure):
    _fields_ = [
        ("output_array", (ctypes.c_float * 9) * 100000),  # [параметры][точки]
        ("num_points", ctypes.c_int)
    ]

class TranCommon(ctypes.Structure): #COMMON блок /TRAN/ в fortran
    _fields_ = [
        ("g2gsm", (ctypes.c_float * 3) * 3),
        ("UTt", ctypes.c_float),
        ("IDAYt", ctypes.c_int),
        ("iyeart", ctypes.c_int),
        ("datt", ctypes.c_float)
    ]

output_data = OutputData()
trans_common = TranCommon()

# Получение доступа к COMMON-блоку
output_data = ctypes.cast(
    getattr(a2000_module, "output_data_"),  # Имя в DLL для COMMON блока OUTPUT_DATA: output_data_ (лишнее подчеркивание)
    ctypes.POINTER(OutputData)
).contents

tran_common = ctypes.cast(
    getattr(a2000_module, "tran_"),
    ctypes.POINTER(TranCommon)
).contents

# a2000_main - функция для расчета магнитной линии
a2000_module.a2000_main_.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # X1
    ctypes.POINTER(ctypes.c_float),  # X2
    ctypes.POINTER(ctypes.c_float),  # X3
    ctypes.POINTER(ctypes.c_float),  # X4
    ctypes.POINTER(ctypes.c_float),  # X5
    ctypes.POINTER(ctypes.c_float),  # X6
    ctypes.POINTER(ctypes.c_float),  # X7
    ctypes.POINTER(ctypes.c_float),  # ut (REAL в Fortran)
    ctypes.POINTER(ctypes.c_int32),  # iy
    ctypes.POINTER(ctypes.c_int32),  # mo
    ctypes.POINTER(ctypes.c_int32),  # id
    ctypes.POINTER(ctypes.c_float),  # ro
    ctypes.POINTER(ctypes.c_float),  # v
    ctypes.POINTER(ctypes.c_float),   # bimf[3] (аргумент 14)
    ctypes.POINTER(ctypes.c_float),  # dst
    ctypes.POINTER(ctypes.c_float),  # al
    ctypes.POINTER(ctypes.c_float),  # x[3]
    ctypes.POINTER(ctypes.c_float),  # sl
    ctypes.POINTER(ctypes.c_float)   # h
]

#pere2 - функция для перехода в другую систему координат
a2000_module.pere2_.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # A, вектор исходных координат (3)
    ctypes.POINTER(ctypes.c_float),  # B, вектор новых координат (3)
    ctypes.POINTER(ctypes.c_float),  # T, матрица перехода (3,3)
    ctypes.POINTER(ctypes.c_int),  # K, индекс перехода, при K>0 используется T, при K<=0 используется T^{-1}
]

a2000_module.a2000f_.argtypes = [
    ctypes.POINTER(ctypes.c_float), # ut, вектор исходной точки
    ctypes.POINTER(ctypes.c_int), # iy
    ctypes.POINTER(ctypes.c_int), # mo
    ctypes.POINTER(ctypes.c_int), # id
    ctypes.POINTER(ctypes.c_float), # ro
    ctypes.POINTER(ctypes.c_float), # v
    ctypes.POINTER(ctypes.c_float), # bimf
    ctypes.POINTER(ctypes.c_float), # dst
    ctypes.POINTER(ctypes.c_float), # al
    ctypes.POINTER(ctypes.c_float), # x0
    ctypes.POINTER(ctypes.c_float), # bm
    ctypes.POINTER(ctypes.c_float) # bdd
]

a2000_module.pstatus_.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # X1
    ctypes.POINTER(ctypes.c_float),  # X2
    ctypes.POINTER(ctypes.c_float),  # X3
    ctypes.POINTER(ctypes.c_float),  # X4
    ctypes.POINTER(ctypes.c_float),  # X5
    ctypes.POINTER(ctypes.c_float),  # X6
    ctypes.POINTER(ctypes.c_float),  # X7
]

a2000_module.p_field_.argtypes = [
    ctypes.POINTER(ctypes.c_float), # "x0", вектор исходной точки
    ctypes.POINTER(ctypes.c_float), # "par",
    ctypes.POINTER(ctypes.c_float), # "bm" - output
    ctypes.POINTER(ctypes.c_float) # "bdd - output
]

a2000_module.submod_.argtypes = [
    ctypes.POINTER(ctypes.c_float), # "ut"
    ctypes.POINTER(ctypes.c_int), # "iy"
    ctypes.POINTER(ctypes.c_int), # "mo"
    ctypes.POINTER(ctypes.c_int), # id
    ctypes.POINTER(ctypes.c_float), # ro
    ctypes.POINTER(ctypes.c_float), # v
    ctypes.POINTER(ctypes.c_float), # bimf
    ctypes.POINTER(ctypes.c_float), # dst
    ctypes.POINTER(ctypes.c_float), # al
    ctypes.POINTER(ctypes.c_float) # par
]

#trans - функция для расчета угла наклона геомагнитного диполя
a2000_module.trans_.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # UT (Universal Time)
    ctypes.POINTER(ctypes.c_int),  # IDAY (день года)
    ctypes.POINTER(ctypes.c_int),  # IYEAR (год)
    ctypes.POINTER(ctypes.c_float),  # tpsi (выходной параметр, нужен в gsm2sm и sm2gsm)
    ctypes.POINTER(ctypes.c_float)  # BD (выходной параметр, нужен в )
]


def r2_calculation(ut,iy,mo,id,ro,v,bimf,dst,al):
    """

    :param ut:
    :param iy:
    :param mo:
    :param id:
    :param ro:
    :param v:
    :param bimf:
    :param dst:
    :param al:
    :return:
    """
    bimf_arr = np.array(bimf, dtype=np.float32)
    par = np.zeros(10, dtype=np.float32)

    args_submod = [
        ctypes.byref(ctypes.c_float(ut)),
        ctypes.byref(ctypes.c_int(iy)),
        ctypes.byref(ctypes.c_int(mo)),
        ctypes.byref(ctypes.c_int(id)),
        ctypes.byref(ctypes.c_float(ro)),
        ctypes.byref(ctypes.c_float(v)),
        bimf_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(ctypes.c_float(dst)),
        ctypes.byref(ctypes.c_float(al)),
        par.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    ]
    a2000_module.submod_(*args_submod)

    return par[6]


def field(X1, X2, X3, X4, X5, X6, X7, ut, iy, mo, id, ro, v, bimf, dst, al, x0):
    """
    Функция рассчитывает вектор (3 компоненты) магнитного поля в SM координатах.
    :param X1: IGRF/dipole field on/off (2/1/0)
    :param X2: RC field on/off (1/0)
    :param X3: tail current field on/off (1/0)
    :param X4: dipole shielding field on/off (1/0)
    :param X5: RC shielding field IMF on/off (1/0)
    :param X6: Region 1 FAC field on/off (1/0)
    :param X7: IMF on/off (1/0)
    :param ut: время UT, (float)
    :param iy: год, (int)
    :param mo: месяц, (int)
    :param id: день месяца, (int)
    :param ro: плотность солнечного ветра
    :param v: скорость солнечного ветра
    :param bimf: компоненты ММП
    :param dst: Dst-индекс, нТл
    :param al: AL-индекс, нТл
    :param x0: [x0, y0, z0], точка, в которой рассчитывается поле, RE, SM
    :return: [bx, by, bz] - вектор магнитного поля в SM координатах
    """
    bimf_arr = np.array(bimf, dtype=np.float32)
    x0_gsm = sm2gsm(x0, ut, id, mo, iy)
    x0_arr = np.array(x0_gsm, dtype=np.float32)

    bm = np.zeros(3, dtype=np.float32)
    bdd = np.zeros((7, 3), dtype=np.float32, order='F')

    args_pstatus = [
        ctypes.byref(ctypes.c_float(X1)),
        ctypes.byref(ctypes.c_float(X2)),
        ctypes.byref(ctypes.c_float(X3)),
        ctypes.byref(ctypes.c_float(X4)),
        ctypes.byref(ctypes.c_float(X5)),
        ctypes.byref(ctypes.c_float(X6)),
        ctypes.byref(ctypes.c_float(X7)),
    ]

    a2000_module.pstatus_(*args_pstatus)

    args_a2000f = [
        ctypes.byref(ctypes.c_float(ut)),
        ctypes.byref(ctypes.c_int(iy)),
        ctypes.byref(ctypes.c_int(mo)),
        ctypes.byref(ctypes.c_int(id)),
        ctypes.byref(ctypes.c_float(ro)),
        ctypes.byref(ctypes.c_float(v)),
        bimf_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(ctypes.c_float(dst)),
        ctypes.byref(ctypes.c_float(al)),
        x0_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        bm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        bdd.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    ]

    a2000_module.a2000f_(*args_a2000f)

    bm_sm = gsm2sm(bm, ut, id, mo, iy)

    return bm_sm


def magnetic_moment(ut, id, mo, iy):
    """

    :param ut:
    :param id:
    :param iy:
    :return:
    """
    iday = idd(iy, mo, id)
    r_e = 6378.16

    ut = ctypes.c_float(ut)
    iday = ctypes.c_int(iday)
    iyear = ctypes.c_int(iy)
    tpsi = ctypes.c_float(0.0)
    BD = ctypes.c_float(0.0)

    a2000_module.trans_(ctypes.byref(ut), ctypes.byref(iday), ctypes.byref(iyear), ctypes.byref(tpsi), ctypes.byref(BD))

    M = np.abs(float(BD.value)) #* 10 ** (-5) #* r_e ** 3 * 10 ** (15)

    return M

def gsm2sm(x0, ut, id, mo, iyear):
    """
    Функция из исходного вектора x0 в координатах GSM возвращает вектор x в SM
    :param x0: [x0, y0, z0] - вектор исходной точки в GSM
    :param ut: время UT, вещественное число
    :param id: день МЕСЯЦА, внутри функции переводится в день ГОДА iday, который нужен для trans_, целое числа
    :param mo: номер месяца
    :param iyear: год, целое число
    :return: x: [x, y, z] - вектор в SM
    """
    iday = idd(iyear, mo, id)

    x0_arr = np.array(x0, dtype=np.float32)
    x_arr = np.zeros(3, dtype=np.float32)
    sm2gsm_arr = np.zeros((3,3), dtype=np.float32, order='F') #order='F' нужен, т.к. в Pythom row-major, а в Fortran column-major
    k = ctypes.c_int(-1)

    ut = ctypes.c_float(ut)
    iday = ctypes.c_int(iday)
    iyear = ctypes.c_int(iyear)
    tpsi = ctypes.c_float(0.0)
    BD = ctypes.c_float(0.0)

    a2000_module.trans_(ctypes.byref(ut), ctypes.byref(iday), ctypes.byref(iyear), ctypes.byref(tpsi), ctypes.byref(BD))

    tpsi = float(tpsi.value)
    tpsi_rad = math.radians(tpsi)
    cos_psi = np.cos(tpsi_rad)
    sin_psi = np.sin(tpsi_rad)
    sm2gsm_matrix = np.array([
        [cos_psi, 0.0, -sin_psi],
        [0.0, 1.0, 0.0],
        [sin_psi, 0.0, cos_psi]
    ], dtype=np.float32)

    sm2gsm_arr[:, :] = sm2gsm_matrix

    args_pere2 = [
        x0_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        x_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        sm2gsm_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(k)
    ]

    a2000_module.pere2_(*args_pere2)

    return x_arr


def idd(iy, mo, id):
    """
    Вычисление порядкового номера дня в году.
    Функция нужна для правильного использования TRANS_ в gsm2sm и sm2gsm
    :param iy: Год.
    :param mo: Месяц (1-12).
    :param id: День месяца (1-31).

    Возвращает:
    int: Порядковый номер дня в году.
    """
    # Количество дней в каждом месяце для невисокосного года
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # Проверка на високосный год
    is_leap_year = (iy % 4 == 0 and (iy % 100 != 0 or iy % 400 == 0))
    if is_leap_year:
        days_in_month[1] = 29  # Февраль в високосном году имеет 29 дней

    # Сумма дней в предыдущих месяцах
    total_days = sum(days_in_month[:mo - 1]) + id

    return total_days


def sm2gsm(x0, ut, id, mo, iyear):
    """
    Функция из исходного вектора x0 в координатах SM возвращает вектор x в GSM
    :param x0: [x0, y0, z0] - вектор исходной точки в SM
    :param ut: время UT, вещественное число
    :param id: день МЕСЯЦА, внутри функции переводится в день ГОДА iday, который нужен для trans_, целое числа
    :param mo: номер месяца
    :param iyear: год, целое число
    :return: x: [x, y, z] - вектор в GSM
    """
    iday = idd(iyear, mo, id)

    x0_arr = np.array(x0, dtype=np.float32)
    x_arr = np.zeros(3, dtype=np.float32)
    sm2gsm_arr = np.zeros((3,3), dtype=np.float32, order='F') #order='F' нужен, т.к. в Python row-major, а в Fortran column-major
    k = ctypes.c_int(1)

    ut = ctypes.c_float(ut)
    iday = ctypes.c_int(iday)
    iyear = ctypes.c_int(iyear)
    tpsi = ctypes.c_float(0.0)
    BD = ctypes.c_float(0.0)

    a2000_module.trans_(ctypes.byref(ut), ctypes.byref(iday), ctypes.byref(iyear), ctypes.byref(tpsi), ctypes.byref(BD))

    tpsi = float(tpsi.value)
    tpsi_rad = math.radians(tpsi)
    cos_psi = np.cos(tpsi_rad)
    sin_psi = np.sin(tpsi_rad)
    sm2gsm_matrix = np.array([
        [cos_psi, 0.0, -sin_psi],
        [0.0, 1.0, 0.0],
        [sin_psi, 0.0, cos_psi]
    ], dtype=np.float32)

    sm2gsm_arr[:, :] = sm2gsm_matrix

    args_pere2 = [
        x0_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        x_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        sm2gsm_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(k)
    ]
    a2000_module.pere2_(*args_pere2)
    return x_arr


def g2gsm(x0, ut, id, mo, iyear):
    """
    Переводит координаты из системы MAG в систему GSM, используя матрицу из TRANS.

    :param x_geo: [x, y, z] в географических координатах
    :param ut: время UT
    :param id: день месяца
    :param mo: номер месяца
    :param iyear: год
    :return: [x, y, z] в GSM координатах
    """
    iday = idd(iyear, mo, id)

    x0_arr = np.array(x0, dtype=np.float32)

    g2gsm_arr = np.zeros((3,3), dtype=np.float32, order='F') #order='F' нужен, т.к. в Python row-major, а в Fortran column-major
    k = ctypes.c_int(1)

    ut = ctypes.c_float(ut)
    iday = ctypes.c_int(iday)
    iyear = ctypes.c_int(iyear)
    tpsi = ctypes.c_float(0.0)
    BD = ctypes.c_float(0.0)

    a2000_module.trans_(ctypes.byref(ut), ctypes.byref(iday), ctypes.byref(iyear), ctypes.byref(tpsi), ctypes.byref(BD))

    for i in range(3):
        for j in range(3):
            g2gsm_arr[i][j] = tran_common.g2gsm[i][j]

    x_gsm_vec = np.dot(g2gsm_arr, x0_arr)

    return x_gsm_vec


def gsm2g(x0, ut, id, mo, iyear):
    """
    Переводит координаты из системы GSM в систему MAG, используя матрицу из TRANS.

    :param x_0: [x, y, z] в GSM координатах
    :param ut: время UT
    :param id: день месяца
    :param mo: номер месяца
    :param iyear: год
    :return: [x, y, z] в MAG координатах
    """
    iday = idd(iyear, mo, id)

    x0_arr = np.array(x0, dtype=np.float32)
    g2gsm_arr = np.zeros((3,3), dtype=np.float32, order='F') #order='F' нужен, т.к. в Python row-major, а в Fortran column-major

    ut = ctypes.c_float(ut)
    iday = ctypes.c_int(iday)
    iyear = ctypes.c_int(iyear)
    tpsi = ctypes.c_float(0.0)
    BD = ctypes.c_float(0.0)

    a2000_module.trans_(ctypes.byref(ut), ctypes.byref(iday), ctypes.byref(iyear), ctypes.byref(tpsi), ctypes.byref(BD))

    for i in range(3):
        for j in range(3):
            g2gsm_arr[i][j] = tran_common.g2gsm[i][j]

    gsm2g_matrix = g2gsm_arr.T

    x_geo_vec = np.dot(gsm2g_matrix, x0_arr)

    return x_geo_vec


def one_step_line(X1, X2, X3, X4, X5, X6, X7,
               ut, iy, mo, id_val, ro, v, bimf,
               dst, al, x, sl, h):
    """
    Вызывает Fortran-подпрограмму a2000_main для шага в одну сторону.
    Функция ничего не возвращает, результат расчета записывается в поле outData.output_array
    """
    # Приводим входные массивы к нужному типу.
    bimf_arr = np.array(bimf, dtype=np.float32)
    x_arr = np.array(x, dtype=np.float32)

    args = [
        ctypes.byref(ctypes.c_float(X1)),
        ctypes.byref(ctypes.c_float(X2)),
        ctypes.byref(ctypes.c_float(X3)),
        ctypes.byref(ctypes.c_float(X4)),
        ctypes.byref(ctypes.c_float(X5)),
        ctypes.byref(ctypes.c_float(X6)),
        ctypes.byref(ctypes.c_float(X7)),
        ctypes.byref(ctypes.c_float(ut)),
        ctypes.byref(ctypes.c_int32(iy)),
        ctypes.byref(ctypes.c_int32(mo)),
        ctypes.byref(ctypes.c_int32(id_val)),
        ctypes.byref(ctypes.c_float(ro)),
        ctypes.byref(ctypes.c_float(v)),
        bimf_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(ctypes.c_float(dst)),
        ctypes.byref(ctypes.c_float(al)),
        x_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(ctypes.c_float(sl)),
        ctypes.byref(ctypes.c_float(h))
    ]
    a2000_module.a2000_main_(*args)
    return


def _trace_line_impl(X1, X2, X3, X4, X5, X6, X7,
               ut, iy, mo, id_val, ro, v, bimf,
               dst, al, x, sl, h):
    """
    Функция рассчитывает магнитную линию по входным параметрам в обе стороны.
        Input:
            X1, ..., X7 - параметры включения различных источников в модель
            ut - время UT в виде вещественного числа
            iy - год
            mo - месяц
            id_val - день
            ro - плотность солнечного ветра
            v - скорость солнечного ветра
            bimf - (b1,b2,b3), компоненты ММП
            dst - dst-индекс
            al - al-индекс
            x - (x,y,z), RE_GSM, исходная точка, из которой считается линия
            sl - максимальная длина линии
            h - шаг расчета
        Output:
            Массив (N,9) с данными о магнитной линии. Столбцы массива:
            x   y   z   B_total     B_x     B_y     B_z     MLT     R
    """

    #   Очень важно создать два отдельных буфера для расчета с шагами h и -h, т.к. output_data.output_array используется для обоих
    #   вызовов и в конце выдаст тот массив, который был записан при последнем вызове one_step_line.

    output_neg = ((ctypes.c_float * 9) * 100000)()
    output_pos = ((ctypes.c_float * 9) * 100000)()

    output_data.output_array = output_neg
    one_step_line(X1, X2, X3, X4, X5, X6, X7, ut, iy, mo, id_val, ro, v, bimf, dst, al, x, sl, -h)
    output_neg = output_data.output_array

    num_points_neg = output_data.num_points
    np_neg = np.ctypeslib.as_array(output_neg)[:num_points_neg].copy() #copy нужно для того, чтобы потом данные не перезаписались на np_pos

    output_data.output_array = output_pos
    one_step_line(X1, X2, X3, X4, X5, X6, X7, ut, iy, mo, id_val, ro, v, bimf, dst, al, x, sl, h)
    output_pos = output_data.output_array
    num_points_pos = output_data.num_points
    np_pos = np.ctypeslib.as_array(output_pos)[:num_points_pos].copy()

    # [1:] здесь убирает точку старта (точку спутника), т.к. она дублируется для разных направлений
    # [::-1] переворачивает массив вверх ногами, т.е. чтобы выходной массив всегда содержал структуру юг->север
    return np.vstack([np_neg[1:][::-1], np_pos])


def trace_line(*args, **kwargs):
    if len(args) == 1 and isinstance(args[0], dict):
        params = args[0]
        return _trace_line_impl(**params)
    elif not args and kwargs:
        return _trace_line_impl(**kwargs)
    elif len(args) == 19:
        return _trace_line_impl(*args)
    else:
        raise TypeError("Функция trace_line, неправильные аргументы")