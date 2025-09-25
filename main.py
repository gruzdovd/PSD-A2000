import a2000_library
import re
import numpy as np
import math
#import spacepy.omni as omni
import io
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline, interp1d, PchipInterpolator
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
import time


def mirror_points_pitch_angle(x, B, pitch_angle, x_sat, equatorial=False): #pitch_angle в градусах
    """
    Функция вычисляет северную и южную зеркальные точки и зеркальное магнитное поле для входного питч-угла (на экваторе или в точке спутника).
    :param x: список [x,y,z] координат магнитной линии, RE
    :param B: список значений магнитного поля в точках x, нТл
    :param pitch_angle: питч-угол, для которого считается зеркальая точка, в зависимости от equatorial либо на экваторе, либо локальный на спутнике
    :param x_sat: положение спутника, RE
    :param equatorial: флаг, если True, то питч-угол во входных данных экваториальный, если False - локальный (на спутнике). По умолчанию локальный
    :return: список значений [[x_n, y_n, z_n], [x_s, y_s, z_s], B_mirror]
    """
    B_min = min(B)
    idx_min = np.argmin(B)
    if idx_min == 0 or idx_min == len(B)-1:
        #Подобное условие сигнализирует о том, что магнитная линия незамкнута
        return None, None, None

    x, B = np.array(x), np.array(B)
    x_sat = np.array(x_sat)

    if equatorial:
        B_mirror = float(B_min / (np.sin(float(pitch_angle * np.pi / 180)))**2)
    else:
        matches = np.isclose(x, x_sat, atol=1e-3).all(axis=1) #совпадение x и x_sat с точностью до 3 знака после запятой (по умолчанию они полностью совпадают без округления)
        index_for_b_sat = np.where(matches)[0]
        B_sat = B[index_for_b_sat[0]]  # поле по модели в точке спутника.
        B_mirror = float(B_sat / (np.sin(float(pitch_angle * np.pi / 180))) ** 2)
    indices_south = np.arange(0, idx_min + 1)  # Индексы первой группы (юг)
    indices_north = np.arange(idx_min, len(B))  # Индексы второй группы (север)

    #Вычисления для севера
    B_north = B[indices_north]
    x_north = x[indices_north]
    differences_north = B_mirror - B_north
    valid_indices_north = np.where(differences_north >= 0)[0] # индексы только положительных значений Bm-B, т.к. ищем ближайшее B, такое что B<=Bm
    closest_index_north = valid_indices_north[np.argmin(differences_north[valid_indices_north])] # индекс ближайшего B<= Bm
    mirror_point_north = x_north[closest_index_north]

    # Вычисления для юга
    B_south = B[indices_south]
    x_south = x[indices_south]
    differences_south = B_mirror - B_south
    valid_indices_south = np.where(differences_south >= 0)[0]
    closest_index_south = valid_indices_south[np.argmin(differences_south[valid_indices_south])]
    mirror_point_south = x_south[closest_index_south]

    return [mirror_point_north, mirror_point_south, B_mirror]


def mirror_points_b_mirr(x, B, B_mirror):
    """
    Функция вычисляет северную и южную зеркальные точки по уже заданному зеркальному полю
    :param x: список [x,y,z] координат магнитной линии, RE
    :param B: список значений магнитного поля в точках x, нТл
    :param B_mirror: зеркальное поле, нТл
    :return: список значений [[x_n, y_n, z_n], [x_s,y_s, z_s]]
    """
    idx_min = np.argmin(B)  # индекс минимального B
    if idx_min == 0 or idx_min == len(B)-1:
        #магнитная линия незамкнута
        return None, None
    if B_mirror < min(B):
        #Подобное условие - попали на слишком близкую к Земле магнитную линию, она нам не нужна
        return None, None

    x, B = np.array(x), np.array(B)

    indices_south = np.arange(0, idx_min + 1)  # Индексы первой группы (юг)
    indices_north = np.arange(idx_min, len(B))  # Индексы второй группы (север)

    #Вычисления для севера
    B_north = B[indices_north]
    x_north = x[indices_north]
    differences_north = B_mirror - B_north
    valid_indices_north = np.where(differences_north >= 0)[0] # индексы только положительных значений Bm-B, т.к. ищем ближайшее B, такое что B<=Bm
    closest_index_north = valid_indices_north[np.argmin(differences_north[valid_indices_north])] # индекс ближайшего B<= Bm
    mirror_point_north = x_north[closest_index_north]

    # Вычисления для юга
    B_south = B[indices_south]
    x_south = x[indices_south]
    differences_south = B_mirror - B_south
    valid_indices_south = np.where(differences_south >= 0)[0]
    closest_index_south = valid_indices_south[np.argmin(differences_south[valid_indices_south])]
    mirror_point_south = x_south[closest_index_south]

    return [mirror_point_north, mirror_point_south]


def curve_integral(x, B, mirror_point_north, mirror_point_south, B_mirror):
    """
    Вычисление второго адиабатического инварианта K = curve integral( sqrt(B_mirror-B)ds ). Питч-угол сюда не подается на вход, это сделано для того, чтобы не проверять, какие
    питч углы - экваториальные или локальные. Вместо этого подается сразу B_mirror, которая посчитана уже с учетом экваториальности/локальности
    питч-угла.
    :param x: список [x,y,z] координат магнитной линии, RE
    :param B: список значений магнитного поля в точках x, нТл
    :param mirror_point_north: северная зеркальная точка [x,y,z], верхний предел интегрирования
    :param mirror_point_south: южная зеркальная точка [x,y,z], нижний предел интегрирования
    :param B_mirror: зеркальное поле, нТл
    :return: Значение интеграла K, Гс^(-1/2)*км
    """

    B_values = np.array(B)
    if (mirror_point_north is None) or (mirror_point_south is None) or (B_mirror is None):
        return None
    start_index = np.where((np.array(x) == mirror_point_south).all(axis=1))[0][0]   #индекс южной зеркальной точки
    end_index = np.where((np.array(x) == mirror_point_north).all(axis=1))[0][0]     #индекс северной зеркальной точки

    coordinates = np.array(x)
    d_coordinates = np.diff(coordinates, axis = 0)                                                  # поэлементная разность
    dx, dy, dz = d_coordinates[:, 0], d_coordinates[:, 1], d_coordinates[:, 2]                      #dx,dy,dz - массивы разностей x1-x2, x3-x2 и т.д.
    segment_length = np.sqrt(dx**2 + dy**2 + dz**2)                                                 # массив элементов кривой dl
    arc_length = np.insert(np.cumsum(segment_length), 0, 0)[start_index:end_index + 1]   # массив накопленных длин дуг в пределах интегрирования, т.е. [0, l1, l1+l2,..., L]

    diff = B_mirror - B_values[start_index:end_index + 1]
    diff[diff < 0] = np.nan
    f = np.sqrt(diff, where=diff >= 0, out=np.full_like(diff, np.nan))                         # поточечная подынтегральная функция в диапазоне пределов интегрирования

    if np.any(np.isnan(f)):
        return None

    integral = simpson(f, x = arc_length)

    return integral * np.sqrt(10 ** (-5))*6371 # Гс^(-1/2)*км


def interpolation(x, y, method="cubic"):
    """
    Функция, интерполирующая y(x)
    :param x: массив данных x
    :param y: массив данных y
    :param method: метод интерполяции
    :return: интерполированная функция
    """
    y = np.array(y)
    x = np.array(x)

    valid_mask = []
    for xi, yi in zip(x, y):
        if xi is not None and yi is not None:
            valid_mask.append(True)
        else:
            valid_mask.append(False)

    x_valid = x[valid_mask].astype(float)
    y_valid = y[valid_mask].astype(float)

    if len(x_valid)<2:
        return None

    if method in ["cubic", "pchip"]: #проверка на монотонность
        diffs = np.diff(x_valid)
        if not np.all(diffs > 0):
            return None

    try:
        if method == "cubic":
            return CubicSpline(x_valid, y_valid)
        elif method == "pchip":
            return PchipInterpolator(x_valid, y_valid)
        elif method == "linear":
            return interp1d(x_valid, y_valid, kind="linear", fill_value="extrapolate")
    except Exception as e:
        print(f"Ошибка при создании интерполятора: {e}")
        return None


def read_van_allen_SW_data(filename_van_allen, filename_SW, filename_magfield, filename_gsm_coordinates):
    """
        Считывает CSV-файл для RBSP и CSV-файл для параметров солнечного ветра SW, объединяет их в один dataframe.

        Для столбцов, имена которых содержат информацию о потоке частиц вида:
          FEDU_Flux_(@_8.182E+00_degrees)_(@_3.300E+01_keV)__cm^(-2)s^(-1)sr^(-1)keV^(-1)
        функция извлекает значение питч-угла и энергии.

        Остальные столбцы попадают в мультииндекс как (None, original_name).

        Возвращает:
          pd.DataFrame: DataFrame с мультииндексом столбцов, где уровень "Energy" и "Pitch"
          определяют параметры потоков (если применимо).
    """
    df_van_allen = pd.read_csv(filename_van_allen, comment="#") #dataframe
    # Регулярное выражение для поиска значений питч-угла и энергии.
    # Предполагается, что шаблон: @_pitchValue_degrees)_(@_energyValue_keV
    pattern = r'@_([\d.E+-]+)_degrees\)_\(@_([\d.E+-]+)_keV'

    new_columns = []
    for col in df_van_allen.columns:
        m = re.search(pattern, col)
        if m:
            # m.group(1) - питч, m.group(2) - энергия.
            pitch = float(m.group(1))
            energy = float(m.group(2))
            new_columns.append(("Energy_"+str(energy), "PitchAngle_"+str(pitch)))
        else:
            if col == "EPOCH____yyyy-mm-ddThh:mm:ss.sssZ":
                new_columns.append("time_vap_tmp")#вспомогательное навание
            elif col == "L____":
                new_columns.append("L")
            else:
                new_columns.append(col)

    df_van_allen.columns = new_columns
    #Удаляем те столбцы, которые содержат энергию "-1e+31", они просто не нужны
    cols_to_drop = [col for col in df_van_allen.columns if col[0].startswith("Energy_") and float(col[0].split("_")[1]) == -1e+31]
    # Удаляем такие столбцы
    df_van_allen.drop(columns=cols_to_drop,inplace=True)
    # Присваиваем мультииндекс DataFrame.

    df_van_allen["Date/time_VAP"] = pd.to_datetime(df_van_allen["time_vap_tmp"])
    df_van_allen["Date/time_VAP"] = df_van_allen["Date/time_VAP"].dt.tz_convert(None) ####
    df_van_allen.drop(columns=["time_vap_tmp"],inplace=True)
    df_van_allen["Time_floor"] = df_van_allen["Date/time_VAP"].dt.round('min') # исключительно вспомогательный столбец, который округляет значение времени до ближайшей минуты, чтобы состыковаться с OMNI
    df_van_allen["Time_floor"] = pd.to_datetime(df_van_allen["Time_floor"]).dt.tz_localize(None) #Для успешного объединения убираем локализацию UTC
    df_van_allen["Time_floor"] = df_van_allen["Time_floor"].dt.strftime("%Y-%m-%d %H:%M:%S") #приводим к формату yyyy-mm-dd hh:mm:ss
    df_van_allen["UT"] = round(df_van_allen["Date/time_VAP"].dt.hour + df_van_allen["Date/time_VAP"].dt.minute / 60 + df_van_allen["Date/time_VAP"].dt.second / 3600 , 3)
    df_van_allen["Year"] = df_van_allen["Date/time_VAP"].dt.year
    df_van_allen["Month"] = df_van_allen["Date/time_VAP"].dt.month
    df_van_allen["Day"] = df_van_allen["Date/time_VAP"].dt.day

    #df_van_allen = df_van_allen.rename(columns={"BX__GSE_nT": "BX_GSM_nT", "Label_not_found____Re": "X_GEO", "Label_not_found____Re.1": "Y_GEO", "Label_not_found____Re.2": "Z_GEO"})

    #df_van_allen[["X_GEO", "Y_GEO", "Z_GEO"]] /= 6371  # Переводим в радиусы Земли (Re)
    #df_van_allen[['X_GSM', 'Y_GSM', 'Z_GSM']] = df_van_allen.apply(convert_GEO_to_GSM, axis=1)


    #geo_coords = np.array(df_van_allen[["X_GEO", "Y_GEO", "Z_GEO"]].values)
    #toolbox.update(QDomni=True)  # Загружает OMNI-данные
    # Преобразуем время в формат Ticktock (нужно для SpacePy)
    #time_ticks = Ticktock(df_van_allen["Date/time_VAP"].dt.strftime('%Y-%m-%dT%H:%M:%S.%3f'), 'UTC')

    #print(omni.get_omni(time_ticks))
    #maginput = {
    #    'Kp': np.full(len(time_ticks.UTC), 2),  # Используем Kp=2 как пример
    #}
    #gsm_coords = get_Lstar(time_ticks, geo_coords, maginput)['gsm']
    #df_van_allen["X_GSM"] = gsm_coords[:, 0]
    #df_van_allen["Y_GSM"] = gsm_coords[:, 1]
    #df_van_allen["Z_GSM"] = gsm_coords[:, 2]
    #df_van_allen["Date/time_VAP"] = df_van_allen["Date/time_VAP"].dt.strftime("%Y-%m-%d %H:%M:%S")  # изменение формата

    #Далее считываем данные о SW
    with open(filename_SW, 'r') as file_SW:
        lines = file_SW.readlines()        #считываем все строки

    column_names_SW = []
    for line in lines:
        if "--" in line: #строки заголовка
            parts = line.split("--") # разбиваем на номер - название
            column_name = parts[1].strip().strip(";") # название столбца по заголовку, strip() и strip(';') удаля.т лишние пробелы и ';' из названия
            column_names_SW.append(column_name)

    data_start_index = len(column_names_SW) + 1  # индекс начала данных, "+1" нужно для пропуска строки "1 2 3 ..."

    df_SW = pd.read_csv(filename_SW, skiprows=data_start_index, header=None)
    df_SW.columns = column_names_SW
    df_SW["Time_SW"] = pd.to_datetime(df_SW["Date/time"])
    df_SW.drop(columns=["Date/time"], inplace=True)
    if df_SW["Time_SW"].dt.tz is not None: # убираем локализацию
        df_SW["Time_SW"] = df_SW["Time_SW"].dt.tz_convert(None)

    if "Dst [nT]" in df_SW.columns:
        df_SW["Dst [nT]"] = pd.to_numeric(df_SW["Dst [nT]"], errors='coerce')  # заполняем 'N/A' числовыми значениями NaN
        df_SW["Dst [nT]"] = df_SW["Dst [nT]"].ffill()  # заполняем NaN повторяющимися значениями
    df_SW["Time_SW"] = df_SW["Time_SW"].dt.strftime("%Y-%m-%d %H:%M:%S")  # приводим к формату yyyy-mm-dd hh:mm:ss

    #hourly_df = df_SW.dropna(subset=['Dst [nT]'])
    #new_rows = []
    #for _, row in hourly_df.iterrows():
    #    time = row['Time_SW']
    #    dst_value = row['Dst [nT]']
#
    #    # Заполнение за полчаса до и после
    #    for minute in range(-30, 31):  # от -30 до +30 минут
    #        new_time = time + pd.Timedelta(minutes=minute)
    #        new_rows.append({'Time_SW': new_time, 'Dst [nT]': dst_value})
    #filled_df = pd.DataFrame(new_rows)
    ## Объединяем с исходным DataFrame
    #df_SW_final = pd.merge(df_SW, filled_df, on='Time_SW', how='left', suffixes=('', '_filled'))
    ## Заполняем пропуски
    #df_SW_final['Dst [nT]'] = df_SW_final['Dst [nT]'].combine_first(df_SW_final['Dst [nT]_filled'])
    ## Удаляем лишний столбец
    #df_SW_final.drop(columns=['Dst [nT]_filled'], inplace=True)
    #df_SW_final["Time_SW"] = df_SW_final["Time_SW"].dt.strftime("%Y-%m-%d %H:%M:%S")  # приводим к формату yyyy-mm-dd hh:mm:ss

    df_merged = df_van_allen.merge(df_SW, left_on="Time_floor", right_on="Time_SW", how="left") #Объединенный dataframe (и OMNI и VAP)
    columns_order = (["Date/time_VAP", "UT", "Year", "Month", "Day"] + list(df_SW.columns[:]) +
                     #["X_GSM", "Y_GSM", "Z_GSM"] +
                     [col for col in df_van_allen.columns if  col not in ["Date/time_VAP", "UT", "Year", "Month", "Day", "Time_floor", "X_GSM", "Y_GSM", "Z_GSM"]]) #немного иной порядок столбцов, просто для удобства проверки
    df_merged = df_merged[columns_order] #перестановка стобцов в соответствии с новым порядком

    #df_merged.drop(columns=["Time_floor"], inplace=True) #этот столбец больше не нужен

    df_magfield = pd.read_csv(filename_magfield, comment="#", usecols=["TIME_yyyy-mm-ddThh:mm:ss.sssZ", "MAGNITUDE_nT"])  # dataframe с магнитным полем, измеренным на RBSP
    df_magfield.rename(columns={"TIME_yyyy-mm-ddThh:mm:ss.sssZ": "Time_magfield", "MAGNITUDE_nT": "B_nT"}, inplace=True)
    df_magfield["Time_magfield"] = pd.to_datetime(df_magfield["Time_magfield"]).dt.tz_convert(None)
    df_magfield = df_magfield.sort_values("Time_magfield")

    df_merged_2 = pd.merge_asof(                    # DataFrame, который содержит всю исходную информацию RBSP, информацию о SW, и магнитное поле B измеренное на RBSP
        df_merged.sort_values("Date/time_VAP"),  # Должно быть отсортировано
        df_magfield,
        left_on="Date/time_VAP",
        right_on="Time_magfield",
        direction="nearest"  # Берёт ближайшее время
    )

    idx = df_merged_2.columns.get_loc("Time_SW") + 1    #меняем порядок отображения столбцов (просто для удобства проверки)
    cols = list(df_merged_2.columns)  # Получаем текущий порядок столбцов
    cols.remove("Time_magfield")  # Удаляем столбцы, которые нужно переместить
    cols.remove("B_nT")
    cols[idx:idx] = ["Time_magfield", "B_nT"]  # Вставляем их после "Time_SW"
    df_merged_2 = df_merged_2[cols]

    df_gsm_coordinates = pd.read_csv(filename_gsm_coordinates, comment="#")
    df_gsm_coordinates.rename(columns={"EPOCH_yyyy-mm-ddThh:mm:ss.sssZ": "Time_gsm_coordinates", "XGSM_EarthRadii": "X_GSM", "YGSM_EarthRadii":"Y_GSM", "ZGSM_EarthRadii": "Z_GSM"}, inplace=True)
    df_gsm_coordinates["Time_gsm_coordinates"] = pd.to_datetime(df_gsm_coordinates["Time_gsm_coordinates"]).dt.tz_convert(None)
    df_gsm_coordinates = df_gsm_coordinates.sort_values("Time_gsm_coordinates")

    df_final = pd.merge_asof(
        # DataFrame, который содержит всю исходную информацию RBSP, информацию о SW, и магнитное поле B измеренное на RBSP
        df_merged_2.sort_values("Date/time_VAP"),  # Должно быть отсортировано
        df_gsm_coordinates,
        left_on="Date/time_VAP",
        right_on="Time_gsm_coordinates",
        direction="nearest"  # Берёт ближайшее время
    )

    idx_B_nT = df_final.columns.get_loc("B_nT") + 1  # меняем порядок отображения столбцов (просто для удобства проверки)
    cols = list(df_final.columns)  # Получаем текущий порядок столбцов
    cols.remove("Time_gsm_coordinates")  # Удаляем столбцы, которые нужно переместить
    cols.remove("X_GSM")
    cols.remove("Y_GSM")
    cols.remove("Z_GSM")
    cols[idx_B_nT:idx_B_nT] = ["Time_gsm_coordinates", "X_GSM", "Y_GSM", "Z_GSM"]  # Вставляем их после "Time_SW"
    df_final = df_final[cols]

    return df_final


def read_sw_data(filename_SW):
    """
    Функция считывает данные о солнечном ветре из файла с swx.sinp.msu.ru
    :param filename_SW: расположение файла
    :return: dataframe с параметрами исходного файла, Dst распространен на ближайшие строки (т.к. он часовой)
    """
    with open(filename_SW, 'r') as file_SW:
        lines = file_SW.readlines()        #считываем все строки

    column_names_SW = []
    for line in lines:
        if "--" in line: #строки заголовка
            parts = line.split("--") # разбиваем на номер - название
            column_name = parts[1].strip().strip(";") # название столбца по заголовку, strip() и strip(';') удаля.т лишние пробелы и ';' из названия
            column_names_SW.append(column_name)

    data_start_index = len(column_names_SW) + 1  # индекс начала данных, "+1" нужно для пропуска строки "1 2 3 ..."

    df_SW = pd.read_csv(filename_SW, skiprows=data_start_index, header=None)
    df_SW.columns = column_names_SW
    df_SW["Time_SW"] = pd.to_datetime(df_SW["Date/time"])
    df_SW.drop(columns=["Date/time"], inplace=True)
    if df_SW["Time_SW"].dt.tz is not None: # убираем локализацию
        df_SW["Time_SW"] = df_SW["Time_SW"].dt.tz_convert(None)

    if "Dst [nT]" in df_SW.columns:
        df_SW["Dst [nT]"] = pd.to_numeric(df_SW["Dst [nT]"], errors='coerce')  # заполняем 'N/A' числовыми значениями NaN
        df_SW["Dst [nT]"] = df_SW["Dst [nT]"].ffill()  # заполняем NaN повторяющимися значениями

    df_SW["UT"] = round(df_SW["Time_SW"].dt.hour + df_SW["Time_SW"].dt.minute / 60 + df_SW["Time_SW"].dt.second / 3600 , 3)
    df_SW["Year"] = df_SW["Time_SW"].dt.year
    df_SW["Month"] = df_SW["Time_SW"].dt.month
    df_SW["Day"] = df_SW["Time_SW"].dt.day

    return df_SW


def read_van_allen_SW_data_2files(filename_van_allen, filename_SW, filename_magfield):
    """
    Считывает CSV-файлы для Van Allen Probes с данными о потоках электронов и магнитном поле; и для параметров солнечного ветра SW, объединяет их в один dataframe.
    Для столбцов, имена которых содержат информацию о потоке частиц вида:
        FEDU_Flux_(@_8.182E+00_degrees)_(@_3.300E+01_keV)__cm^(-2)s^(-1)sr^(-1)keV^(-1)
    функция извлекает значение питч-угла и энергии.
    Остальные столбцы попадают в мультииндекс как (None, original_name).
    :param filename_van_allen: расположение .csv с данными о потоках электронов в зависимости от энергии и питч-углов. источник данных: https://cdaweb.gsfc.nasa.gov/
    :param filename_SW: расположение .csv с данными о солнечном ветре (для входных параметров модели магнитосферы). источник данных: https://swx.sinp.msu.ru/
    :param filename_magfield: расположение .csv с данными о магнитном поле и GSM коорданитах спутника. источник данных: https://cdaweb.gsfc.nasa.gov/
    :return: Dataframe с объединенной информацией о солнечном ветре и данных со спутника. Мультииндекс столбцов, где уровень "Energy" и "PitchAngle"
    определяют параметры потоков (если применимо)
    """

    df_van_allen = pd.read_csv(filename_van_allen, comment="#")
    # регулярное выражение для поиска значений питч-угла и энергии.
    # предполагается, что шаблон: @_pitchValue_degrees)_(@_energyValue_keV
    pattern = r'@_([\d.E+-]+)_degrees\)_\(@_([\d.E+-]+)_keV'

    new_columns = []
    for col in df_van_allen.columns:
        m = re.search(pattern, col)
        if m:
            # m.group(1) - питч, m.group(2) - энергия.
            pitch = float(m.group(1))
            energy = float(m.group(2))
            new_columns.append(("Energy_"+str(energy), "PitchAngle_"+str(pitch)))
        else:
            if col == "EPOCH____yyyy-mm-ddThh:mm:ss.sssZ":
                new_columns.append("time_vap_tmp")#вспомогательное навание
            elif col == "L____":
                new_columns.append("L")
            else:
                new_columns.append(col)

    df_van_allen.columns = new_columns
    #Удаляем те столбцы, которые содержат энергию "-1e+31"
    cols_to_drop = [col for col in df_van_allen.columns if col[0].startswith("Energy_") and float(col[0].split("_")[1]) == -1e+31]
    df_van_allen.drop(columns=cols_to_drop,inplace=True)

    df_van_allen["Date/time_VAP"] = pd.to_datetime(df_van_allen["time_vap_tmp"])
    df_van_allen["Date/time_VAP"] = df_van_allen["Date/time_VAP"].dt.tz_convert(None)
    df_van_allen.drop(columns=["time_vap_tmp"],inplace=True)
    df_van_allen["Time_floor"] = df_van_allen["Date/time_VAP"].dt.round('min') # вспомогательный столбец с округленным до ближайшей минуты временем, чтобы состыковаться с OMNI
    df_van_allen["Time_floor"] = pd.to_datetime(df_van_allen["Time_floor"]).dt.tz_localize(None) #Для успешного объединения убираем локализацию UTC
    df_van_allen["Time_floor"] = df_van_allen["Time_floor"].dt.strftime("%Y-%m-%d %H:%M:%S") #приводим к формату yyyy-mm-dd hh:mm:ss
    df_van_allen["UT"] = round(df_van_allen["Date/time_VAP"].dt.hour + df_van_allen["Date/time_VAP"].dt.minute / 60 + df_van_allen["Date/time_VAP"].dt.second / 3600 , 3)
    df_van_allen["Year"] = df_van_allen["Date/time_VAP"].dt.year
    df_van_allen["Month"] = df_van_allen["Date/time_VAP"].dt.month
    df_van_allen["Day"] = df_van_allen["Date/time_VAP"].dt.day

    #Считываем данные о SW
    with open(filename_SW, 'r') as file_SW:
        lines = file_SW.readlines()        #считываем все строки

    column_names_SW = []
    for line in lines:
        if "--" in line: #строки заголовка
            parts = line.split("--") # разбиваем на номер - название
            column_name = parts[1].strip().strip(";") # название столбца по заголовку, strip() и strip(';') удаля.т лишние пробелы и ';' из названия
            column_names_SW.append(column_name)

    data_start_index = len(column_names_SW) + 1  # индекс начала данных, "+1" нужно для пропуска строки "1 2 3 ..."

    df_SW = pd.read_csv(filename_SW, skiprows=data_start_index, header=None)
    df_SW.columns = column_names_SW
    df_SW["Time_SW"] = pd.to_datetime(df_SW["Date/time"])
    df_SW.drop(columns=["Date/time"], inplace=True)
    if df_SW["Time_SW"].dt.tz is not None: # убираем локализацию
        df_SW["Time_SW"] = df_SW["Time_SW"].dt.tz_convert(None)

    if "Dst [nT]" in df_SW.columns:
        df_SW["Dst [nT]"] = pd.to_numeric(df_SW["Dst [nT]"], errors='coerce')  # заполняем 'N/A' числовыми значениями NaN
        df_SW["Dst [nT]"] = df_SW["Dst [nT]"].ffill()  # заполняем NaN повторяющимися значениями
    df_SW["Time_SW"] = df_SW["Time_SW"].dt.strftime("%Y-%m-%d %H:%M:%S")  # приводим к формату yyyy-mm-dd hh:mm:ss

    df_merged = df_van_allen.merge(df_SW, left_on="Time_floor", right_on="Time_SW", how="left") #Объединенный dataframe (и OMNI и VAP)
    columns_order = (["Date/time_VAP", "UT", "Year", "Month", "Day"] + list(df_SW.columns[:]) +
                     [col for col in df_van_allen.columns if  col not in ["Date/time_VAP", "UT", "Year", "Month", "Day", "Time_floor", "X_GSM", "Y_GSM", "Z_GSM"]]) #немного иной порядок столбцов, просто для удобства проверки
    df_merged = df_merged[columns_order] #перестановка стобцов в соответствии с новым порядком

    df_magfield = pd.read_csv(filename_magfield, comment="#", usecols=["TIME_yyyy-mm-ddThh:mm:ss.sssZ", "MAGNITUDE_nT", "X_GSM_km", "Y_GSM_km", "Z_GSM_km"])  # dataframe с магнитным полем, измеренным на RBSP
    df_magfield.rename(columns={"TIME_yyyy-mm-ddThh:mm:ss.sssZ": "Time_magfield", "MAGNITUDE_nT": "B_nT", "X_GSM_km": "X_GSM", "Y_GSM_km":"Y_GSM", "Z_GSM_km":"Z_GSM"}, inplace=True)
    df_magfield["Time_magfield"] = pd.to_datetime(df_magfield["Time_magfield"]).dt.tz_convert(None)
    df_magfield = df_magfield.sort_values("Time_magfield")

    cols_to_scale = ['X_GSM', 'Y_GSM', 'Z_GSM'] #из км переводим в RE
    df_magfield[cols_to_scale] = df_magfield[cols_to_scale] / 6371


    df_merged_2 = pd.merge_asof(                 # DataFrame, который содержит всю исходную информацию VAP, информацию о SW, и магнитное поле B измеренное на VAP
        df_merged.sort_values("Date/time_VAP"),  # Должно быть отсортировано
        df_magfield,
        left_on="Date/time_VAP",
        right_on="Time_magfield",
        direction="nearest"  # Берёт ближайшее время
    )

    idx = df_merged_2.columns.get_loc("Time_SW") + 1    #меняем порядок отображения столбцов (для удобства проверки)
    cols = list(df_merged_2.columns)  # Получаем текущий порядок столбцов
    cols.remove("Time_magfield")  # Удаляем столбцы, которые нужно переместить
    cols.remove("B_nT")
    cols.remove("X_GSM")
    cols.remove("Y_GSM")
    cols.remove("Z_GSM")
    cols[idx:idx] = ["Time_magfield", "B_nT", "X_GSM", "Y_GSM", "Z_GSM"]  # Вставляем их после "Time_SW"
    df_merged_2 = df_merged_2[cols]

    return df_merged_2


def find_passes(df, min_L=2.5, branch='both', L_threshold=0.01, time_gap_threshold=3600):
    """
    Выделяет пересечения внешнего радиационного пояса с разделением на восходящие/нисходящие ветви.
    :param df: исходный DataFrame
    :param min_L: граничное значение L, с которого начинается внешний пояс (2.5 Re)
    :param branch: направление витков: 'up' - восходящие витки (к апогею), 'down' - нисходящие витки (к перигею), 'both' - оба вида одновременно
    :param L_threshold: допуск для определения границ
    :param time_gap_threshold:
    :return: Отфильтрованные данные с колонками 'pass_id' - номер витка и 'branch' - вид витка
    """
    """
    Выделяет пересечения внешнего радиационного пояса с разделением на восходящие/нисходящие ветви.
    Строго отбрасывает неполные пересечения, не начинающиеся/не заканчивающиеся на min_L.

    Параметры:
    df (pd.DataFrame): Исходный DataFrame
    min_L (float): Граничное значение L (2.5 Re)
    branch (str): 'up', 'down' или 'both'
    L_threshold (float): Допуск для определения границ (0.01 Re)

    Возвращает:
    pd.DataFrame: Отфильтрованные данные с колонками 'pass_id' и 'branch'
    """
    df = df.copy()
    df.sort_values('Date/time_VAP', inplace=True)
    df.reset_index(drop=True, inplace=True)

    #идентификация всех точки в поясе (L >= min_L)
    in_belt = df[df['L'] >= min_L - L_threshold].copy()

    #производная dL/dt для определения направления
    in_belt['dL'] = in_belt['L'].diff()
    in_belt['dt'] = in_belt['Date/time_VAP'].diff().dt.total_seconds()
    in_belt['dL_dt'] = in_belt['dL'] / in_belt['dt']

    in_belt['direction'] = 'unknown'
    in_belt.loc[in_belt['dL_dt'] > 0, 'direction'] = 'up'
    in_belt.loc[in_belt['dL_dt'] < 0, 'direction'] = 'down'

    #смена направления
    direction_changes = (in_belt['direction'] != in_belt['direction'].shift(1))
    time_gaps = in_belt['dt'] > time_gap_threshold
    in_belt['segment_id'] = (direction_changes | time_gaps).cumsum()

    results = []
    current_pass_id = 0

    for seg_id, segment in in_belt.groupby('segment_id'):
        if len(segment) < 2:
            continue

        direction = segment['direction'].iloc[0]
        start_L = segment['L'].iloc[0]
        end_L = segment['L'].iloc[-1]

        is_complete = False
        if direction == 'up':
            #восходящий сегмент должен начинаться ~min_L и заканчиваться максимумом
            is_complete = start_L >= min_L
        elif direction == 'down':
            #нисходящий сегмент должен начинаться максимумом и заканчиваться ~min_L
            is_complete = end_L >= min_L

        if not is_complete:
            continue

        current_pass_id += 1

        segment = segment.copy()
        segment['pass_id'] = current_pass_id
        segment['branch'] = direction

        if branch == 'both' or branch == direction:
            results.append(segment)

    if not results:
        return pd.DataFrame(columns=df.columns.tolist() + ['pass_id', 'branch'])

    result_df = pd.concat(results)

    result_df.drop(columns=['dL', 'dt', 'dL_dt', 'direction', 'segment_id'], inplace=True, errors='ignore')

    return result_df.sort_values('Date/time_VAP').reset_index(drop=True)


def kinetic_energy(mu, B, alpha_K):
    """
    Функция расчета кинетической энергии электрона
    :param mu: магнитный момент, для которого рассчитывается энергия, задается в main(), МэВ/Гс
    :param B: магнитное поле в точке спутника, задается из файла с данными, нТл
    :param alpha_K: питч-угол, соответствующий заданному фиксированному K, задается из интерполяции значений K(alpha), градусы
    :return: значение кинетической энергии E(mu,K), кэВ
    """
    m_0 = 0.511 # энергия покоя электрона, МэВ/c^2

    mu = np.array(mu, ndmin=1)
    B = np.array(B*10**(-5), ndmin=1) # в Гс
    alpha_K = np.array(alpha_K, ndmin=1)

    sin_alpha = np.sin(np.radians(alpha_K))

    # p^2 = (2 * m0 * B * mu) / sin^2(alphaK)
    p_squared = (2.0 * m_0 * B * mu) / (sin_alpha ** 2) # (МэВ/с)^2

    p_squared[p_squared < 0] = np.nan

    #полная релятивистская энергия E_tot = sqrt( (p*c)^2 + (m0*c^2)^2 ), МэВ
    E_tot = np.sqrt(p_squared + m_0 ** 2)

    #кинетическая энергия E_kin = E_tot - m0*c^2, кэВ
    E_kin = (E_tot - m_0)*(10**3)

    return E_kin


def fit_j_alpha(j_values, alpha_values, initial_guess = 0, energy=None):
    """
    Аппроксимации питч-углового распределения j(alpha) функцией j(alpha) = C0 * sin(alpha) -C1 * (sin(alpha))^(C2). Необходима для того, чтобы найти значение, соответствующее
    alpha_K - питч-углу для фиксированного K
    :param j_values: array, потоки j для разных значений угла. энергия при этом фиксирована, т.е. задается j(alpha,E)
    :param alpha_values: array, углы, градусы
    :param initial_guess: флаг проводимой аппроксимации. 0 - для всех энергий начальные параметры одинаковы [max(j_values)/2,max(j_values)/2,1.5],
    1 - происходит перебор C0,C1,C2 для поиска минимальной ошибки, полученный результат подается в качестве начальных параметров
    :return: функция j(alpha), alpha в градусах в результате аппроксимации j(alpha) = C0 * sin(alpha) -C1 * (sin(alpha))^(C2)
    """
    j_values = np.array(j_values, dtype=float, copy=True)
    alpha_values = np.array(alpha_values, dtype=float, copy=True)

    #отбрасываем -1e+31, которое иногда присутствует в данных и фильтруем alpha [0;180]
    mask = (j_values >= 0) & (np.sin(np.radians(alpha_values)) >= 0)
    j_values = j_values[mask]
    alpha_values = alpha_values[mask]

    if j_values.size == 0:
        return None

    angles_rad = np.radians(alpha_values)

    def model(alpha, params):
        #здесь alpha в радианах
        (C0, C1, C2) = params
        return C0 * np.sin(alpha) - C1 * np.abs((np.sin(alpha))) ** C2

    def objective(params):
        #невязка для C0,C1,C2
        pred = model(angles_rad, params)
        return np.sum((pred - j_values) ** 2)

    if initial_guess == 1:
        c0_range = np.linspace(-max(j_values), max(j_values), 10)
        c1_range = np.linspace(-max(j_values)/3, max(j_values)/3, 10)
        c2_range = np.linspace(0.5, 10, 10)
        best_grid_score = np.inf
        best_initial_params = (None, None, None)
        for C0 in c0_range:
            for C1 in c1_range:
                for C2 in c2_range:
                    err = objective((C0, C1, C2))
                    if err < best_grid_score:
                        best_grid_score = err
                        best_initial_params = (C0, C1, C2)
        init_params = list(best_initial_params)

    init_params1 = [max(j_values)/2,max(j_values)/2,1.5]
    if initial_guess == 1:
        res = minimize(objective, init_params, method='Nelder-Mead')
    elif initial_guess == 0:
        res = minimize(objective, init_params1, method='Nelder-Mead')


    best_params = res.x

    def j_alpha_func(alpha):
        #alpha в градусах
        return model(alpha * np.pi /180, best_params)

    j_alpha_func.best_params = best_params #прикрепляем к функции полученные коэффициенты аппроксимации, чтобы при случае на них посмотреть
    if initial_guess ==1:
        j_alpha_func.initial_params = init_params
    elif initial_guess ==0:
        j_alpha_func.initial_params = init_params1

    return j_alpha_func


def compute_p_squared(E):
    """
    Функция рассчитывает импульс частицы по заданной кинетической энергии
    :param E: конерктное значение кинетической энергии (не массив), кэВ
    :return: импульс в квадрате (одно значение, не массив), (кэВ/c)^2
    """
    m_e = 511 # масса электрона, кэВ/c^2
    return E**2 + 2 * m_e * E


def compute_psd(E_array, j_array):
    """
    Функция вычисляет плотность электронов в фазовом пространстве f = j / p^2
    :param E_array: массив энергий, кэВ
    :param j_array: массив соответствующих потоков, см^(-2)с^(-1)ср^(-1)кэВ^(-1)
    :return: массив PSD f(E) = j/p^2
    """
    E_array = np.array(E_array, dtype=float, copy=True)
    j_array = np.array(j_array, dtype=float, copy=True)

    p_squared_array = [compute_p_squared(E) for E in E_array]
    p_squared_array = np.array(p_squared_array)

    f_array = j_array / p_squared_array
    return f_array


def fit_f_E(x_data, y_data):
    """
    Аппроксимация данных (x_data, y_data) экспоненциальной функцией:
        y(x) = a * exp(b*x)
    Функция, создающая "интерполятор" для f(E)
    :param x_data: массив x, (предполагается энергия, МэВ)
    :param y_data: массив y, (предполагается PSD f(E) )
    :return:
    fit_func(x) : функция (замыкание),
        при вызове fit_func(x) возвращает a*exp(b*x) c найденными параметрами
    fit_func.best_params : кортеж (a, b),
        оптимальные параметры аппроксимации
    fit_func.covar : 2D-массив,
        ковариационная матрица аппроксимации (из curve_fit)
    """
    x_data = np.array(x_data, dtype=float, copy=True)
    y_data = np.array(y_data, dtype=float, copy=True)
    #print(x_data, y_data)
    x_b = 1000 #граница кусочной аппроксимации

    # Модельная функция: y(x) = a * x^(-b) * exp(-x/E0)
    def model_right(x, a, b):
        return a * np.exp(-x/b)

    def model2(x, a, b):
        return a * x**(-b)

    mask_left = (x_data < x_b)
    mask_right = (x_data >= x_b)
    x_left, y_left = x_data[mask_left], y_data[mask_left]
    x_right, y_right = x_data[mask_right], y_data[mask_right]

    ########################
    # Левый участок
    ########################
    log_x_left = np.log(x_left)
    log_y_left = np.log(y_left)

    p_left = np.polyfit(log_x_left, log_y_left, deg=1)

    slope_left = p_left[0]
    intercept_left = p_left[1]
    b_left = -slope_left
    a_left = np.exp(intercept_left)

    ############### Правый участок
    # Начальные оценки параметров (p0): a, b, E0
    #p0_left = (y_left[0], 1.0)
    p0_right = (y_right[0], np.mean(x_right))
    weights = 1 / y_left
    #Фит правой и левой части
    #popt_left, pcov_left = curve_fit(model2, x_left, y_left,sigma=weights, absolute_sigma=True, p0=p0_left)
    popt_right, pcov_right = curve_fit(model_right, x_right, y_right, p0=p0_right)

    # функция-замыкание для удобства
    def fit_func(x):
        # работает, если x скаляр или массив
        x = np.array(x, ndmin=1)  # векторизуем
        left_vals = a_left * x**(-b_left)
        right_vals = model_right(x, *popt_right)
        # Выбираем что слева/справа
        vals = np.where(x < x_b, left_vals, right_vals)
        return vals if vals.size>1 else vals[0]

    fit_func.x_b = x_b
    fit_func.best_params_left = (a_left, b_left)
    fit_func.best_params_right = popt_right
    #fit_func.cov_left = pcov_left
    fit_func.cov_right = pcov_right

    return fit_func


def fit_f_E_1(x_data, y_data):
    """
    Аппроксимация данных (x_data, y_data) степенной функцией:
        y(x) = a * x^b
    :param x_data: массив x, (предполагается энергия, кэВ)
    :param y_data: массив y, (предполагается PSD f(E) )
    :return:
    fit_func(x) : функция (замыкание),
        при вызове fit_func(x) возвращает a*exp(b*x) c найденными параметрами
    fit_func.best_params : кортеж (a, b),
        оптимальные параметры аппроксимации
    """
    x_data = np.array(x_data, dtype=float, copy=True)
    y_data = np.array(y_data, dtype=float, copy=True)

    mask = y_data > 0

    x_data = x_data[mask]
    y_data = y_data[mask]

    if x_data.size == 0 or y_data.size == 0:
        return None

    log_x = np.log(x_data)
    log_y = np.log(y_data)

    p_left = np.polyfit(log_x, log_y, deg=1)

    slope_left = p_left[0]
    intercept_left = p_left[1]
    b_left = -slope_left
    a_left = np.exp(intercept_left)

    def fit_func(x):
        x = np.array(x, ndmin=1)  # векторизуем
        vals = a_left * x**(-b_left)
        return vals if vals.size>1 else vals[0]

    fit_func.best_params_left = (a_left, b_left)

    return fit_func

def get_phi(point):
    """
    Из вектора в декартовых координатах [x,y,z] возвращется угол phi в сферических
    :param point: вектор [x,y,z]
    :return: phi, градусы
    """
    x0, y0, z0 = point[0], point[1], point[2]
    return math.degrees(math.atan2(y0,x0))


def get_theta(point):
    """
    Из вектора в декартовых координатах [x,y,z] возвращется угол theta в сферических
    :param point: вектор [x,y,z]
    :return: theta, градусы
    """
    x0, y0, z0 = point[0], point[1], point[2]
    r0 = np.sqrt(x0 ** 2 + y0 ** 2 + z0 ** 2)
    cos_theta = z0 / r0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return math.degrees(math.acos(cos_theta))


def set_theta(point, theta):
    """
    Функция устанавливает конкретное значение угла тетта при сохранении остальных сферических координат
    :param point: вектор [x0,y0,z0]
    :param theta: нужный угол, радианы
    :return: преобразованный вектор [x,y,z]
    """
    x0, y0, z0 = point[0], point[1], point[2]
    r0 = np.sqrt(x0 ** 2 + y0 ** 2 + z0 ** 2)
    phi0 = math.atan2(y0, x0)
    x1 = r0 * math.sin(theta) * math.cos(phi0)
    y1 = r0 * math.sin(theta) * math.sin(phi0)
    z1 = r0 * math.cos(theta)
    return (x1, y1, z1)


def set_phi(point, phi):
    """
        Функция устанавливает конкретное значение угла phi при сохранении остальных сферических координат
        :param point: вектор [x0,y0,z0]
        :param phi: нужный угол, радианы
        :return:преобразованный вектор [x,y,z]
    """
    x0, y0, z0 = point[0], point[1], point[2]
    r0 = np.sqrt(x0 ** 2 + y0 ** 2 + z0 ** 2)
    theta0 = math.acos(z0 / r0)
    x1 = r0 * math.sin(theta0) * math.cos(phi)
    y1 = r0 * math.sin(theta0) * math.sin(phi)
    z1 = r0 * math.cos(theta0)
    return (x1, y1, z1)


def set_r(point, r):
    """
        Функция устанавливает конкретное значение радиуса R при сохранении остальных сферических координат
        :param point: вектор [x0,y0,z0]
        :param r: нужный радиус, RE
        :return:
    """
    x0, y0, z0 = point[0], point[1], point[2]
    r0 = np.sqrt(x0 ** 2 + y0 ** 2 + z0 ** 2)
    theta0 = math.acos(z0 / r0)
    phi0 = math.atan2(y0,x0)
    x1 = r * math.sin(theta0) * math.cos(phi0)
    y1 = r * math.sin(theta0) * math.sin(phi0)
    z1 = r * math.cos(theta0)
    return (x1, y1, z1)


def calculate_K(theta, phi, start_point_sm, b_mirror_target, solar_params):
    """
    Обертка для вычисления K(theta, phi)
    :param theta: широтный угол theta, градусы
    :param phi: азимутальный угол phi, градусы
    :param start_point_sm: [x, y, z], RE SM. Исходная точка, относительно которой происходит поворот. Предполагается,
                            что это точка на исходной м. линии на r~1.2
    :param b_mirror_target: нТл. Зеркальное магнитное поле для исходной магнитной линии, т.е. то поле, которое мы считаем
                            постоянным вдоль всего дрейфа
    :param solar_params: параметры солнечного ветра для модели : (X1,X2,X3,X4,X5,X6,X7,ut,iy,mo,id,ro,v,bimf,dst,al,sl,h)
    :return: Словарь, аргументы которого:
            - "theta": широтный угол theta (исходный), градусы
            - "phi": азимутальный угол phi (исходный), градусы
            - "flag_open_fieldline": флаг открытой силовой линии, 1 - линия открыта, 0 - нет
            - "error": сообщение об ошибке, если она возникла: "Bmir_lt_Bmin" или "Line is open"
            - "K": значение интеграла K, Гс^(-1/2)*R_E
            - "L": значение L-координаты (точки на экваторе), R_E
            - "theta_foot": значение широтного угла у подножья магнитной линии в северном полушарии для SM координат, градусы
            - "B(3)_SM" (list[list[float]]): компоненты магнитного поля в точках x_shell, нТл, SM
            - "x_shell" (list[list[float]]): координаты оболочки между зеркальными линиями, R_E, SM
            - "B_shell" (list[float]): модуль магнитного поля в точках x_shell, нТл
            - "x_shell_full (list[list[float]])": координаты полных магнитных линий оболочки, R_E, SM
            - "B_shell_full" (list[float]): модуль магнитного поля в точках x_shell_full, нТл
            - "start_point" (list[float]): вектор исходной точки, SM
    """
    X1, X2, X3, X4, X5, X6, X7, ut, iy, mo, id, ro, v, bimf, dst, al, sl, h = solar_params

    result = {
        "theta": theta,
        "phi": phi,
        "flag_open_fieldline": 0,
        "error": None,
        "K": None,
        "L": None,
        "theta_foot": None,
        "B(3)_SM": None,
        "x_shell": None,
        "B_shell": None,
        "x_shell_full": None,
        "B_shell_full": None,
        "start_point": None
    }

    try:
        new_start_point_sm = set_phi(start_point_sm, np.radians(phi)) #SM
        new_start_point_sm = set_theta(new_start_point_sm, np.radians(theta))
        result["start_point"] = new_start_point_sm

        new_start_point_gsm = a2000_library.sm2gsm(new_start_point_sm, ut, id, mo, iy)
        data = a2000_library.trace_line(X1, X2, X3, X4, X5, X6, X7, ut, iy, mo, id, ro, v, bimf, dst, al, new_start_point_gsm, sl, h)
        x_line_gsm = data[:, 0:3]
        x_line_sm = np.apply_along_axis(lambda x: a2000_library.gsm2sm(x, ut, id, mo, iy), 1, arr=x_line_gsm)

        theta_foot = get_theta(x_line_sm[-1]) #theta у подножья линии на севере
        result["theta_foot"] = theta_foot

        b_line = data[:, 3]
        b_gsm = data[:, 4:7]
        b_sm = np.apply_along_axis(lambda b: a2000_library.gsm2sm(b, ut, id, mo, iy), 1, arr=b_gsm)
        mirror_north, mirror_south = mirror_points_b_mirr(x_line_sm, b_line, b_mirror_target)
        if mirror_north is None or mirror_south is None:
            #result["flag_open_fieldline"] = 1
            if b_mirror_target < min(b_line):
                result["error"] = "Bmir_lt_Bmin"
            else:
                result["error"] = "Line is open"
                result["flag_open_fieldline"] = 1
            return result

        result["K"] = curve_integral(x_line_sm, b_line, mirror_north, mirror_south, b_mirror_target)
        result["L"] = np.linalg.norm(x_line_sm[np.argmin(b_line)])

        result["x_shell_full"] = x_line_sm
        result["B_shell_full"] = b_line

        #result["x_equator"] = x_line_sm[np.argmin(b_line)]
        #result["B_equator"] = min(b_line)
        #result["B(3)_SM"] = b_sm

        start_idx = next(i for i, x in enumerate(x_line_sm) if np.array_equal(x, mirror_south))
        end_idx = next(i for i, x in enumerate(x_line_sm) if np.array_equal(x, mirror_north))
        result["x_shell"] = x_line_sm[start_idx:end_idx + 1]
        result["B_shell"] = b_line[start_idx:end_idx + 1]
        result["B(3)_SM"] = b_sm[start_idx:end_idx + 1]

    except Exception as e:
        result["error"] = str(e)
        print(f"Error for theta={theta}, phi={phi}: {str(e)}")

    return result


def bisection_wrapper(phi, initial_theta, b_mirror_target, K_target, start_point_sm, solar_params, tol=0.1, max_iter=15):
    """
    Реализация метода бисекции для поиска решения уравнения K(theta) - K_target = 0
    :param phi:
    :param initial_theta: начальное приближение угла theta, начиная от которого ищутся углы theta для K(theta)>K_target и K(theta)<K_target
    :param b_mirror_target: заданное значение зеркального магнитного поля (посчитанное на исходной линии дрейфа), нТл
    :param K_target: заданное изначально значение K, Гс^1/2*км
    :param start_point_sm: исходная пространственная точка расчета K
    :param solar_params: входные параметры для модели поля
    :param tol: точность K(theta)
    :param max_iter: максимальное количество итераций метода бисекции
    :return: результат выполнения calculate_K (т.е. словарь с необходимой магнитной линией и т.п.) для найденного theta, при котором |K(theta)-K_target| < tol
    """

    min_diff = float('inf')
    theta_min = initial_theta
    theta_max = initial_theta
    max_expand_steps = 50
    step = 0.5

    expand_steps = 0
    while expand_steps < max_expand_steps:
        k_result = calculate_K(theta_min, phi, start_point_sm, b_mirror_target, solar_params)
        if k_result["K"] is None and k_result["error"] == "Bmir_lt_Bmin":
            theta_min += 0.5
            expand_steps += 1
            continue
        if k_result["K"] is None and k_result["error"] == "Line is open":
            theta_min -= 0.5
            expand_steps += 1
            continue

        if k_result.get("K") is not None and k_result["K"] <= K_target:
            break
        theta_min -= step
        expand_steps += 1

    step = 0.5
    expand_steps = 0

    while expand_steps < max_expand_steps:
        k_result = calculate_K(theta_max, phi, start_point_sm, b_mirror_target, solar_params)
        if k_result["K"] is None and k_result["error"] == "Bmir_lt_Bmin":
            theta_max += 0.5
            expand_steps += 1
            continue
        if k_result["K"] is None and k_result["error"] == "Line is open":
            theta_max -= 0.5
            expand_steps += 1
            continue
        if k_result.get("K") is None and k_result["error"] is None:
            theta_max = theta_min
            continue
        if k_result.get("K") is not None and k_result["K"] >= K_target:
            break

        theta_max += step
        expand_steps += 1

    k_min = calculate_K(theta_min, phi, start_point_sm, b_mirror_target, solar_params)
    k_max = calculate_K(theta_max, phi, start_point_sm, b_mirror_target, solar_params)

    if k_min.get("K") is None:
        print(f"phi={phi},theta_min={theta_min}, k_min не посчиталось")
        k_min["flag_open_fieldline"] = 1
        return k_min
    if k_max.get("K") is None:
        print(f"phi={phi},theta_max={theta_max}, k_max не посчиталось")
        k_min["flag_open_fieldline"] = 1
        return k_min
#
    if (k_min["K"] - K_target) * (k_max["K"] - K_target) > 0:
        print(f"phi={phi}, theta_min={theta_min}, theta_max={theta_max}\tk_min и k_max по одну сторону от K_target")
        k_min["flag_open_fieldline"] = 1
        return k_min

    for _ in range(max_iter):
        theta_mid = (theta_min + theta_max) / 2
        result = calculate_K(theta_mid, phi, start_point_sm, b_mirror_target, solar_params)

        if result["flag_open_fieldline"] == 1:
            return result
        if result["K"] is None:
            result["flag_open_fieldline"] = 1
            return result

        current_diff = abs(result["K"]-K_target)
        if current_diff < min_diff:
            min_diff = current_diff
            best_result = result
        if current_diff < tol:
            break

        k_min = calculate_K(theta_min, phi, start_point_sm, b_mirror_target, solar_params)
        if k_min.get("K") is None or result.get("K") is None:
            result["flag_open_fieldline"] = 1
            return result

        if (k_min["K"] - K_target) * (result["K"] - K_target) < 0:
            theta_max = theta_mid
        else:
            theta_min = theta_mid

    if abs(best_result["K"]-K_target) > tol:
        best_result["error"] = "K is calculated with small accuracy"

    return best_result


def trace_drift(x_sat, solar_params, K_target):
    """
    Трассировка дрейфа электрона вокруг Земли - получение координат магнитных линий оболочки, расчет магнитного потока и L_star. Используются параллельные вычисления
    :param x_sat: вектор [x,y,z] спутника, RE
    :param solar_params: входные параметры для модели магнитного поля
    :param K_target: заданное значение второго инварианта K
    :return: значение L_star, RE
    """
    (X1, X2, X3, X4, X5, X6, X7, ut, iy, mo, id, ro, v, bimf, dst, al, sl, h) = solar_params
    file_candidates = open(f"calculated magnetic field for drift trajectory\\{iy}_{mo}_{id}_{ut}.txt", "w")
    file_candidates_full_trajectory = open(f"calculated magnetic field for drift trajectory\\{iy}_{mo}_{id}_{ut}_full.txt", "w")

    start_point = [x_sat[0], x_sat[1], x_sat[2]] # Исходная точка в координатах GSM

    data_from_model = a2000_library.trace_line(X1, X2, X3, X4, X5, X6, X7, ut, iy,mo,id,ro,v,bimf,dst,al,start_point,sl,h)

    x_line_gsm = data_from_model[:, 0:3] #первые три столбца выходных модельных файлов - координаты магнитной линии в GSM
    b_line = data_from_model[:, 3] #четвертый столбец выходных модельных файлов - абсолютное магнитное поле (от СК не зависит)

    l_magfield = np.linalg.norm(x_line_gsm[np.argmin(b_line)])

    x_line_sm = np.apply_along_axis(lambda x: a2000_library.gsm2sm(x, ut, id, mo, iy), axis=1, arr=x_line_gsm) # перевод линии в SM координаты
    start_point_sm = a2000_library.gsm2sm(start_point, ut, id, mo, iy) # Стартовая точка в SM координатах

    K = []
    flag_open_fieldline = 0
    for alpha in range(5, 90, 5):
        mirror_point_north = mirror_points_pitch_angle(x_line_sm, b_line, alpha, start_point_sm, equatorial=True)[0]
        mirror_point_south = mirror_points_pitch_angle(x_line_sm, b_line, alpha, start_point_sm, equatorial=True)[1]
        b_mirror = mirror_points_pitch_angle(x_line_sm, b_line, alpha, start_point_sm, equatorial=True)[2]
        if mirror_point_south is None or mirror_point_north is None:
            flag_open_fieldline = 1
            break
        K.append(curve_integral(x_line_sm, b_line, mirror_point_north, mirror_point_south, b_mirror))

    if flag_open_fieldline == 1:
        return

    pitch_angles = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85])
    cubic_interp_inverse = interpolation(np.flip(K), np.flip(pitch_angles), method="cubic")
    pitch_angle_target = cubic_interp_inverse(K_target)  # нужный питч угол для заданного K

    b_mirror_target = mirror_points_pitch_angle(x_line_sm, b_line, pitch_angle_target, start_point_sm, equatorial=True)[2]

    print(f"Initial data:\tb_mirr_target={b_mirror_target}\tK={calculate_K(get_theta(start_point_sm),get_phi(start_point_sm),start_point_sm,b_mirror_target, solar_params)["K"]}\tL:{l_magfield}")

    # За точку, от которой будут идти шаги по азимуту, берется точка исходной магнитной линии на расстоянии r=1.2 (на самом деле чуть больше)
    # По этой причине индекс равен 0.2/h (шаги по умолчанию 0.01 или 0.001)
    initial_theta = get_theta(x_line_sm[int(0.2 / h)])
    step_phi = 6
    initial_phi = get_phi(start_point_sm) + step_phi
    final_phi = initial_phi + 360

    results = []
    new_start_point = x_line_sm[int(0.2 / h)]

    start_whole_phi = time.perf_counter()
    with ProcessPoolExecutor() as executor:
        futures = []
        for phi in np.arange(initial_phi, final_phi, step_phi):
            futures.append(executor.submit(bisection_wrapper,
                                           phi=phi,
                                           initial_theta=initial_theta,
                                           b_mirror_target=b_mirror_target,
                                           K_target=K_target,
                                           start_point_sm=new_start_point,
                                           solar_params=solar_params,
                                           tol=0.01,
                                           max_iter=50
                                           ))
        for future in futures:
            result = future.result()
            results.append(result)
    end_whole_phi = time.perf_counter()
    #print(f"parallel время на все фи: {end_whole_phi - start_whole_phi:.6f}")

    thetas_foot = [] # углы theta подножья магнитных линий
    phis = [] #список phi, который соответствует списку thetas_phi
    for result in results:
        if result["x_shell"] is not None:
            file_candidates.write(
                f"phi={result["phi"]},theta={result["theta"]}\tK:{result["K"]}\tL:{result["L"]}\n")
            for counter in range(len(result["x_shell"])):
                file_candidates.write(
                    f"{result["x_shell"][counter][0]}\t{result["x_shell"][counter][1]}\t{result["x_shell"][counter][2]}\t{result["B_shell"][counter]}\t{result["B(3)_SM"][counter][0]}\t{result["B(3)_SM"][counter][1]}\t{result["B(3)_SM"][counter][2]}\n")
            file_candidates.write("-------------------------------\n")

            file_candidates_full_trajectory.write(
                f"phi={result["phi"]},theta={result["theta"]}\tK:{result["K"]}\tL:{result["L"]}\n")
            for counter in range(0, len(result["x_shell_full"])):
                file_candidates_full_trajectory.write(
                    f"{result["x_shell_full"][counter][0]}\t{result["x_shell_full"][counter][1]}\t{result["x_shell_full"][counter][2]}\t{result["B_shell_full"][counter]}\n")
            file_candidates_full_trajectory.write("-------------------------------\n")

        if result["x_shell"] is not None:
            thetas_foot.append(result["theta_foot"])
            phis.append(result["phi"])

    model_params = (X1, X2, X3, X4, X5, X6, X7, ut, iy, mo, id, ro, v, bimf, dst, al)
    flux = calculate_flux_north(thetas_foot, phis, model_params)
    M = a2000_library.magnetic_moment(ut, id, mo, iy)
    L_star = 2 * np.pi * M / flux

    file_candidates.close()
    file_candidates_full_trajectory.close()

    return L_star


def trace_drift_without_parallel(x_sat, solar_params, K_target):
    """
    Трассировка дрейфа электрона вокруг Земли - получение координат магнитных линий оболочки, расчет магнитного потока и L_star. Не используются параллельные вычисления
    :param x_sat: вектор [x,y,z] спутника, RE
    :param solar_params: входные параметры для модели магнитного поля
    :param K_target: заданное значение второго инварианта K
    :return: значение L_star, RE
    """
    (X1, X2, X3, X4, X5, X6, X7, ut, iy, mo, id, ro, v, bimf, dst, al, sl, h) = solar_params
    #file_candidates = open(f"calculated magnetic field for drift trajectory\\{iy}_{mo}_{id}_{ut}.txt", "w")
    #file_candidates_full_trajectory = open(f"calculated magnetic field for drift trajectory\\{iy}_{mo}_{id}_{ut}_full.txt", "w")

    start_point = [x_sat[0], x_sat[1], x_sat[2]] # Исходная точка в координатах GSM

    data_from_model = a2000_library.trace_line(X1, X2, X3, X4, X5, X6, X7, ut, iy,mo,id,ro,v,bimf,dst,al,start_point,sl,h)

    x_line_gsm = data_from_model[:, 0:3] #первые три столбца выходных модельных файлов - координаты магнитной линии в GSM
    b_line = data_from_model[:, 3] #четвертый столбец выходных модельных файлов - абсолютное магнитное поле (от СК не зависит)

    l_magfield = np.linalg.norm(x_line_gsm[np.argmin(b_line)])

    x_line_sm = np.apply_along_axis(lambda x: a2000_library.gsm2sm(x, ut, id, mo, iy), axis=1, arr=x_line_gsm) # перевод линии в SM координаты

    start_point_sm = a2000_library.gsm2sm(start_point, ut, id, mo, iy) # Стартовая точка в SM координатах

    K = []
    flag_open_fieldline = 0
    for alpha in range(5, 90, 5):
        mirror_point_north = mirror_points_pitch_angle(x_line_sm, b_line, alpha, start_point_sm, equatorial=True)[0]
        mirror_point_south = mirror_points_pitch_angle(x_line_sm, b_line, alpha, start_point_sm, equatorial=True)[1]
        b_mirror = mirror_points_pitch_angle(x_line_sm, b_line, alpha, start_point_sm, equatorial=True)[2]
        if mirror_point_south is None or mirror_point_north is None:
            flag_open_fieldline = 1
            break
        K.append(curve_integral(x_line_sm, b_line, mirror_point_north, mirror_point_south, b_mirror))

    if flag_open_fieldline == 1:
        return

    pitch_angles = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85])
    cubic_interp_inverse = interpolation(np.flip(K), np.flip(pitch_angles), method="cubic")
    pitch_angle_target = cubic_interp_inverse(K_target)  # нужный питч угол для заданного K

    b_mirror_target = mirror_points_pitch_angle(x_line_sm, b_line, pitch_angle_target, start_point_sm, equatorial=True)[2]
    print(f"Initial data:\tb_mirr_target={b_mirror_target}\tK={calculate_K(get_theta(start_point_sm),get_phi(start_point_sm),start_point_sm,b_mirror_target, solar_params)["K"]}\tL:{l_magfield}")

    # За точку, от которой будут идти шаги по азимуту, берется точка исходной магнитной линии на расстоянии r=1.2 (на самом деле чуть больше)
    # По этой причине индекс равен 0.2/h (шаги по умолчанию 0.01 или 0.001)
    initial_theta = get_theta(x_line_sm[int(0.2 / h)])
    step_phi = 6
    initial_phi = get_phi(start_point_sm) + step_phi
    final_phi = initial_phi + 360

    results = []
    new_start_point = x_line_sm[int(0.2 / h)]

    start_whole_phi = time.perf_counter()
    for phi in np.arange(initial_phi, final_phi, step_phi):
        result = bisection_wrapper(
            phi=phi,
            initial_theta=initial_theta,
            b_mirror_target=b_mirror_target,
            K_target=K_target,
            start_point_sm=new_start_point,
            solar_params=solar_params,
            tol=0.01,
            max_iter=50
        )
        results.append(result)
    end_whole_phi = time.perf_counter()
    print(f"parallel время на все фи: {end_whole_phi - start_whole_phi:.6f}")

    thetas_foot = [] # список theta_foot, который будет заполняться для каждого phi
    phis = [] #список phi, который соответствует списку thetas_phi
    for result in results:
        #if result["x_shell"] is not None:
        #    file_candidates.write(
        #        f"phi={result["phi"]},theta={result["theta"]}\tK:{result["K"]}\tL:{result["L"]}\n")
        #    for counter in range(len(result["x_shell"])):
        #        file_candidates.write(
        #            f"{result["x_shell"][counter][0]}\t{result["x_shell"][counter][1]}\t{result["x_shell"][counter][2]}\t{result["B_shell"][counter]}\t{result["B(3)_SM"][counter][0]}\t{result["B(3)_SM"][counter][1]}\t{result["B(3)_SM"][counter][2]}\n")
        #    file_candidates.write("-------------------------------\n")
        #    file_candidates_full_trajectory.write(
        #        f"phi={result["phi"]},theta={result["theta"]}\tK:{result["K"]}\tL:{result["L"]}\n")
        #    for counter in range(0, len(result["x_shell_full"])):
        #        file_candidates_full_trajectory.write(
        #            f"{result["x_shell_full"][counter][0]}\t{result["x_shell_full"][counter][1]}\t{result["x_shell_full"][counter][2]}\t{result["B_shell_full"][counter]}\n")
        #    file_candidates.write("-------------------------------\n")

        if result["x_shell"] is not None:
            thetas_foot.append(result["theta_foot"])
            phis.append(result["phi"])


    model_params = (X1, X2, X3, X4, X5, X6, X7, ut, iy, mo, id, ro, v, bimf, dst, al)
    flux = calculate_flux_north(thetas_foot, phis, model_params)
    M = a2000_library.magnetic_moment(ut, id, mo, iy)
    L_star = 2 * np.pi * M / flux

    #file_candidates.close()
    #file_candidates_full_trajectory.close()
    return L_star


def calculate_flux_north(thetas_foot, phis, model_params):
    """
    Расчет магнитного потока через полярную область на севере, ограниченную орбитой дрейфа
    :param thetas_foot: массив углов theta подножья магнитных линий, градусы
    :param phis: массив углов phi, градусы
    :param model_params: параметры параболоидной модели
    :return: магнитный поток flux, Гс*м^2
    """
    X1, X2, X3, X4, X5, X6, X7, ut, iy, mo, id, ro, v, bimf, dst, al = model_params

    phis_rad = np.radians(phis)
    thetas_foot_rad = np.radians(thetas_foot)

    dphi = np.mean(np.diff(phis_rad))

    flux = 0.0
    for i, phi in enumerate(phis_rad):
        theta_foot = thetas_foot_rad[i]

        theta_points = np.linspace(0, theta_foot, num=300)
        dtheta = theta_points[1] - theta_points[0]

        sin_theta = np.sin(theta_points)
        cos_theta = np.cos(theta_points)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        x = sin_theta * cos_phi
        y = sin_theta * sin_phi
        z = cos_theta

        B = np.array([a2000_library.field(X1, X2, X3, X4, X5, X6, X7,
                                         ut, iy, mo, id, ro, v, bimf, dst, al,
                                         [x[j], y[j], z[j]]) for j in range(len(x))])

        B_dot_n = np.abs(B[:, 0] * x + B[:, 1] * y + B[:, 2] * z)

        integrand = B_dot_n * sin_theta
        flux += simpson(integrand, theta_points) * dphi

    return flux


def one_step_sat(data, i, model_input, K_target, mu_target, is_in_radbelt=True):
    """
    Функция полного расчета плотности в фазовом пространстве f и L_star для заданных mu,K. Результат - значение f и значение L_star
    :param data: dataframe с исходными данными со спутника и о солнечном ветре
    :param i: индекс обрабатываемой строки dataframe
    :param model_input: входные параметры модели магнитосферы
    :param K_target: заданное значение инварианта K
    :param mu_target: заданное значение инварианта mu
    :param is_in_radbelt: флаг о фильтрации данных: данные только из пересечения радиационного пояса или нет
    :return: результат расчетов, словарь:
        year,month,day,ut,f,l_star,energy,error
    """
    start_iteration = time.perf_counter()

    (X1, X2, X3, X4, X5, X6, X7, sl, h) = model_input
    ut, iy, mo, id = data["UT"][i], data["Year"][i], data["Month"][i], data["Day"][i]
    result = {
        "year": iy,
        "month": mo,
        "day": id,
        "ut": ut,
        "f": None,
        "l_star": None,
        "energy": None,
        "error": None
    }
    if "N/A" in data.iloc[i].astype(str).str.strip().values:  # пропускаем строки с "N/A"
        result["error"] = "SW data is N/A"
        return result

    ro, v, dst, al = float(data["Proton density"][i]), float(data["SW speed"][i]), float(data["Dst [nT]"][i]), float(
        data["AL-index"][i])
    bimf = [data["GSM B_x"][i], data["GSM B_y"][i], data["GSM B_z"][i]]
    B_Sat = data["B_nT"][i]
    x_sat = [data["X_GSM"][i], data["Y_GSM"][i], data["Z_GSM"][i]]

    if is_in_radbelt == True:
        branch = data["branch"][i]
        pass_id = data["pass_id"][i]
    else:
        branch = "None"
        pass_id = "None"

    result["branch"] = branch
    result["pass_id"] = pass_id

    x_sat_sm = a2000_library.gsm2sm(x_sat, ut, id, mo, iy)

    a2000_params = [X1, X2, X3, X4, X5, X6, X7, ut, iy, mo, id, ro, v, bimf, dst, al, x_sat, sl, h]
    initial_magnetic_line = a2000_library.trace_line(*a2000_params)
    x_line_gsm = initial_magnetic_line[:, 0:3]
    b_total = initial_magnetic_line[:, 3]
    x_line_sm = np.apply_along_axis(lambda x: a2000_library.gsm2sm(x, ut, id, mo, iy), axis=1, arr=x_line_gsm)

    K = []
    mirror_point_north_values = []
    mirror_point_south_values = []
    b_mirror_values = []
    flag_open_fieldline = 0
    for alpha in range(5, 90, 5):
        mirror_point_north, mirror_point_south, b_mirror = mirror_points_pitch_angle(x_line_sm, b_total, alpha,
                                                                                     x_sat_sm, equatorial=False)
        mirror_point_south_values.append(mirror_point_south)
        mirror_point_north_values.append(mirror_point_north)
        b_mirror_values.append(b_mirror)
        if mirror_point_south is None or mirror_point_north is None:
            flag_open_fieldline = 1
            break  # по идее считать дальше нет смысла, т.к. мы далее сразу получим return
        K.append(curve_integral(x_line_sm, b_total, mirror_point_north, mirror_point_south, b_mirror))
    if flag_open_fieldline == 1:
        result["error"] = "Initial line is open"
        return result

    pitch_angles = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85])

    cubic_interp_inverse = interpolation(np.flip(K), np.flip(pitch_angles))
    local_pitch_angle_target = cubic_interp_inverse(K_target)
    if (local_pitch_angle_target > 90) and all(k > K_target for k in K):
        result["error"] = "K > K_target, спутник не на экваторе"
        return result

    energy_target = kinetic_energy(mu_target, B_Sat, local_pitch_angle_target)
    result["energy"] = energy_target


    target_energies = (
    '33.0', '54.0', '80.0', '108.0', '143.0', '184.0', '226.0', '235.0', '346.0', '470.0', '597.0', '749.0', '909.0',
    '1064.0', '1079.0', '1575.0', '1728.0', '2280.0', '2619.0', '3618.0', '4062.0')
    valid_energies = []
    j_E_alphaK = []
    for energy in target_energies:
        energy_cols = [col for col in data.columns if col[0] == f'Energy_{energy}']
        df_energy = data[energy_cols]
        alpha_values = [float(col[1].split('_')[-1]) for col in df_energy.columns]
        j_alpha_values = df_energy.iloc[i].values

        j_alpha_values = np.array(j_alpha_values, dtype=float)
        alpha_values = np.array(alpha_values, dtype=float)
        mask = (j_alpha_values >= 0)
        j_alpha_values = j_alpha_values[mask]
        alpha_values = alpha_values[mask]

        j_alpha_fitted = fit_j_alpha(j_alpha_values, alpha_values, 1, energy)

        if j_alpha_fitted is None:  # некоторые данные содержат -1e31. такое пропускается
            continue

        j_E_alphaK.append(j_alpha_fitted(local_pitch_angle_target))
        valid_energies.append(float(energy))  # энергии, в которых нет пропусков данных

    j_E_alphaK = np.array(j_E_alphaK, dtype=float)
    energies_for_j = np.array(valid_energies, dtype=float)

    f_E = compute_psd(energies_for_j, j_E_alphaK)
    f_E_fitted = fit_f_E_1(energies_for_j, f_E)

    if f_E_fitted is not None:
        f = f_E_fitted(energy_target)
    else:
        f = "None"

    trace_shell_params = [X1, X2, X3, X4, X5, X6, X7, ut, iy, mo, id, ro, v, bimf, dst, al, sl, h]
    l_star = trace_drift_without_parallel(x_sat, trace_shell_params, K_target)

    result["l_star"] = l_star
    result["f"] = f

    end_iteration = time.perf_counter()
    #print(f"затрачено на одну итерацию {end_iteration - start_iteration:.6f}")

    return result


def field_animation(data):
    """
    Создание гиф-анимации среза магнитосферы в плоскости полдень-полночь по параболоидной модели A2000
    :param data: dataframe с данными о солнечном ветре (входные данные для модели)
    :return: функция ничего не возвращает, создает .gif файл с иллюстрацией поля
    """
    gif_frames = []
    for num in range(0, 20):
        if "N/A" in data.iloc[num].astype(str).str.strip().values:  # пропускаем строки с "N/A"
            continue
        ut, iy, mo, id = data["UT"][num], data["Year"][num], data["Month"][num], data["Day"][num]
        ro, v, dst, al = float(data["Proton density"][num]), float(data["SW speed"][num]), float(
            data["Dst [nT]"][num]), float(data["AL-index"][num])
        bimf = [data["GSM B_x"][num], data["GSM B_y"][num], data["GSM B_z"][num]]
        hh = ut2hms(ut)['hh']
        mm = ut2hms(ut)['mm']
        ss = ut2hms(ut)['ss']
        sl, h = 100, 0.1

        X1, X2, X3, X4, X5, X6, X7 = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0

        full_field = []
        start_points = []
        start_points_sm = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
        for start_point_sm in start_points_sm:
            if get_phi(start_point_sm) == 0:
                initial_lat = 60
            else:
                initial_lat = 50
            for lat in np.linspace(initial_lat, 90, int((90 - initial_lat) / 1)):
                theta = 90 - lat
                start_point_sm = set_theta(start_point_sm, np.radians(theta))
                start_points.append(start_point_sm)
                start_point_gsm = a2000_library.sm2gsm(start_point_sm, ut, id, mo, iy)
                a2000_params = [X1, X2, X3, X4, X5, X6, X7, ut, iy, mo, id, ro, v, bimf, dst, al, start_point_gsm, sl,
                                h]
                initial_magnetic_line = a2000_library.trace_line(*a2000_params)
                r2 = a2000_library.r2_calculation(ut, iy, mo, id, ro, v, bimf, dst, al)
                x_line_gsm = initial_magnetic_line[:, 0:3]
                b = initial_magnetic_line[:, 3]

                x_line_sm = np.apply_along_axis(lambda x: a2000_library.gsm2sm(x, ut, id, mo, iy), axis=1,
                                                arr=x_line_gsm)
                if np.argmin(b) == 0 or np.argmin(b) == len(b) - 1:
                    # открытые линии хвоста
                    linecolor = 'red'
                elif (np.linalg.norm(x_line_sm[np.argmin(b)]) > r2) and get_phi(start_point_sm) != 0:
                    # линии за передним краем токового слоя
                    linecolor = 'orange'
                else:
                    # замкнутные линии
                    linecolor = 'blue'
                result = {
                    'color': linecolor,
                    'line': x_line_sm
                }
                full_field.append(result)

        fig = plt.figure(figsize=(8, 4))
        for result in full_field:
            x = result['line'][:, 0]
            z = result['line'][:, 2]
            plt.plot(x, z, color=result['color'])

        plt.xlim(-40, 12)
        plt.ylim(-10, 20)
        plt.title(f"{id}.{mo}.{iy} {hh}:{mm}:{ss}")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)

        img = Image.open(buf)
        gif_frames.append(img.copy())  # СОХРАНЯЕМ КОПИЮ

        img.close()
        buf.close()
        plt.close(fig)

    if gif_frames:
        gif_frames[0].save(
            'magnetic_lines_animation.gif',
            format='GIF',
            append_images=gif_frames[1:],
            save_all=True,
            duration=300,  # Длительность кадра в миллисекундах
            loop=0  # Бесконечный цикл
        )


def ut2hms(ut):
    total_seconds = round(ut * 3600)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return {
        'hh': f"{hours:02d}",
        'mm': f"{minutes:02d}",
        'ss': f"{seconds:02d}"
    }


def main():

    raw_data = read_van_allen_SW_data_2files("test_data\\RBSPA_REL04_ECT-MAGEIS-L3_10-11.10.2017.csv",
                                             "test_data\\sw_data 10-11.10.2017.txt",
                                             "test_data\\RBSP-A_MAGNETOMETER_1SEC-GSM_10-11.10.2017.csv")

    K_target = 700
    mu_target = 100


    data = find_passes(raw_data,branch='both')

    file_result = open("result.txt", "w")
    file_result.write(f"Year\tMonth\tDay\tUT\tL_star\tf\tEnergy\tbranch\tpass_id\n")

    X1, X2, X3, X4, X5, X6, X7 = 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    sl, h = 100, 0.01
    model_sources = [X1, X2, X3, X4, X5, X6, X7, sl, h]

    results = []
    start_whole_program = time.perf_counter()
    with ProcessPoolExecutor() as executor:
        futures = []
        for i in range(0, 24, 12):
            futures.append(executor.submit(one_step_sat,
                                           data=data,
                                           i=i,
                                           model_input=model_sources,
                                           K_target=K_target,
                                           mu_target=mu_target,
                                           is_in_radbelt=True
                                           ))
        for future in futures:
            result = future.result()
            results.append(result)

    for result in results:
        if result["error"] is None:
            file_result.write(
                f"{result["year"]}\t{result["month"]}\t{result["day"]}\t{result["ut"]}\t{result["l_star"]}\t{result["f"]}\t{result["energy"]}\t{result["branch"]}\t{result["pass_id"]}\n")
        else:
            file_result.write(
                f"{result["year"]}\t{result["month"]}\t{result["day"]}\t{result["ut"]}\tN/A\tN/A\tN/A\tN/A\tN/A\t{result["error"]}\n")

    end_whole_program = time.perf_counter()
    #print(f"время на всю программу: {end_whole_program - start_whole_program:.6f}")

if __name__ == "__main__":
    main()