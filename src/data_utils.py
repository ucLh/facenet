import os
import shutil
import re


def remove(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.isfile(path):
        os.remove(path)
    else:
        print('File {} does not exist.'.format(path))


def preprocess_queries_data():
    IMAGE_DIR = 'series'
    images = os.listdir(IMAGE_DIR)
    image_paths = [IMAGE_DIR + '/' + x for x in images]
    folders = list(filter(os.path.isdir, image_paths))
    paths_and_names = zip(folders, images)
    for x in paths_and_names:
        print(x)
        folder = x[0]
        name = x[1]
        sub_files = os.listdir(folder)
        for file in sub_files:
            match_obj = re.match(r'[\w-]+\.jpg|[\w-+]\.img|[\w-+]\.jpeg|[\w-]+\.JPG', file)
            if match_obj is None:
                remove(folder + '/' + file)


def preprocess_series_data():
    IMAGE_DIR = 'series'
    images = os.listdir(IMAGE_DIR)
    images = [IMAGE_DIR + '/' + x for x in images]
    folders = filter(os.path.isdir, images)
    for folder in folders:
        print(folder)
        # remove(folder + '/orb')
        # remove(folder + '/0')
        # remove(folder + '/1')
        # remove(folder + '/database.db')
        # remove(folder + '/reconstruction_log.txt')
        # remove(folder + '/objects')
        # remove(folder + '/old')
        os.system('mv ' + folder + '/old')

        # sub_images = os.listdir(folder + '/images')
        # for image in sub_images:
        #    os.rename(folder + '/images/' + image, folder + '/' + image)
        # os.rmdir(folder + '/images')


def move_needed_classes():
    # needed_classes_dir = 'queries'
    classes_dir = '/media/topcon/Buildings/current_train'
    # needed_classes = os.listdir(needed_classes_dir)
    classes = os.listdir(classes_dir)
    # needed_classes = list(map(lambda c: c.split('__')[0], needed_classes))
    needed_classes = ['Spb_konnush_05_church', 'Peterhof_Rest_cafeGrandOrangery', 'Peterhof_Shop_Gradusi',
                      'Peterhof_Rest_Kladovaya', 'Peterhof_NavalPolytechnicInstitutePopova', 'Spb_konnush_06_build14',
                      'Peterhof_CentralResearchInstitute', 'Peterhof_Rest_Kladovaya', 'Peterhof_Rest_cafeVena',
                      'Peterhof_Museum_Benua_first', 'Peterhof_Stable', 'Peterhof_Shop_Dixi', 'Bari_4cityHall',
                      'Lomonosov_Temple_SideDoor', 'Spb_konnush_07_jazbar48', 'Peterhof_Museum_MonPlesir_side',
                      'Peterhof_CentralResearchInstitute', 'Peterhof_Stable_arch', 'Copenhagen_1_rest_McJoys',
                      'Peterhof_Stable_second', 'Peterhof_Shop_Gradusi', 'Peterhof_Office', 'Peterhof_Balneary',
                      'Peterhof_DrinkingPumpRoom', 'Peterhof_Shop_Dixi', 'Peterhof_Rest_Kladovaya',
                      'Spb_vo_office_cafe_doors', 'helsinki_2_apteeki', 'Bari_1officeDoor', 'Peterhof_Office',
                      'Spb_vo_office_cafe_doors', 'tver_ozernaya1', 'Peterhof_Museum_outhouse',
                      'Spb_bolshaya-morskaya_15', 'Spb_konnush_14_library', 'Peterhof_Stable_second',
                      'Peterhof_Museum_Zaks', 'Spb_HealthCenter', 'Peterhof_NavalPolytechnicInstitutePopova',
                      'Peterhof_CinemaAvrora', 'Peterhof_Museum_MonPlesir_bathhouse',
                      'Peterhof_CentralResearchInstitute', 'Peterhof_Shop_Gradusi', 'Spb_vo_office_cafe_doors',
                      'Spb_bolshaya-morskaya_15', 'Bari_3Rest', 'Spb_vo_office_cafe_doors',
                      'Peterhof_Museum_MonPlesir_kitchen', 'Peterhof_Office', 'Peterhof_Rest_CafeLove',
                      'Peterhof_DrinkingPumpRoom', 'Spb_Rest_TaurideGarden', 'helsinki_1_andante',
                      'Spb_konnush_14_library', 'Spb_konnush_10_kiton', 'Peterhof_Museum_Benua_first',
                      'Spb_vo_office_cafe_doors', 'Peterhof_Balneary', 'Peterhof_Balneary', 'Peterhof_Stable_goal',
                      'Peterhof_Museum_TowerOfTtheCavalryHouse', 'Spb_konnush_09_prada', 'Murino_bodyboom_day',
                      'tver_officeOut2', 'Bari_1officeDoor', 'Spb_konnush_11_dlt', 'Peterhof_Shop_Dixi',
                      'helsinki_3_usadba', 'Peterhof_Shop_Dixi', 'Peterhof_Office', 'Spb_konnush_12_pyshki',
                      'Peterhof_Rest_XL', 'Peterhof_CinemaAvrora', 'Peterhof_DrinkingPumpRoom', 'Peterhof_Office',
                      'Peterhof_Rest_cafeConditerskay', 'Spb_vo_office_orange_room', 'Peterhof_Stable_goal',
                      'Peterhof_Rest_XL', 'Spb_vo_office_cafe_doors', 'Spb_bolshaya-morskaya_15',
                      'Peterhof_CinemaCaskad_facade', 'Peterhof_Stable', 'Peterhof_NavalPolytechnicInstitutePopova',
                      'Spb_bolshaya-morskaya_15', 'Lomonosov_Museum_PeterPalace', 'Peterhof_Shop_Dixi',
                      'Spb_vo_office_cafe_doors', 'Peterhof_Shop_Dixi', 'Spb_levashovskiy_20', 'Spb_konnush_12_pyshki',
                      'Peterhof_Rest_Cassik', 'Bari_2officeBackDoor', 'Peterhof_Shop_Dixi', 'Spb_bolshaya-morskaya_15',
                      'Peterhof_Rest_AlexHaus', 'Peterhof_Shop_Dixi', 'Peterhof_StationRailway',
                      'Peterhof_Stable_second', 'Peterhof_Office', 'Bari_2officeBackDoor', 'Peterhof_Stable',
                      'Peterhof_Rest_cafeGrandOrangery', 'Peterhof_Stable_goal', 'Peterhof_CinemaCaskad_side_entrance',
                      'Peterhof_Shop_Dixi', 'Peterhof_Shop_Dixi', 'Peterhof_Rest_Kladovaya', 'Peterhof_Stable',
                      'Spb_vo_office_cafe_doors', 'Spb_konnush_08_frLavka', 'Spb_konnush_05_church',
                      'Peterhof_Rest_cafeVena', 'Spb_TaurideGarden', 'Peterhof_Shop_Amakids',
                      'Peterhof_Museum_MonPlesir_side', 'Peterhof_Museum_MonPlesir_bathhouse', 'Spb_konnush_12_pyshki',
                      'Peterhof_Shop_Dixi', 'Bari_1officeDoor', 'Peterhof_Rest_XL']

    for class_ in classes:
        tmp = class_.split('__')[0]
        if tmp in needed_classes:
            print('mv ' + classes_dir + '/' + class_ + ' /media/topcon/Buildings/series_for_test/')
            os.system('mv ' + classes_dir + '/' + class_ + ' /media/topcon/Buildings/series_for_test/')


def get_class(dir_name):
    return dir_name.split('__')[0]


def get_folder2label_dict(image_dir):
    image_names = os.listdir(image_dir)
    # image_names = list(map(get_class, image_names))
    image_names.sort()

    labels_dict = {image_names[i]: i for i in range(len(image_names))}

    return labels_dict


def get_needed_labels(folders_list, labels_dict):
    result = []
    result1 = []
    for k in labels_dict.keys():
        if get_class(k) in folders_list:
            result.append((labels_dict[k], k))
            result1.append(labels_dict[k])

    return result1, result


move_needed_classes()
