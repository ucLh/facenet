from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import math


def get_exif(filename):
    image = Image.open(filename)
    image.verify()
    return image._getexif()


def get_labeled_exif(exif):
    labeled = {}
    for (key, val) in exif.items():
        labeled[TAGS.get(key)] = val

    return labeled


def get_geotagging(exif):
    if not exif:
        raise ValueError("No EXIF metadata found")

    geotagging = {}
    for (idx, tag) in TAGS.items():
        if tag == 'GPSInfo':
            if idx not in exif:
                raise ValueError("No EXIF geotagging found")

            for (key, val) in GPSTAGS.items():
                if key in exif[idx]:
                    geotagging[val] = exif[idx][key]

    return geotagging


def get_decimal_from_dms(dms, ref):

    degrees = dms[0][0] / dms[0][1]
    minutes = dms[1][0] / dms[1][1] / 60.0
    seconds = dms[2][0] / dms[2][1] / 3600.0

    if ref in ['S', 'W']:
        degrees = -degrees
        minutes = -minutes
        seconds = -seconds

    return round(degrees + minutes + seconds, 5)


def get_coordinates(geotags):
    lat = get_decimal_from_dms(geotags['GPSLatitude'], geotags['GPSLatitudeRef'])

    lon = get_decimal_from_dms(geotags['GPSLongitude'], geotags['GPSLongitudeRef'])

    return lat, lon


def check_coords_in_radius(center, query, dist=0.2):
    """
    :param center: A tuple of latitude and longitude of the center of coordinates circle
    :param query: A tuple of latitude and longitude for a query coordinates
    :param dist: In kilometers
    :return:
    """
    lat_center, lon_center = center
    lat_querie, lon_querie = query
    lon1 = lon_center - dist / abs(math.cos(math.radians(lat_center)) * 111.0)  # 1 градус широты = 111 км
    lon2 = lon_center + dist / abs(math.cos(math.radians(lat_center)) * 111.0)
    lat1 = lat_center - (dist / 111.0)
    lat2 = lat_center + (dist / 111.0)

    return (lat1 <= lat_querie <= lat2) and (lon1 <= lon_querie <= lon2)


def check_image_in_radius(center_img_path, query_img_path, dist=0.2):
    try:
        exif_center = get_exif(center_img_path)
        geotags_c = get_coordinates(get_geotagging(exif_center))

        exif_query = get_exif(query_img_path)
        geotags_q = get_coordinates(get_geotagging(exif_query))

        return check_coords_in_radius(geotags_c, geotags_q, dist)
    except :
        return True


def get_center(center_img_path, dist=0.2):
    exif_center = get_exif(center_img_path)
    geotags_c = get_coordinates(get_geotagging(exif_center))
    lat_center, lon_center = geotags_c

    lon1 = lon_center - dist / abs(math.cos(math.radians(lat_center)) * 111.0)  # 1 градус широты = 111 км
    lon2 = lon_center + dist / abs(math.cos(math.radians(lat_center)) * 111.0)
    lat1 = lat_center - (dist / 111.0)
    lat2 = lat_center + (dist / 111.0)

    return lon1, lon2, lat1, lat2

center_img = '../datasets/series_for_test/Peterhof_Balneary__1_GalaxyJ5_8/20181219_141314.jpg'
query_img = '../datasets/series_for_test/Peterhof_CinemaAvrora__GalaxyJ5_8/20181225_161515.jpg'

print(check_image_in_radius(center_img, query_img))
