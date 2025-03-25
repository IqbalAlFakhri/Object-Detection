import math

def calculate_distance(x_left, x_right, t = 11.0, x = 640, y = 110.0):
    """
    Parameters:
        t (float): Lebar baseline (jarak antara dua kamera) dalam cm
        x (float): Perbedaan koordinat x antara kamera kiri dan kanan
        y (float): Sudut bidang pandang (FOV) dalam derajat

    Returns:
        float: Perkiraan jarak objek dalam cm
    """
    # Konversi sudut dari derajat ke radian
    y_radian = math.radians(y)

    # Perhitungan jarak dengan rumus
    try:
        disparity = x_left - x_right  # Hitung disparitas
        if disparity == 0:
            return float('inf')  # Hindari pembagian oleh nol

        distance = (t * x) / (2 * math.tan(y_radian / 2) * (disparity))
    except ZeroDivisionError:
        distance = float('inf')  # Jika pembagian dengan nol, jarak dianggap tak terhingga

    # #optimasi kamera main-code-build.py
    # real_distance = 10
    #
    # if distance <= 5:
    #     distance = "jarak terlalu dekat"
    # else:
    #     distance = real_distance + (((distance - 5)/3)*5)

    #optimasi kamera main-code-undistort.py
    real_distance = 20

    if distance <= 5:
        distance = "jarak terlalu dekat"
    elif distance < 13:
        distance = real_distance + (((distance - 5)/2)*5)
    else:
        distance = real_distance + (((distance - 5) / 2) * 5) + 1

    return distance

