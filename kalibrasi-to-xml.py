import numpy as np
import cv2 as cv
import glob

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################
# Untuk papan catur dengan 8x8 kotak, jumlah inner corners adalah 7x7.
chessboardSize = (7, 7)
# Resolusi frame sesuai tampilan (640 x 480)
frameSize = (640, 480)

# Termination criteria untuk cornerSubPix
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Siapkan object points: (0,0,0), (1,0,0), ... lalu kalikan dengan ukuran kotak (2 cm)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
objp = objp * 2  # Setiap kotak berukuran 2 cm

# Array untuk menyimpan object points dan image points dari semua gambar
objpoints = []  # 3D points di dunia nyata
imgpointsL = []  # 2D points untuk kamera kiri
imgpointsR = []  # 2D points untuk kamera kanan

# Ambil daftar file gambar untuk masing-masing kamera
images_L = glob.glob('images_calib/left/*.png')
images_R = glob.glob('images_calib/right/*.png')

for imgLeftPath, imgRightPath in zip(images_L, images_R):
    img_L = cv.imread(imgLeftPath)
    img_R = cv.imread(imgRightPath)

    # Ubah ukuran gambar sesuai frameSize (jika belum sesuai)
    img_L = cv.resize(img_L, frameSize)
    img_R = cv.resize(img_R, frameSize)

    grayL = cv.cvtColor(img_L, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(img_R, cv.COLOR_BGR2GRAY)

    # Cari sudut papan catur pada masing-masing gambar
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)

    # Jika ditemukan di kedua gambar, perbaiki posisi sudut dan simpan data
    if retL and retR:
        objpoints.append(objp)

        # Refinement sudut
        cornersL = cv.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        imgpointsL.append(cornersL)

        cornersR = cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
        imgpointsR.append(cornersR)

        # Tampilkan hasil deteksi untuk verifikasi (opsional)
        cv.drawChessboardCorners(img_L, chessboardSize, cornersL, retL)
        cv.imshow('Chessboard Left', img_L)
        cv.drawChessboardCorners(img_R, chessboardSize, cornersR, retR)
        cv.imshow('Chessboard Right', img_R)
        cv.waitKey(500)  # tampilkan selama 500 ms

cv.destroyAllWindows()

############## KALIBRASI KAMERA INDIVIDUAL #######################################################

# Kalibrasi untuk kamera kiri
retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, frameSize, 1, frameSize)

# Kalibrasi untuk kamera kanan
retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, frameSize, 1, frameSize)

########## KALIBRASI STEREO #############################################################
# Gunakan flag CALIB_FIX_INTRINSIC untuk menjaga parameter intrinsik yang telah dihitung
flags = cv.CALIB_FIX_INTRINSIC
criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Stereo calibration untuk menentukan rotasi, translasi, essential dan fundamental matrix
retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR,
    newCameraMatrixL, distL,
    newCameraMatrixR, distR,
    frameSize,
    criteria_stereo, flags)

########## STEREO RECTIFICATION #########################################################
# rectifyScale: 0=crop citra, 1=menampilkan seluruh citra
rectifyScale = 1
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv.stereoRectify(
    newCameraMatrixL, distL, newCameraMatrixR, distR,
    frameSize, rot, trans, rectifyScale, (0, 0))

# Inisialisasi peta remapping untuk masing-masing kamera
stereoMapL = cv.initUndistortRectifyMap(
    newCameraMatrixL, distL, rectL, projMatrixL, frameSize, cv.CV_16SC2)
stereoMapR = cv.initUndistortRectifyMap(
    newCameraMatrixR, distR, rectR, projMatrixR, frameSize, cv.CV_16SC2)

# Simpan parameter stereo map ke file XML
print("Saving parameters!")
cv_file = cv.FileStorage('stereoMap-1.xml', cv.FILE_STORAGE_WRITE)
cv_file.write('stereoMapL_x', stereoMapL[0])
cv_file.write('stereoMapL_y', stereoMapL[1])
cv_file.write('stereoMapR_x', stereoMapR[0])
cv_file.write('stereoMapR_y', stereoMapR[1])
cv_file.release()

print("Calibration complete and stereo map saved to stereoMap-1.xml")
