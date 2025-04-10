# Chesschange
#  카메라 캘리브레이션 및 렌즈 왜곡 보정

 **체스보드 영상**을 이용하여 카메라를 캘리브레이션하고,  
얻어진 내부 파라미터를 바탕으로 **렌즈 왜곡 보정**을 수행합니다.

---

##  1. 카메라 캘리브레이션 (Camera Calibration)

- 캘리브레이션에 사용된 영상: `chessboard.avi`
- OpenCV의 `cv2.findChessboardCorners()`와 `cv2.calibrateCamera()`를 이용해 캘리브레이션 수행

###  카메라 내부 파라미터 (Camera Matrix)

[ [673.42682005, 0.0, 245.84823196], [0.0, 670.85149225, 351.89660938], [0.0, 0.0, 1.0] ]


###  왜곡 계수 (Distortion Coefficients)

[ 0.14594799, -0.91663695, -0.00058594, 0.00784773, 2.09888954 ]


###  평균 재투영 오차 (RMSE)

0.04546833059292765


---

##  2. 렌즈 왜곡 보정 (Distortion Correction)

- 위에서 구한 파라미터를 기반으로 왜곡 보정 수행
- 적용 대상:
  - `chess.mp4` → `undistorted_output.avi`

###  보정 전/후 비교 이미지

| 원본 영상                   | 왜곡 보정 영상                        |
|-------------------------|---------------------------------|
| ![원본](./data/chess.mp4) | ![보정](./undistorted_output.avi) |


