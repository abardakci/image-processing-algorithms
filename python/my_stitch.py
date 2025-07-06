import cv2
import numpy as np

# --- 1. Görselleri yükle ---
img1 = cv2.imread("/home/alper/code-repo/image_processing_cpp/python/imges/goldengate-03.png")  # sol görüntü
img2 = cv2.imread("/home/alper/code-repo/image_processing_cpp/python/imges/goldengate-02.png")  # sağ görüntü

# --- 2. Özellik çıkarımı ---
# Alternatifler: cv2.ORB_create(), cv2.SURF_create(), cv2.SIFT_create()
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# --- 3. Eşleştirme ---
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# --- 4. Lowe’s Ratio Test (outlier temizleme) ---
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

# Eşleşme sayısı yeterli mi?
if len(good) < 4:
    raise Exception("Yeterli eşleşme yok!")

# --- 5. Eşleşen noktaları çıkar ---
src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

# --- 6. Homografi matrisi (RANSAC ile) ---
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# --- 7. Warping (image1 → image2 üzerine) ---
height1, width1 = img1.shape[:2]
height2, width2 = img2.shape[:2]

# image1 warp edilip geniş bir tuval üstüne oturtulacak
result_width = width1 + width2
result = cv2.warpPerspective(img1, H, (result_width, height1))
result[0:height2, 0:width2] = img2  # image2'yi üzerine yerleştir

# --- 8. Sonuç göster ---
cv2.imshow("Panorama", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
