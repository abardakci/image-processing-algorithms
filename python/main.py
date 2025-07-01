import cv2
import numpy as np

assets_path = "/home/alper/Code_Repo/ImageProcessing C++/assets/";

def main1():

    img = cv2.imread("len_full.jpg")
    if img is None:
        raise ValueError("img load error")

    region_size = 30   
    ruler = 10.0       
    num_iterations = 10

    slic = cv2.ximgproc.createSuperpixelSLIC(img, algorithm=cv2.ximgproc.SLICO,
                                                region_size=region_size, ruler=ruler)

    slic.iterate(num_iterations)

    slic.enforceLabelConnectivity()

    mask = slic.getLabelContourMask()
    labels = slic.getLabels()
    number_of_superpixels = slic.getNumberOfSuperpixels()

    output = img.copy()
    output[mask == 255] = [0, 0, 255]

    cv2.imshow("SLIC Superpixels", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    img = cv2.imread(assets_path + "lee.jpg")
    height, width = img.shape[:2]

    img = cv2.resize(img, (width//2, height//2))

    slic = cv2.ximgproc.createSuperpixelSLIC(img, algorithm=cv2.ximgproc.SLIC, region_size=10, ruler=20.0)
    slic.iterate(10)
    slic.enforceLabelConnectivity()

    labels = slic.getLabels()
    num_labels = slic.getNumberOfSuperpixels()

    output = np.zeros_like(img, dtype=np.uint8)

    for label in range(num_labels):
        mask = (labels == label)
        mean_color = img[mask].mean(axis=0).astype(np.uint8)
        output[mask] = mean_color

    # mask_contour = slic.getLabelContourMask()
    # output[mask_contour == 255] = [0, 0, 255]

    cv2.imshow("Colored sp", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
