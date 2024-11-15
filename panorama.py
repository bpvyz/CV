import cv2
import matplotlib.pyplot as plt
import numpy as np

MIN_MATCH_COUNT = 10

def load_image_gray(file_path):
    return cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2GRAY)

def load_image_rgb(file_path):
    return cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)


def warp2(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)

    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax - xmin, ymax - ymin))
    result[t[1]:h1 + t[1], t[0]:w1 + t[0]] = img1
    return result


def compute_homography(image1, image2, sift, flann):
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)

    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return homography, kp1, kp2, good_matches

def main():
    sift = cv2.SIFT_create()
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    images_gray = [
        load_image_gray("MaterijalLV3/1.JPG"),
        load_image_gray("MaterijalLV3/2.JPG"),
        load_image_gray("MaterijalLV3/3.JPG")
    ]

    images_rgb = [
        load_image_rgb("MaterijalLV3/1.JPG"),
        load_image_rgb("MaterijalLV3/2.JPG"),
        load_image_rgb("MaterijalLV3/3.JPG")
    ]

    for i, image in enumerate(images_rgb):
        plt.subplot(2, 3, i + 1)
        plt.imshow(image)
        plt.title(f"Original rgb {i + 1}")
    for i, image in enumerate(images_gray):
        plt.subplot(2, 3, 3 + i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f"Original grayscale {i + 1}")
    plt.show()

    homography_12, kp1, kp2, good_matches_12 = compute_homography(images_gray[0], images_gray[1], sift, flann)
    homography_23, kp2, kp3, good_matches_23 = compute_homography(images_gray[1], images_gray[2], sift, flann)

    match_image_12 = cv2.drawMatches(images_gray[0], kp1, images_gray[1], kp2, good_matches_12, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    match_image_23 = cv2.drawMatches(images_gray[1], kp2, images_gray[2], kp3, good_matches_23, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.subplot(2, 1, 1)
    plt.imshow(match_image_12)
    plt.title("Karakteristične tačke između slike 1 i 2")

    plt.subplot(2, 1, 2)
    plt.imshow(match_image_23)
    plt.title("Karakteristične tačke između slike 2 i 3")
    plt.show()

    plt.show()

    stitched_image_12 = warp2(images_rgb[1], images_rgb[0], homography_12)

    plt.imshow(stitched_image_12)
    plt.title("Spojena slika 1 i 2")
    plt.show()

    homography_123, kp12, kp23, good_matches_123 = compute_homography(stitched_image_12, images_rgb[2], sift, flann)
    match_image_123 = cv2.drawMatches(stitched_image_12, kp12, images_rgb[2], kp23, good_matches_123, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.imshow(match_image_123)
    plt.title("Karakteristične tačke između slike 1-2 i 3")
    plt.show()

    final_stitched_image = warp2(images_rgb[2], stitched_image_12, homography_123)

    plt.imshow(final_stitched_image)
    plt.title("Konačno spojena slika")
    plt.show()

if __name__ == "__main__":
    main()
